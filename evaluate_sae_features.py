# Automated evaluation of SAE features for a list of subjects.
#
# stage 1 ("base"): for every base prompt x base subject, generate one image
#   with sdxl-turbo, remember its seed/prompt, and SAM3-mask the base subject.
#   Everything lands in {out_dir}/base/manifest.json and is reused on later
#   runs - an entry is only (re)generated if its image or mask is missing.
#
# stage 2 ("discover", generate_clean_inference.py-style): for every subject,
#   fill each dream prompt's <sks> with it, generate + cache the block
#   activations in the same run_with_cache pass, encode them with each
#   block's SAE (stored compactly as top-k indices/values), SAM3-mask the
#   subject, then fit per-latent 1D ridge logistic probes on that subject's
#   own images only (SAM3 subject patches = positives, every other patch of
#   those same images = negatives) and keep both the
#   lowest-BCE latent and the highest-F1 latent. Their BCE, loss explained
#   (1 - bce / constant-predictor bce), F1, precision, recall and activation
#   stats go in {out_dir}/features/{subject}.json.
#
# stage 3 ("ablate", generate_clean_ablate.py-style): for every block and
#   subject, regenerate each base image with the same seed/prompt while
#   adding the chosen latent's decoder direction * value * strength inside
#   the base subject's mask (one run if bce and f1 picked the same latent).
#   A random latent per block is injected the same way as a control.
#   Each edited image is then scored:
#     - SAM3 mask of the subject on the edited image vs the original mask
#       (IoU, precision, recall) and SAM3's confidence
#     - VQAScore (vqa_scorer.py, Qwen2.5-VL) that the subject is in
#       the image, with the unedited base image as the baseline
#     - whether the base subject is gone (SAM3 + VQAScore on it)
#     - background preservation outside the mask (PSNR) and how much the
#       masked region changed
#     - CLIPScore (vqa_scorer.py, transformers CLIP) for the subject
#
# stage 4 ("remove"): for every block and subject, regenerate each of that
#   subject's dream images (same subject-filled prompt + seed) while setting
#   the chosen latent to 0 at every patch (no mask) and leaving the rest of
#   the block's activation untouched, i.e. subtracting that latent's decoded
#   contribution. Random latents are zeroed the same way as a control. SAM3
#   then looks for the subject on the result; "removed" means it finds
#   nothing. The removal rate is taken over dream images where SAM3 did find
#   the subject originally. With --aux_prompt_file, every auxiliary prompt
#   (subject filled in only where it has the placeholder) is generated once
#   unedited and once per removal the same way, so a feature can be checked on
#   prompts other than the dream ones (e.g. unsafe prompts). With --use_nsfw,
#   the LAION CLIP-based NSFW classifier (score_words.NSFWScorer) scores every
#   original and removal image, so e.g. a nudity feature's removal can be
#   judged by how much the NSFW probability drops. Rows go to
#   {out_dir}/removal_results.csv, {out_dir}/summary_removal.csv and
#   {outputs_dir}/removal_results.csv.
#
#   Per-image rows go to {out_dir}/results.csv and grouped means to
#   {out_dir}/summary_*.csv. Every metric averaged per subject x block x
#   feature choice x strength goes to {outputs_dir}/results.csv, merged
#   across runs and subject shards (keyed by out_dir + subject).
#
# Model use is split into passes (generate -> SAM3 -> VQA -> CLIP) so only
# one big model is on the GPU at a time. Every pass skips work whose output
# file already exists, so the script can be rerun or sharded by subject
# with --subject_list.

import os
import json
import time
import gc

import numpy as np
import pandas as pd
import torch
from PIL import Image

from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from experiment_helpers.init_helpers import default_parser, repo_api_init
from experiment_helpers.image_helpers import concat_images_horizontally, concat_images_vertically

from attribution import DEFAULT_BLOCK_LIST
from sdxl_pipe import HookedStableDiffusionXLWithUNetPipeline
from sam3_repo.sam3.model_builder import build_sam3_image_model
from sam3_repo.sam3.model.sam3_image_processor import Sam3Processor

from generate_clean_swap import load_sae, load_feature_mean, highlight_pixel_mask
from generate_clean_inference import read_prompts, resize_mask_to_grid
from generate_clean_ablate import make_add_position_hook_dict
from sparse_probe import select_bce_and_f1

parser = default_parser({"repo_id": "jlbaker361/nsfw"})

parser.add_argument("--base_prompt_file", type=str, default="prompt_dir/base_prompts.txt")
parser.add_argument("--base_subject_file", type=str, default="prompt_dir/base_subjects.txt")
parser.add_argument("--subject_file", type=str, default="prompt_dir/subjects.txt")
parser.add_argument("--dream_prompt_file", type=str, default="prompt_dir/dream_prompts.txt")
parser.add_argument("--subject_list", nargs="*", default=None,
                    help="overrides --subject_file, e.g. to shard subjects across jobs")
parser.add_argument("--placeholder", type=str, default="<sks>",
                    help="token replaced by the subject; templates without it get the subject appended")
parser.add_argument("--out_dir", type=str, default="evaluation/sae_eval")
parser.add_argument("--outputs_dir", type=str, default="evaluation/outputs",
                    help="folder for results.csv: every metric per subject x block, merged across runs")

parser.add_argument("--sae_source", type=str, default="local", choices=["local", "saeuron"])
parser.add_argument("--block_list", nargs="*", default=None)
parser.add_argument("--mode", type=str, default="diff", choices=["diff", "out"],
                    help="what the SAE encodes: block output minus input ('diff') or the raw output")

parser.add_argument("--seed", type=int, default=0, help="base image i uses seed + i")
parser.add_argument("--dream_seed", type=int, default=1000, help="dream prompt j uses dream_seed + j")
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--size", type=int, default=512)

parser.add_argument("--bce_ridge", type=float, default=1e-8)
parser.add_argument("--bce_newton_steps", type=int, default=30)

parser.add_argument("--strength_list", nargs="*", type=float, default=[10.0])
parser.add_argument("--inject_value", type=str, default="checkpoint_mean", choices=["checkpoint_mean", "pos_mean"],
                    help="activation placed on the latent before * strength: the SAE checkpoint's mean.pt "
                         "(same as generate_clean_ablate.py) or its mean over the subject's positive patches")
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--end_step", type=int, default=1000)
parser.add_argument("--n_random_controls", type=int, default=1,
                    help="random latents per block injected as a baseline")

parser.add_argument("--aux_prompt_file", type=str, default=None,
                    help="extra prompts for stage 4: a .txt (one per line) or a .csv with a 'prompt' column "
                         "(e.g. unsafe.csv). Each is generated unedited and with every removal")
parser.add_argument("--aux_limit", type=int, default=0, help="only use the first N aux prompts (0 = all)")
parser.add_argument("--aux_seed", type=int, default=2000, help="aux prompt k uses aux_seed + k")
parser.add_argument("--use_nsfw", action="store_true",
                    help="score stage 4 originals and removals with the LAION CLIP-based NSFW classifier")
parser.add_argument("--nsfw_threshold", type=float, default=0.5,
                    help="NSFW probability at or above which an image counts as flagged")

parser.add_argument("--vqa_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                    help="Hugging Face Qwen2.5-VL model for vqa_scorer.VQAScore "
                         "(7B needs ~17 GB of GPU memory)")
parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14",
                    help="Hugging Face CLIP model for vqa_scorer.CLIPScore; ignored with --disable_clip")
parser.add_argument("--text_template", type=str, default="a photo of a {}",
                    help="text scored by VQAScore/CLIPScore for a subject")
parser.add_argument("--score_batch_size", type=int, default=16)

for flag in ["base", "dream_generate", "dream_sparsify", "dream_masks", "discover",
             "ablate_generate", "ablate_masks", "vqa", "clip", "summary",
             "remove_generate", "remove_masks", "remove_summary"]:
    parser.add_argument(f"--disable_{flag}", action="store_true")


# ---------------------------------------------------------------- helpers

def safe(s: str) -> str:
    return s.strip().replace(" ", "_").replace("/", "_")


def fill_prompt(template: str, subject: str, placeholder: str) -> str:
    if placeholder in template:
        text = template.replace(placeholder, subject)
    else:
        text = f"{template.strip()} {subject}"
    return " ".join(text.split())


def read_lines(path: str) -> list:
    # read_prompts strips blank lines; keep the order of the file
    return read_prompts(path)


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def is_cuda(device) -> bool:
    return device == "cuda" or (hasattr(device, "type") and device.type == "cuda")


class Models:
    '''
    Lazily loads the one big model a pass needs and frees the others first,
    so SDXL, SAM3 and the VQA model never share the GPU.
    '''

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.pipe = None
        self.sam = None
        self.vqa = None
        self.clip = None
        self.nsfw = None
        self.saes = {}

    def free(self, keep: str = None):
        # Move to CPU before dropping, so a stray reference elsewhere
        # can't keep the weights on the GPU.
        for name in ["pipe", "sam", "vqa", "clip", "nsfw"]:
            obj = getattr(self, name)
            if name != keep and obj is not None:
                (obj if name == "pipe" else obj.model).to("cpu")
                setattr(self, name, None)
        if keep not in ("pipe", "sae"):
            for sae in self.saes.values():
                sae.to("cpu")
            self.saes = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_pipe(self):
        self.free(keep="pipe")
        if self.pipe is None:
            dtype = torch.float16 if (torch.cuda.is_available() and self.args.mixed_precision == "fp16") else torch.float32
            pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
                "stabilityai/sdxl-turbo", torch_dtype=dtype,
                variant=("fp16" if dtype == torch.float16 else None),
            )
            pipe.pipe.set_progress_bar_config(disable=True)
            pipe.to(self.device)
            self.pipe = pipe
        return self.pipe

    def get_sae(self, block: str):
        if block not in self.saes:
            self.saes[block] = load_sae(block, self.args.sae_source).to(self.device)
        return self.saes[block]

    def get_sam(self):
        self.free(keep="sam")
        if self.sam is None:
            if is_cuda(self.device):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            model = build_sam3_image_model(device=str(self.device) if is_cuda(self.device) else "cpu")
            self.sam = Sam3Processor(model, device=self.device)
        return self.sam

    def get_vqa(self):
        self.free(keep="vqa")
        if self.vqa is None:
            from vqa_scorer import VQAScore
            cuda = is_cuda(self.device)
            self.vqa = VQAScore(model_id=self.args.vqa_model,
                                device=str(self.device) if cuda else "cpu",
                                dtype=torch.bfloat16 if cuda else torch.float32)
        return self.vqa

    def get_clip(self):
        self.free(keep="clip")
        if self.clip is None:
            from vqa_scorer import CLIPScore
            cuda = is_cuda(self.device)
            self.clip = CLIPScore(model_id=self.args.clip_model,
                                  device=str(self.device) if cuda else "cpu",
                                  dtype=torch.float16 if cuda else torch.float32)
        return self.clip

    def get_nsfw(self):
        self.free(keep="nsfw")
        if self.nsfw is None:
            self.nsfw = NSFWClassifier(self.device)
        return self.nsfw


class NSFWClassifier:
    '''
    LAION's CLIP-based NSFW detector (rewards.get_nsfw_model) on top of the
    normalized CLIP ViT-L/14 image embedding, same as generate_clean.py.
    .model holds both networks so Models.free can move them off the GPU.
    '''

    def __init__(self, device):
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        from rewards import get_nsfw_model
        self.device = device
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.scorer = get_nsfw_model()
        self.scorer.device = device
        self.model = torch.nn.ModuleList([self.clip, self.scorer.nsfw_model]).to(device)

    @torch.no_grad()
    def batch(self, images, texts=None):
        images = [Image.open(p).convert("RGB") for p in images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        embeds = torch.nn.functional.normalize(self.clip(**inputs).image_embeds, dim=-1)
        return [{"score": float(p)} for p in self.scorer(embeds).float().cpu()]


def generate(pipe, prompt: str, seed: int, args, position_hook_dict: dict = None) -> Image.Image:
    gen = torch.Generator().manual_seed(seed)
    kwargs = dict(height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                  num_inference_steps=args.num_inference_steps, generator=gen)
    if position_hook_dict:
        return pipe.run_with_hooks(prompt, position_hook_dict=position_hook_dict, **kwargs).images[0]
    return pipe(prompt, **kwargs).images[0]


def sam_mask(sam_processor, image: Image.Image, query: str, device):
    '''
    Union of every SAM3 instance mask for `query`, plus the best instance
    score (0 when SAM3 finds nothing).
    '''
    def run():
        state = sam_processor.set_image(image)
        out = sam_processor.set_text_prompt(state=state, prompt=query)
        masks, scores = out["masks"], out["scores"]
        w, h = image.size
        if len(scores) == 0:
            return np.zeros((h, w), dtype=bool), 0.0
        return np.any(masks.squeeze(1).cpu().numpy(), axis=0), float(scores.float().max().cpu())

    if is_cuda(device):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return run()
    return run()


def sam_cache_path(image_path: str, query: str) -> str:
    return f"{image_path}.sam.{safe(query)}.npz"


def ensure_sam_masks(models: Models, pairs: list, device):
    '''
    pairs: (image_path, query). Writes {image}.sam.{query}.npz with
    pixel_mask/score for every pair not already done.
    '''
    todo = [(p, q) for p, q in pairs if not os.path.exists(sam_cache_path(p, q))]
    print(f"SAM3: {len(todo)} of {len(pairs)} masks to compute")
    if not todo:
        return
    sam = models.get_sam()
    for n, (path, query) in enumerate(todo):
        image = Image.open(path).convert("RGB")
        mask, score = sam_mask(sam, image, query, device)
        np.savez_compressed(sam_cache_path(path, query), pixel_mask=mask, score=np.float32(score))
        if n % 200 == 0:
            print(f"  SAM3 {n}/{len(todo)}")


def load_sam(image_path: str, query: str):
    with np.load(sam_cache_path(image_path, query)) as d:
        return d["pixel_mask"].astype(bool), float(d["score"])


def score_cache_path(image_path: str, kind: str, text: str) -> str:
    return f"{image_path}.{kind}.{safe(text)}.json"


def ensure_text_scores(score_fn, kind: str, pairs: list, batch_size: int, model: str):
    '''
    Score every (image_path, text) pair with a vqa_scorer scorer, cached per
    pair. Each image is scored against its own text only. A cached score
    from a different model is recomputed.
    '''
    def done(p, t):
        path = score_cache_path(p, kind, t)
        return os.path.exists(path) and load_json(path, {}).get("model") == model

    todo = [(p, t) for p, t in pairs if not done(p, t)]
    print(f"{kind}: {len(todo)} of {len(pairs)} scores to compute")
    if not todo:
        return
    fn = score_fn()
    for start in range(0, len(todo), batch_size):
        part = todo[start:start + batch_size]
        outs = fn.batch([p for p, _ in part], [t for _, t in part])
        for (p, t), out in zip(part, outs):
            save_json(score_cache_path(p, kind, t), {"text": t, "model": model, **out})
        done_n = start + len(part)
        if done_n % (10 * batch_size) < batch_size or done_n == len(todo):
            print(f"  {kind} {done_n}/{len(todo)}")


def load_text_score(image_path: str, kind: str, text: str):
    path = score_cache_path(image_path, kind, text)
    if not os.path.exists(path):
        return None
    return load_json(path, {}).get("score")


# ---------------------------------------------------------------- stage 1

def base_manifest(args) -> list:
    '''
    One entry per base prompt x base subject, reusing seeds/prompts already
    recorded in manifest.json so a rerun never silently changes a base image.
    '''
    base_dir = os.path.join(args.out_dir, "base")
    manifest_path = os.path.join(base_dir, "manifest.json")
    saved = {e["name"]: e for e in load_json(manifest_path, [])}

    entries = []
    i = 0
    for p_i, template in enumerate(read_lines(args.base_prompt_file)):
        for subject in read_lines(args.base_subject_file):
            name = f"{i:03d}_{safe(subject)}_p{p_i}"
            entry = saved.get(name) or {
                "name": name,
                "prompt": fill_prompt(template, subject, args.placeholder),
                "subject": subject,
                "seed": args.seed + i,
            }
            entry["image"] = os.path.join(base_dir, "images", f"{name}.jpg")
            entries.append(entry)
            i += 1
    return entries


def run_base(args, models: Models, device) -> list:
    base_dir = os.path.join(args.out_dir, "base")
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
    entries = base_manifest(args)

    todo = [e for e in entries if not os.path.exists(e["image"])]
    print(f"base: {len(todo)} of {len(entries)} images to generate")
    if todo:
        pipe = models.get_pipe()
        for e in todo:
            generate(pipe, e["prompt"], e["seed"], args).save(e["image"])
        del pipe

    ensure_sam_masks(models, [(e["image"], e["subject"]) for e in entries], device)
    for e in entries:
        mask, score = load_sam(e["image"], e["subject"])
        e["mask_area"] = float(mask.mean())
        e["mask_score"] = score
        highlighted = e["image"].replace(".jpg", "_highlighted.jpg")
        if not os.path.exists(highlighted):
            highlight_pixel_mask(Image.open(e["image"]).convert("RGB"), mask).save(highlighted)
        if mask.sum() == 0:
            print(f"  ! SAM3 found no '{e['subject']}' in {e['name']} - it will be skipped in stage 3")

    save_json(os.path.join(base_dir, "manifest.json"), entries)
    return entries


# ---------------------------------------------------------------- stage 2

def dream_entries(args, subjects: list) -> list:
    dream_dir = os.path.join(args.out_dir, "dream")
    entries = []
    for subject in subjects:
        for j, template in enumerate(read_lines(args.dream_prompt_file)):
            name = f"{safe(subject)}__{j:02d}"
            entries.append({
                "name": name,
                "subject": subject,
                "prompt": fill_prompt(template, subject, args.placeholder),
                "seed": args.dream_seed + j,
                "image": os.path.join(dream_dir, "images", f"{name}.jpg"),
                "embedding": os.path.join(dream_dir, "embeddings", f"{name}.npz"),
                "sparse": os.path.join(dream_dir, "sparse", f"{name}.npz"),
            })
    return entries


def run_dream_generate(args, models: Models, entries: list, block_list: list):
    todo = [e for e in entries if not (os.path.exists(e["image"]) and os.path.exists(e["embedding"]))]
    print(f"dream: {len(todo)} of {len(entries)} images to generate + cache")
    if not todo:
        return
    pipe = models.get_pipe()
    positions = [f"unet.{block}" for block in block_list]
    for e in todo:
        output, cache = pipe.run_with_cache(
            prompt=e["prompt"], positions_to_cache=positions, save_input=True, save_output=True,
            num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
            height=args.size, width=args.size, generator=torch.Generator().manual_seed(e["seed"]),
            output_type="pil",
        )
        output.images[0].save(e["image"])
        result = {}
        for block in block_list:
            pos = f"unet.{block}"
            # (batch, steps, C, H, W) -> last step, same as generate_clean_inference.generate_and_cache
            result[f"saved_input.{block}"] = cache["input"][pos][:, -1].cpu().float().numpy()
            result[f"saved_output.{block}"] = cache["output"][pos][:, -1].cpu().float().numpy()
        np.savez(e["embedding"], **result)


@torch.no_grad()
def run_dream_sparsify(args, models: Models, entries: list, block_list: list):
    '''
    Same encoding as sparsify.sparsify_embeddings, but saves only each
    patch's top-k latent indices/values ({block}__idx, {block}__val, both
    (h, w, k)) instead of the dense (h, w, n_dirs) code.
    '''
    todo = [e for e in entries if not os.path.exists(e["sparse"])]
    print(f"sparsify: {len(todo)} of {len(entries)} to encode")
    if not todo:
        return
    models.free(keep="sae")
    device = models.device
    for e in todo:
        result = {}
        with np.load(e["embedding"]) as data:
            for block in block_list:
                sae = models.get_sae(block)
                out = data[f"saved_output.{block}"]
                x = out - data[f"saved_input.{block}"] if args.mode == "diff" else out
                x = torch.tensor(x, device=device).squeeze(0).permute(1, 2, 0)
                latents = sae.encode(x)  # (h, w, n_dirs), relu'd top-k
                vals, inds = torch.topk(latents, k=sae.k, dim=-1)
                result[f"{block}__idx"] = inds.cpu().numpy().astype(np.int32)
                result[f"{block}__val"] = vals.cpu().numpy().astype(np.float32)
                result[f"{block}__n_dirs"] = np.int64(latents.shape[-1])
        np.savez(e["sparse"], **result)


def load_block_codes(entries: list, block: str):
    '''
    Stacks every entry's per-patch top-k codes for one block:
    idx/val (n_images * h * w, k), plus each row's entry index and grid size.
    '''
    idx_list, val_list, owner = [], [], []
    grid, n_dirs = None, None
    for n, e in enumerate(entries):
        with np.load(e["sparse"]) as d:
            idx, val = d[f"{block}__idx"], d[f"{block}__val"]
            n_dirs = int(d[f"{block}__n_dirs"])
        grid = idx.shape[:2]
        idx_list.append(idx.reshape(-1, idx.shape[-1]))
        val_list.append(val.reshape(-1, val.shape[-1]))
        owner.append(np.full(grid[0] * grid[1], n))
    return np.concatenate(idx_list), np.concatenate(val_list), np.concatenate(owner), grid, n_dirs


def features_dir(args) -> str:
    return os.path.join(args.out_dir, "features")


def load_features(args, subjects: list) -> dict:
    '''
    {subject: {block: probe result}} - one json per subject so jobs
    sharded by --subject_list never write the same file.
    '''
    out = {}
    for subject in subjects:
        path = os.path.join(features_dir(args), f"{safe(subject)}.json")
        if os.path.exists(path):
            out[subject] = load_json(path, {})
    return out


def run_discover(args, entries: list, subjects: list, block_list: list) -> dict:
    os.makedirs(features_dir(args), exist_ok=True)
    features = load_features(args, subjects)

    by_subject = {}
    for n, e in enumerate(entries):
        by_subject.setdefault(e["subject"], []).append(n)

    for block in block_list:
        idx_all, val_all, owner, (gh, gw), n_dirs = load_block_codes(entries, block)
        patch_labels = {}  # entry index -> (gh*gw,) bool for its own subject
        for n, e in enumerate(entries):
            mask, _ = load_sam(e["image"], e["subject"])
            patch_labels[n] = resize_mask_to_grid(mask, gh, gw).reshape(-1)

        for subject in subjects:
            if block in features.get(subject, {}):
                continue
            # only this subject's own dream images: SAM3's subject patches
            # are positives, the rest of those same images are negatives
            own = by_subject.get(subject, [])
            rows = np.isin(owner, own)
            labels = np.zeros(len(owner), dtype=bool)
            for n in own:
                labels[owner == n] = patch_labels[n]
            labels = labels[rows]

            n_pos = int(labels.sum())
            if n_pos == 0 or n_pos == len(labels):
                print(f"  skipped '{subject}' @ {block}: no positive/negative contrast")
                continue

            result = select_bce_and_f1(idx_all[rows], val_all[rows], labels, n_dirs,
                                       args.bce_ridge, args.bce_newton_steps)
            result["n_images"] = len(own)
            features.setdefault(subject, {})[block] = result
            print(f"'{subject}' @ {block}: bce latent {result['bce']['idx']} "
                  f"(bce={result['bce']['bce']:.4f}, explained={result['bce']['loss_explained']:.3f}) | "
                  f"f1 latent {result['f1']['idx']} (f1={result['f1']['f1']:.3f})")
        for subject, per_block in features.items():
            save_json(os.path.join(features_dir(args), f"{safe(subject)}.json"), per_block)
    return features


# ---------------------------------------------------------------- stage 3

def ablate_jobs(args, base_entries: list, features: dict, subjects: list, block_list: list) -> list:
    '''
    One job per block x subject x distinct chosen latent x strength x usable
    base image, plus random-latent controls (subject=None, scored later
    against every subject).
    '''
    ablate_dir = os.path.join(args.out_dir, "ablate", "images")
    usable = [e for e in base_entries if e.get("mask_area", 0) > 0]
    jobs = []

    for block in block_list:
        sblock = safe(block.replace(".", "_"))
        variants = []
        for subject in subjects:
            info = features.get(subject, {}).get(block)
            if info is None:
                continue
            if info["same_feature"]:
                variants.append((subject, "bce+f1", info["bce"]))
            else:
                variants.append((subject, "bce", info["bce"]))
                variants.append((subject, "f1", info["f1"]))

        for r in range(args.n_random_controls):
            # latent filled in by resolve_random_latents
            variants.append((None, f"random{r}", {"idx": None, "pos_mean": None}))

        for subject, selection, info in variants:
            for strength in args.strength_list:
                folder = safe(subject) if subject else "_random"
                for e in usable:
                    tag = f"{selection}_s{strength:g}"
                    jobs.append({
                        "block": block, "subject": subject, "selection": selection,
                        "feature_idx": info["idx"], "pos_mean": info.get("pos_mean"),
                        "strength": strength, "base": e["name"],
                        "image": os.path.join(ablate_dir, sblock, folder, tag, f"{e['name']}.jpg"),
                    })
    return jobs


def resolve_random_latents(args, models: Models, jobs: list, block_list: list):
    '''
    Random controls get a fixed latent per block/slot, recorded in
    ablate/random_latents.json so reruns and shards agree.
    '''
    path = os.path.join(args.out_dir, "ablate", "random_latents.json")
    chosen = load_json(path, {})
    rng = np.random.default_rng(args.seed)
    for block in block_list:
        for r in range(args.n_random_controls):
            key = f"{block}__random{r}"
            if key not in chosen:
                n_dirs = models.get_sae(block).n_dirs
                chosen[key] = int(rng.integers(n_dirs))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(path, chosen)
    for job in jobs:
        if job["subject"] is None:
            job["feature_idx"] = chosen[f"{job['block']}__{job['selection']}"]
    return chosen


@torch.no_grad()
def run_ablate_generate(args, models: Models, jobs: list, base_by_name: dict, device):
    todo = [j for j in jobs if not os.path.exists(j["image"])]
    print(f"ablate: {len(todo)} of {len(jobs)} images to generate")
    if not todo:
        return
    pipe = models.get_pipe()
    masks = {}
    for n, job in enumerate(todo):
        base = base_by_name[job["base"]]
        if base["name"] not in masks:
            masks[base["name"]], _ = load_sam(base["image"], base["subject"])
        sae = models.get_sae(job["block"])
        idx = job["feature_idx"]

        if args.inject_value == "pos_mean" and job["pos_mean"] is not None:
            value = job["pos_mean"]
        elif args.sae_source == "local":
            value = load_feature_mean(job["block"], idx, device)
        else:
            value = job["pos_mean"] or 1.0
        to_vec = torch.zeros(sae.n_dirs, device=device, dtype=torch.float32)
        to_vec[idx] = value

        hook_dict = make_add_position_hook_dict({job["block"]: sae}, {job["block"]: to_vec},
                                                args.start_step, args.end_step, job["strength"],
                                                device, masks[base["name"]])
        os.makedirs(os.path.dirname(job["image"]), exist_ok=True)
        generate(pipe, base["prompt"], base["seed"], args, hook_dict).save(job["image"])
        if n % 200 == 0:
            print(f"  ablate {n}/{len(todo)}")


def scored_pairs(jobs: list, subjects: list, base_by_name: dict):
    '''
    Every (job, subject) row that gets scored: a feature job against its
    own subject, a random control against every subject.
    '''
    for job in jobs:
        for subject in ([job["subject"]] if job["subject"] else subjects):
            yield job, subject, base_by_name[job["base"]]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def build_results(args, jobs: list, subjects: list, base_by_name: dict, features: dict):
    fmt = args.text_template.format
    rows = []
    cache = {}

    def arr(path):
        if path not in cache:
            cache[path] = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return cache[path]

    for job, subject, base in scored_pairs(jobs, subjects, base_by_name):
        if not os.path.exists(job["image"]):
            continue
        base_mask, _ = load_sam(base["image"], base["subject"])
        subj_mask, subj_score = load_sam(job["image"], subject)
        old_mask, old_score = load_sam(job["image"], base["subject"])
        base_subj_mask, base_subj_score = load_sam(base["image"], subject)

        inter = float((subj_mask & base_mask).sum())
        union = float((subj_mask | base_mask).sum())
        edited, original = arr(job["image"]), arr(base["image"])
        diff = np.abs(edited - original).mean(axis=-1)

        info = features.get(subject, {}).get(job["block"], {})
        chosen = info.get("f1" if job["selection"] == "f1" else "bce", {}) if job["subject"] else {}

        row = {
            "block": job["block"], "subject": subject, "selection": job["selection"],
            "feature_idx": job["feature_idx"], "strength": job["strength"],
            "base": base["name"], "base_subject": base["subject"], "base_prompt": base["prompt"],
            "seed": base["seed"], "image": job["image"],
            # how the feature was chosen (stage 2)
            "probe_bce": chosen.get("bce"), "probe_loss_explained": chosen.get("loss_explained"),
            "probe_f1": chosen.get("f1"),
            # 1) SAM3 overlap with the original region
            "mask_iou": inter / union if union else 0.0,
            "mask_precision": inter / subj_mask.sum() if subj_mask.sum() else 0.0,
            "mask_recall": inter / base_mask.sum() if base_mask.sum() else 0.0,
            "subject_area": float(subj_mask.mean()),
            "subject_sam_score": subj_score,
            "subject_sam_score_before": base_subj_score,
            "subject_iou_before": float((base_subj_mask & base_mask).sum()) / float((base_subj_mask | base_mask).sum())
            if (base_subj_mask | base_mask).sum() else 0.0,
            # is the original object still there?
            "base_subject_remaining": float((old_mask & base_mask).sum()) / float(base_mask.sum()),
            "base_subject_sam_score": old_score,
            # 2) VQAScore / CLIPScore
            "vqa_subject": load_text_score(job["image"], "vqa", fmt(subject)),
            "vqa_subject_before": load_text_score(base["image"], "vqa", fmt(subject)),
            "vqa_base_subject": load_text_score(job["image"], "vqa", fmt(base["subject"])),
            "clip_subject": load_text_score(job["image"], "clip", fmt(subject)),
            "clip_subject_before": load_text_score(base["image"], "clip", fmt(subject)),
            # 3) locality of the edit
            "background_psnr": psnr(edited[~base_mask], original[~base_mask]),
            "foreground_change": float(diff[base_mask].mean()),
            "background_change": float(diff[~base_mask].mean()) if (~base_mask).any() else 0.0,
        }
        for k in ["vqa", "clip"]:
            a, b = row[f"{k}_subject"], row[f"{k}_subject_before"]
            row[f"{k}_subject_gain"] = (a - b) if (a is not None and b is not None) else None
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no scored rows yet")
        return df
    df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)

    metrics = ["mask_iou", "mask_precision", "mask_recall", "subject_sam_score", "base_subject_remaining",
               "vqa_subject", "vqa_subject_gain", "vqa_base_subject", "clip_subject_gain",
               "background_psnr", "foreground_change", "background_change",
               "probe_bce", "probe_loss_explained", "probe_f1"]
    metrics = [m for m in metrics if m in df and df[m].notna().any()]
    df["kind"] = np.where(df["selection"].str.startswith("random"), "random", df["selection"])
    df.groupby(["block", "kind", "strength"])[metrics].mean().to_csv(
        os.path.join(args.out_dir, "summary_by_block.csv"))
    df.groupby(["subject", "block", "kind", "strength"])[metrics].mean().to_csv(
        os.path.join(args.out_dir, "summary_by_subject.csv"))
    print(df.groupby(["block", "kind", "strength"])[["mask_iou", "vqa_subject", "vqa_subject_gain"]]
          .mean().to_string())
    write_outputs_results(args, df)
    return df


def write_outputs_results(args, df: pd.DataFrame, filename: str = "results.csv",
                          keys: list = ("subject", "block", "kind", "strength")):
    '''
    {outputs_dir}/{filename}: one row per subject x block x feature choice
    (x strength) with the mean of every metric and the image count. Rows from
    earlier runs are kept unless this run (same out_dir) rescored that subject.
    '''
    keys = list(keys)
    skip = set(keys) | {"feature_idx", "seed", "pos_mean"}
    metrics = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(keys)
    table = grouped[metrics].mean()
    table.insert(0, "n_images", grouped.size())
    table.insert(0, "feature_idx", grouped["feature_idx"].agg(
        lambda x: ";".join(str(int(i)) for i in sorted(set(x.dropna())))))
    table = table.reset_index()
    table.insert(0, "out_dir", args.out_dir)

    os.makedirs(args.outputs_dir, exist_ok=True)
    path = os.path.join(args.outputs_dir, filename)
    if os.path.exists(path):
        old = pd.read_csv(path)
        replaced = (old["out_dir"] == args.out_dir) & old["subject"].isin(table["subject"])
        table = pd.concat([old[~replaced], table], ignore_index=True)
    table = table.sort_values(["out_dir"] + keys)
    table.to_csv(path, index=False)
    print(f"wrote {len(table)} rows to {path}")


def save_panels(jobs: list, base_by_name: dict):
    '''
    One strip per block/subject/selection/strength: base images on top of
    their edited versions, for eyeballing.
    '''
    groups = {}
    for job in jobs:
        if os.path.exists(job["image"]):
            groups.setdefault(os.path.dirname(job["image"]), []).append(job)
    for folder, group in groups.items():
        out = folder + "_panel.jpg"
        if os.path.exists(out):
            continue
        top = concat_images_horizontally([Image.open(base_by_name[j["base"]]["image"]).resize((256, 256)) for j in group])
        bottom = concat_images_horizontally([Image.open(j["image"]).resize((256, 256)) for j in group])
        concat_images_vertically([top, bottom]).save(out)


# ---------------------------------------------------------------- stage 4

def make_zero_hook(sae, feature_idx: int, mode: str, start_step: int, end_step: int, device):
    '''
    Sets one latent to 0 at every patch and leaves everything else as is:
    encode the block's diff (or output), take that latent's activation, and
    subtract its decoded contribution (no pre_bias - it's a delta) from the
    block's output. Patches where the latent wasn't in the top-k are untouched.
    '''
    step_counter = {"step": 0}

    def hook_fn(module, input, output):
        step = step_counter["step"]
        if start_step <= step <= end_step:
            out = output[0] if isinstance(output, tuple) else output
            orig_dtype = out.dtype
            x = out - input[0] if mode == "diff" else out
            x = x.permute(0, 2, 3, 1).float()
            latents = sae.encode(x)
            onehot = torch.zeros_like(latents)
            onehot[..., feature_idx] = latents[..., feature_idx]
            delta = sae.decoder(onehot).permute(0, 3, 1, 2)
            out = (out.float() - delta).to(device=device, dtype=orig_dtype)
            output = (out, *output[1:]) if isinstance(output, tuple) else out
        step_counter["step"] = step + 1
        return output

    return hook_fn


def read_aux_prompts(args) -> list:
    if not args.aux_prompt_file:
        return []
    if args.aux_prompt_file.endswith(".csv"):
        prompts = [str(p).strip() for p in pd.read_csv(args.aux_prompt_file)["prompt"].dropna()]
        prompts = [p for p in prompts if p]
    else:
        prompts = read_lines(args.aux_prompt_file)
    return prompts[:args.aux_limit] if args.aux_limit > 0 else prompts


def aux_entries(args, subjects: list) -> list:
    '''
    Same shape as dream_entries. A prompt with the placeholder gets one
    entry per subject; one without it is shared by every subject (same
    image, so it's generated once).
    '''
    aux_dir = os.path.join(args.out_dir, "aux", "images")
    entries = []
    for k, template in enumerate(read_aux_prompts(args)):
        for subject in subjects:
            if args.placeholder in template:
                name = f"aux_{safe(subject)}__{k:03d}"
                prompt = " ".join(template.replace(args.placeholder, subject).split())
            else:
                name = f"aux__{k:03d}"
                prompt = " ".join(template.split())
            entries.append({"name": name, "subject": subject, "prompt": prompt, "seed": args.aux_seed + k,
                            "image": os.path.join(aux_dir, f"{name}.jpg")})
    return entries


def run_aux_generate(args, models: Models, entries: list):
    todo = {e["image"]: e for e in entries if not os.path.exists(e["image"])}
    print(f"aux: {len(todo)} of {len(set(e['image'] for e in entries))} images to generate")
    if not todo:
        return
    os.makedirs(os.path.join(args.out_dir, "aux", "images"), exist_ok=True)
    pipe = models.get_pipe()
    for e in todo.values():
        generate(pipe, e["prompt"], e["seed"], args).save(e["image"])


def remove_jobs(args, sources: dict, features: dict, random_latents: dict, subjects: list, block_list: list) -> list:
    '''
    One job per block x subject x distinct chosen latent (+ random controls)
    x each of that subject's originals in every source ("dream", "aux").
    '''
    remove_dir = os.path.join(args.out_dir, "remove", "images")
    by_subject = {}
    for source, entries in sources.items():
        for e in entries:
            by_subject.setdefault(e["subject"], []).append((source, e))

    jobs = []
    for block in block_list:
        sblock = safe(block.replace(".", "_"))
        for subject in subjects:
            info = features.get(subject, {}).get(block)
            if info is None:
                continue
            variants = [("bce+f1", info["bce"]["idx"])] if info["same_feature"] else \
                [("bce", info["bce"]["idx"]), ("f1", info["f1"]["idx"])]
            variants += [(f"random{r}", random_latents[f"{block}__random{r}"])
                         for r in range(args.n_random_controls)]
            for selection, idx in variants:
                for source, e in by_subject.get(subject, []):
                    jobs.append({
                        "block": block, "subject": subject, "selection": selection, "source": source,
                        "feature_idx": int(idx), "dream": e["name"], "prompt": e["prompt"],
                        "seed": e["seed"], "original": e["image"],
                        "image": os.path.join(remove_dir, sblock, safe(subject), selection, f"{e['name']}.jpg"),
                    })
    return jobs


@torch.no_grad()
def run_remove_generate(args, models: Models, jobs: list):
    todo = [j for j in jobs if not os.path.exists(j["image"])]
    print(f"remove: {len(todo)} of {len(jobs)} images to generate")
    if not todo:
        return
    pipe = models.get_pipe()
    for n, job in enumerate(todo):
        sae = models.get_sae(job["block"])
        hook = make_zero_hook(sae, job["feature_idx"], args.mode, args.start_step, args.end_step, models.device)
        os.makedirs(os.path.dirname(job["image"]), exist_ok=True)
        generate(pipe, job["prompt"], job["seed"], args, {f"unet.{job['block']}": hook}).save(job["image"])
        if n % 200 == 0:
            print(f"  remove {n}/{len(todo)}")


NSFW_MODEL = "laion/clip-autokeras-binary-nsfw"


def ensure_nsfw_scores(args, models: Models, images: list):
    # the text slot is unused by the classifier; a fixed one reuses the vqa/clip cache layout
    ensure_text_scores(models.get_nsfw, "nsfw", [(p, "image") for p in images],
                       args.score_batch_size, model=NSFW_MODEL)


def build_removal_results(args, jobs: list):
    rows = []
    for job in jobs:
        if not (os.path.exists(sam_cache_path(job["image"], job["subject"]))
                and os.path.exists(sam_cache_path(job["original"], job["subject"]))):
            continue
        before_mask, before_score = load_sam(job["original"], job["subject"])
        after_mask, after_score = load_sam(job["image"], job["subject"])
        found_before = bool(before_mask.any())
        row = {
            "block": job["block"], "subject": job["subject"], "selection": job["selection"],
            "source": job["source"], "feature_idx": job["feature_idx"], "dream": job["dream"],
            "prompt": job["prompt"], "seed": job["seed"], "image": job["image"],
            "found_before": found_before,
            "sam_score_before": before_score, "area_before": float(before_mask.mean()),
            "sam_score_after": after_score, "area_after": float(after_mask.mean()),
            # only meaningful when SAM3 found the subject before the edit
            "removed": float(not after_mask.any()) if found_before else np.nan,
        }
        if args.use_nsfw:
            a = load_text_score(job["original"], "nsfw", "image")
            b = load_text_score(job["image"], "nsfw", "image")
            row.update({
                "nsfw_before": a, "nsfw_after": b,
                "nsfw_change": (b - a) if (a is not None and b is not None) else None,
                "nsfw_flagged_before": float(a >= args.nsfw_threshold) if a is not None else None,
                "nsfw_flagged_after": float(b >= args.nsfw_threshold) if b is not None else None,
                # of the originals the classifier flagged, did the removal un-flag it?
                "nsfw_unflagged": float(b < args.nsfw_threshold)
                if (a is not None and b is not None and a >= args.nsfw_threshold) else np.nan,
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no removal rows yet")
        return df
    df.to_csv(os.path.join(args.out_dir, "removal_results.csv"), index=False)

    df["kind"] = np.where(df["selection"].str.startswith("random"), "random", df["selection"])
    df["found_before"] = df["found_before"].astype(float)
    df = df.rename(columns={"removed": "removal_rate"})
    metrics = ["removal_rate", "found_before", "sam_score_before", "sam_score_after", "area_before", "area_after",
               "nsfw_before", "nsfw_after", "nsfw_change", "nsfw_flagged_before", "nsfw_flagged_after",
               "nsfw_unflagged"]
    metrics = [m for m in metrics if m in df]
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    keys = ["subject", "block", "kind", "source"]
    grouped = df.groupby(keys)
    summary = grouped[metrics].mean()
    summary.insert(0, "n_found_before", grouped["found_before"].sum())
    summary.to_csv(os.path.join(args.out_dir, "summary_removal.csv"))
    shown = [m for m in ["removal_rate", "sam_score_after", "nsfw_before", "nsfw_after", "nsfw_unflagged"] if m in df]
    print(df.groupby(["source", "block", "kind"])[shown].mean().to_string())

    write_outputs_results(args, df, filename="removal_results.csv", keys=keys)
    return df


# ---------------------------------------------------------------- main

def main(args):
    api, accelerator, device = repo_api_init(args)
    os.makedirs(args.out_dir, exist_ok=True)
    block_list = args.block_list if args.block_list else list(DEFAULT_BLOCK_LIST)
    subjects = args.subject_list if args.subject_list else read_lines(args.subject_file)
    if args.limit > 0:
        subjects = subjects[:args.limit]
    models = Models(args, device)

    # stage 1
    if args.disable_base:
        base_entries = load_json(os.path.join(args.out_dir, "base", "manifest.json"), [])
    else:
        base_entries = run_base(args, models, device)
    base_by_name = {e["name"]: e for e in base_entries}

    # stage 2
    entries = dream_entries(args, subjects)
    for sub in ["images", "embeddings", "sparse"]:
        os.makedirs(os.path.join(args.out_dir, "dream", sub), exist_ok=True)
    if not args.disable_dream_generate:
        run_dream_generate(args, models, entries, block_list)
    if not args.disable_dream_sparsify:
        run_dream_sparsify(args, models, entries, block_list)
    if not args.disable_dream_masks:
        ensure_sam_masks(models, [(e["image"], e["subject"]) for e in entries], device)
    if args.disable_discover:
        features = load_features(args, subjects)
    else:
        features = run_discover(args, entries, subjects, block_list)

    # stage 3
    jobs = ablate_jobs(args, base_entries, features, subjects, block_list)
    resolve_random_latents(args, models, jobs, block_list)
    if not args.disable_ablate_generate:
        run_ablate_generate(args, models, jobs, base_by_name, device)
        save_panels(jobs, base_by_name)

    fmt = args.text_template.format
    if not args.disable_ablate_masks:
        pairs = set()
        for job, subject, base in scored_pairs(jobs, subjects, base_by_name):
            if os.path.exists(job["image"]):
                pairs.update([(job["image"], subject), (job["image"], base["subject"]), (base["image"], subject)])
        ensure_sam_masks(models, sorted(pairs), device)

    text_pairs = set()
    for job, subject, base in scored_pairs(jobs, subjects, base_by_name):
        if os.path.exists(job["image"]):
            text_pairs.update([(job["image"], fmt(subject)), (job["image"], fmt(base["subject"])),
                               (base["image"], fmt(subject))])
    text_pairs = sorted(text_pairs)
    if not args.disable_vqa:
        ensure_text_scores(models.get_vqa, "vqa", text_pairs, args.score_batch_size, model=args.vqa_model)
    if not args.disable_clip:
        ensure_text_scores(models.get_clip, "clip", text_pairs, args.score_batch_size, model=args.clip_model)

    if not args.disable_summary:
        build_results(args, jobs, subjects, base_by_name, features)

    # stage 4
    random_latents = load_json(os.path.join(args.out_dir, "ablate", "random_latents.json"), {})
    aux = aux_entries(args, subjects)
    r_jobs = remove_jobs(args, {"dream": entries, "aux": aux}, features, random_latents, subjects, block_list)
    if not args.disable_remove_generate:
        run_aux_generate(args, models, aux)
        run_remove_generate(args, models, r_jobs)
    if not args.disable_remove_masks:
        pairs = set()
        for j in r_jobs:
            for path in [j["image"], j["original"]]:
                if os.path.exists(path):
                    pairs.add((path, j["subject"]))
        ensure_sam_masks(models, sorted(pairs), device)
    if args.use_nsfw:
        ensure_nsfw_scores(args, models, sorted({p for j in r_jobs for p in [j["image"], j["original"]]
                                                if os.path.exists(p)}))
    if not args.disable_remove_summary:
        build_removal_results(args, r_jobs)


if __name__ == '__main__':
    print_details()
    start = time.time()
    args = parser.parse_args()
    print_args(parser)
    print(args)
    main(args)
    seconds = time.time() - start
    print(f"successful generating:) time elapsed: {seconds} seconds = {seconds / 3600} hours")
    print("all done!")
