# UnlearnCanvas (Zhang et al., NeurIPS 2024 D&B, arxiv.org/abs/2402.11846,
# github.com/OPTML-Group/UnlearnCanvas - checked out in ./UnlearnCanvas)
# version of evaluate_sae_features.py: its 20 objects and 50 styles are the
# concepts, and each one is "unlearned" by zeroing a single SAE latent at
# every patch (evaluate_sae_features.make_zero_hook) on sdxl-turbo.
#
# stage 1 ("discover"): every object x style x --discover_seeds is generated
#   from the benchmark's own prompt ("A {object} image in {style} style.")
#   while caching the SAE blocks' activations AND the UNet's cross-attention
#   (attn2) probabilities for the object's and the style's text tokens. The
#   activations are SAE-encoded (top-k only, same as evaluate_sae_features).
#   With --object_discover_prompt_file (e.g. prompt_dir/dream_prompts.txt),
#   objects are discovered from those prompts instead (the placeholder filled
#   with the object, x --discover_seeds, no style); the object x style grid
#   is then only generated for style targets.
#
# stage 2 ("masks"): positives for a concept come from one of three masks:
#   - "attention": the object's/style's token cross-attention map averaged
#     over every attn2 layer and step; the top --frac of patches (at each SAE
#     block's own grid) are positive. Works for objects and styles.
#   - "sam": SAM3's mask for the object (objects only).
#   - "grad_eclip": Grad-ECLIP heat map (grad_eclip_mask.py, as in
#     generate_clean_style.py) of the concept text; top --frac is positive.
#   --object_mask_methods / --style_mask_methods pick which run per type.
#
# stage 3 ("probe"): per concept x mask method x block, 1D ridge logistic
#   probes over every latent (sparse_probe.select_bce_and_f1) on the images
#   that contain the concept (positive patches vs the rest of those images;
#   --negatives all also adds every patch of the other images). Keeps the
#   lowest-BCE and the highest-F1 latent -> {out_dir}/features/{concept}__{method}.json
#   (one file per mask method, so parallel per-method jobs sharing an out_dir
#   never overwrite each other; an older combined {concept}.json is still read).
#   Only the --*_mask_methods of this run are loaded and tested in stage 4.
#
# stage 4 ("answers"): the UnlearnCanvas answer set is every --eval_objects x
#   --eval_styles x --eval_seeds prompt. With --eval_scope target (default)
#   a concept's latents are only tested on the prompts it appears in: an
#   object on its styles x seeds, a style on its objects x seeds. Each prompt
#   is generated unedited ("base") and with each of its concepts' latents
#   zeroed everywhere, so one latent choice per concept covers the grid once
#   for objects and once for styles. Random latents per block are zeroed on
#   the same prompts as a control. An image is shared by every concept/method/
#   rule that zeroes the same (block, latent) on the same prompt.
#   --eval_scope all tests every latent on the whole grid (UnlearnCanvas's
#   own protocol, needed for IRA/CRA) at |grid| images per latent.
#
# stage 5 ("score"), per concept and per chosen latent (plus base/random):
#   UnlearnCanvas metrics with its pretrained ViT-L/16 style + object
#   classifiers (--style_ckpt/--class_ckpt, from the repo's Google Drive
#   cls_model folder; skipped if missing), same transform/heads as
#   machine_unlearning/evaluation/quantitative/accuracy.py:
#     UA  - answer images with the target that are NOT classified as it
#     IRA - same-domain classifier accuracy on images without the target
#           (only --eval_scope all has any)
#     CRA - other-domain classifier accuracy: with --eval_scope target, on
#           the target images themselves (remove "Bears" from a Bears/Van_Gogh
#           prompt - is it still Van Gogh?); with --eval_scope all, on the
#           images without the target as in UnlearnCanvas, and the
#           target-image version is reported as CRA_target
#   and the evaluate_sae_features metrics: SAM3 removal rate of the object
#   (objects only), VQAScore / CLIPScore of the concept text on target
#   images, and PSNR vs the unedited image (target and non-target images).
#   Per-image rows -> {out_dir}/uc_results_{methods}.csv.gz, means per concept x
#   method x block x rule -> {out_dir}/uc_summary_{methods}.csv (with the unedited
#   model's numbers as *_base columns) and {outputs_dir}/uc_results.csv.
#
# stage 6 ("inject", evaluate_sae_features.py's stage 3): the base images
#   (--base_prompt_file x --base_subject_file, SAM3-masked on the base
#   subject; shared with evaluate_sae_features' stage 1 code) are regenerated
#   with the same seed/prompt while adding each chosen latent's decoder
#   direction * value * strength inside the base subject's mask (random
#   latents as a control). Scored with evaluate_sae_features' metrics - SAM3
#   of the concept on the edit vs the original mask (IoU/precision/recall,
#   objects only), whether the base subject is gone, VQAScore / CLIPScore of
#   the concept vs the unedited image, background PSNR and fore/background
#   change - plus the UnlearnCanvas classifiers (is the edit now classified as
#   the injected object/style, and its probability, vs the unedited image).
#   Rows -> {out_dir}/inject_results_{methods}.csv, means ->
#   {out_dir}/inject_summary_{methods}.csv and {outputs_dir}/uc_inject_results.csv.
#
# --top_k k: the probe keeps each rule's best k latents (the top-1, then the
# next best with a positive probe weight) and stage 4 zeroes all k together,
# with k random latents per block as the control. Images go to
# answers/{block}/latents{a}_{b}_.../ (k=1 keeps answers/{block}/latent{idx}/),
# tables carry top_k and the latent set, and per-run files get a _k{k} tag.
# Injection (stage 6) adds one latent's direction, so it only runs with k=1.
#
# --n_objects N: only the first N objects of --object_list, for discovery,
# the answer grid (so IRA is over those N) and the targets. --limit only cuts
# the targets.
#
# Panels: {out_dir}/panels/{concept}_{methods}.jpg - the first --panel_rows
# target prompts x every variant (unedited, each method/block/rule, random),
# captioned with the classifiers' object / style predictions.
#
# Every pass skips work whose output already exists, so it can be rerun or
# sharded with --target_objects / --target_styles.
#
# Only the probe results (features/) and the result tables depend on the
# mask method / rule. Everything else - discovery images, activations,
# attention/SAM3/Grad-ECLIP maps, base and unedited answer images, zeroed and
# injected images (keyed by latent), random control latents, and every
# cached classifier/VQA/CLIP score - lives under --cache_dir (default:
# --out_dir), so runs with different methods/rules can share it. Files are
# written then renamed, so parallel runs never read a half-written one.
# --prepare_only fills the cache with everything that doesn't depend on the
# probe (stages 1-2 for all mask methods, base/unedited images, random
# controls) and exits; run it first, then the per-method runs in parallel.

# ---------------------------------------------------------------- metrics
#
# Removal (stage 5) -> uc_results.csv.gz, uc_summary.csv, {outputs_dir}/uc_results.csv
# "target images" = answer-set prompts that contain the concept.
#
#   metric                  source                    computed on                      meaning
#   UA                      UnlearnCanvas classifier  target images                    fraction NOT classified as the target
#   CRA                     other-domain classifier   target images (--eval_scope      is the other half still recognized?
#                                                     target, the default)             (remove Cats -> still Van Gogh?)
#   IRA                     same-domain classifier    non-target images (--eval_scope  are other concepts of the same type
#                                                     all only)                        still recognized?
#   CRA_target              other-domain classifier   target images (--eval_scope      CRA's target-image version; with
#                                                     all only)                        "all", CRA uses non-target images
#   p_target                classifier softmax        target images                    probability given to the target
#   style_acc, object_acc   both classifiers          every image                      raw accuracy of each classifier
#   sam_removed             SAM3                      target images, objects only      fraction where SAM3 finds no object
#   sam_score, sam_area     SAM3                      target images, objects only      SAM3 confidence, mask share of image
#   vqa, clip               VQAScore / CLIPScore      target images                    still matches "a photo of a {object}"
#                                                                                      / "an image in {style} style"?
#   psnr_target             pixels                    target images                    change vs the unedited image
#   psnr_retain             pixels                    non-target images (--eval_scope  collateral change on unrelated prompts
#                                                     all only)
#
#   uc_summary.csv also has UA_base, CRA_base, p_target_base, sam_removed_base,
#   vqa_base, clip_base (and IRA_base / CRA_target_base where they
#   apply): the same metrics on the unedited answer set. Every row carries
#   probe_bce / probe_loss_explained / probe_f1: how well the chosen latent
#   separated the mask on the discovery images.
#
# Injection (stage 6) -> inject_results.csv, inject_summary.csv, {outputs_dir}/uc_inject_results.csv
# "mask" = SAM3 mask of the base subject (cube, man, dog, ...) on the unedited base image.
#
#   metric                                source                 meaning
#   uc_classified_as (+_before)           UnlearnCanvas          is the edit classified as the injected
#                                         classifier             object/style?
#   uc_p_target (+_before, _gain)         classifier softmax     probability of the injected concept, and its change
#   mask_iou / mask_precision /           SAM3, objects only     overlap of the injected object's mask with the
#     mask_recall                                                original region
#   subject_area, subject_sam_score       SAM3, objects only     size of the injected object's mask, SAM3 confidence
#     (+_before)
#   base_subject_remaining,               SAM3                   how much of the original subject is still found
#     base_subject_sam_score                                     in its region
#   vqa_subject, clip_subject             VQAScore / CLIPScore   does the image now match the injected concept's text?
#     (+_before, _gain)
#   background_psnr                       pixels                 background preservation outside the mask
#   foreground_change, background_change  pixels                 mean absolute pixel change inside / outside the mask
#
#   Rows also carry probe_bce / probe_loss_explained / probe_f1.
#
# Grouping: every metric is averaged per concept x mask method (attention /
# sam / grad_eclip) x block x rule (bce, f1, or bce+f1 when both pick the same
# latent), and per strength for injection. "random*" rows zero/inject a random
# latent instead of the probe's pick (control); "base" rows in the removal
# tables are the unedited model.

import os
import time
import argparse
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from experiment_helpers.init_helpers import default_parser, repo_api_init

from attribution import DEFAULT_BLOCK_LIST
from generate_clean_inference import resize_mask_to_grid
from grad_eclip_mask import load_clip, grad_eclip_pixel_map, top_frac_patch_mask
from sparse_probe import select_bce_and_f1
from evaluate_sae_features import (
    Models, safe, load_json, save_json, generate, ensure_sam_masks, load_sam, sam_cache_path,
    ensure_text_scores, score_cache_path, load_text_score, run_dream_sparsify, load_block_codes,
    make_zero_hook, run_remove_generate, psnr, write_outputs_results,
    run_base, run_ablate_generate, save_image, save_npz, fill_prompt, read_lines,
)


def load_uc_constants(path: str):
    spec = importlib.util.spec_from_file_location("uc_const", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.theme_available), list(mod.class_available)


UC_CONST = "UnlearnCanvas/machine_unlearning/evaluation/constants/const.py"
THEMES, CLASSES = load_uc_constants(UC_CONST)  # THEMES includes "Seed_Images" (classifier head is 51-way)
STYLES = [t for t in THEMES if t != "Seed_Images"]

parser = default_parser({"repo_id": "jlbaker361/nsfw"})

parser.add_argument("--object_list", nargs="*", default=None, help="UnlearnCanvas objects used (default: all 20)")
parser.add_argument("--n_objects", type=int, default=0,
                    help="only the first N of --object_list (0 = all): discovery, the answer grid and the targets")
parser.add_argument("--style_list", nargs="*", default=None, help="UnlearnCanvas styles used (default: all 50)")
parser.add_argument("--target_objects", nargs="*", default=None,
                    help="objects to unlearn (default: --object_list); pass nothing after the flag for none")
parser.add_argument("--target_styles", nargs="*", default=None,
                    help="styles to unlearn (default: --style_list); pass nothing after the flag for none")
parser.add_argument("--template", type=str, default="A {object} image in {style} style.",
                    help="UnlearnCanvas's own answer-set prompt")
parser.add_argument("--discover_seeds", nargs="*", type=int, default=[0])
parser.add_argument("--object_discover_prompt_file", type=str, default=None,
                    help="discover objects from these prompts (--placeholder filled with the object, e.g. "
                         "prompt_dir/dream_prompts.txt) instead of the object x style grid")
parser.add_argument("--eval_objects", nargs="*", default=None, help="answer-set objects (default: --object_list)")
parser.add_argument("--eval_styles", nargs="*", default=None, help="answer-set styles (default: --style_list)")
parser.add_argument("--eval_seeds", nargs="*", type=int, default=[188, 288, 588, 688, 888],
                    help="UnlearnCanvas's answer-set seeds")
parser.add_argument("--eval_scope", type=str, default="target", choices=["target", "all"],
                    help="'target': test each concept's latents only on prompts containing it; "
                         "'all': on the whole answer set (adds IRA/CRA, |grid| images per latent)")

parser.add_argument("--out_dir", type=str, default="evaluation/uc_eval")
parser.add_argument("--cache_dir", type=str, default=None,
                    help="images/activations/masks/scores shared across runs (default: --out_dir)")
parser.add_argument("--prepare_only", action="store_true",
                    help="fill --cache_dir with everything that doesn't depend on the probe, for every "
                         "mask method, then exit")
parser.add_argument("--outputs_dir", type=str, default="evaluation/outputs")

parser.add_argument("--object_mask_methods", nargs="*", default=["attention", "sam"],
                    choices=["attention", "sam", "grad_eclip"])
parser.add_argument("--style_mask_methods", nargs="*", default=["attention", "grad_eclip"],
                    choices=["attention", "grad_eclip"])
parser.add_argument("--frac", type=float, default=0.25,
                    help="top fraction of patches that count as positive for the attention / grad_eclip masks")
parser.add_argument("--attn_map_size", type=int, default=64, help="grid every cross-attention map is resized to")
parser.add_argument("--rules", nargs="*", default=["bce", "f1"], choices=["bce", "f1"],
                    help="which probe rule(s) pick the latent that gets zeroed/injected")
parser.add_argument("--negatives", type=str, default="own", choices=["own", "all"],
                    help="'own': negatives are the non-mask patches of the concept's images; "
                         "'all': plus every patch of the discovery images without the concept")
parser.add_argument("--grad_eclip_object_text", type=str, default="a photo of a {}")
parser.add_argument("--grad_eclip_style_text", type=str, default="{} style")
parser.add_argument("--grad_eclip_model", type=str, default="ViT-B-16")
parser.add_argument("--grad_eclip_pretrained", type=str, default="openai")

parser.add_argument("--sae_source", type=str, default="local", choices=["local", "saeuron"])
parser.add_argument("--block_list", nargs="*", default=None)
parser.add_argument("--mode", type=str, default="diff", choices=["diff", "out"])
parser.add_argument("--bce_ridge", type=float, default=1e-8)
parser.add_argument("--bce_newton_steps", type=int, default=30)
parser.add_argument("--n_random_controls", type=int, default=1)
parser.add_argument("--top_k", type=int, default=1,
                    help="zero each concept's best k latents per method x block x rule together (and k random "
                         "latents per block as the control). Injection (stage 6) only runs with top_k 1")
parser.add_argument("--seed", type=int, default=0, help="picks the random control latents")
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--end_step", type=int, default=1000)

parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--size", type=int, default=512)

parser.add_argument("--style_ckpt", type=str, default="UnlearnCanvas/ckpts/cls_model/style50-001.pth")
parser.add_argument("--class_ckpt", type=str, default="UnlearnCanvas/ckpts/cls_model/style50_cls.pth")
parser.add_argument("--vqa_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
parser.add_argument("--object_text", type=str, default="a photo of a {}", help="VQAScore/CLIPScore text for objects")
parser.add_argument("--style_text", type=str, default="an image in {} style", help="VQAScore/CLIPScore text for styles")
parser.add_argument("--score_batch_size", type=int, default=16)

parser.add_argument("--base_prompt_file", type=str, default="prompt_dir/base_prompts.txt")
parser.add_argument("--base_subject_file", type=str, default="prompt_dir/base_subjects.txt")
parser.add_argument("--placeholder", type=str, default="<sks>")
parser.add_argument("--panel_rows", type=int, default=6, help="target prompts (rows) per concept panel")
parser.add_argument("--panel_size", type=int, default=160, help="side of each panel cell in pixels")
parser.add_argument("--strength_list", nargs="*", type=float, default=[10.0])
parser.add_argument("--inject_value", type=str, default="checkpoint_mean", choices=["checkpoint_mean", "pos_mean"],
                    help="activation placed on the latent before * strength: the SAE checkpoint's mean.pt "
                         "or its mean over the concept's positive discovery patches")

for flag in ["discover_generate", "sparsify", "masks", "probe", "answers_generate", "answer_masks",
             "uc", "vqa", "clip", "psnr", "summary", "panels",
             "inject", "base", "inject_generate", "inject_masks", "inject_summary"]:
    parser.add_argument(f"--disable_{flag}", action="store_true")


# ---------------------------------------------------------------- helpers

def words(name: str) -> str:
    return name.replace("_", " ")


def fill(args, obj: str, style: str) -> str:
    return args.template.format(object=words(obj), style=words(style))


def sam_query(obj: str) -> str:
    return words(obj).lower()


def concept_text(args, ctype: str, concept: str) -> str:
    return (args.object_text if ctype == "object" else args.style_text).format(words(concept))


def grad_eclip_text(args, ctype: str, concept: str) -> str:
    return (args.grad_eclip_object_text if ctype == "object" else args.grad_eclip_style_text).format(words(concept))


def token_span(tokenizer, prompt: str, text: str) -> list:
    '''Positions of text's tokens inside the tokenized prompt (CLIP BPE is per word, so they line up).'''
    ids = tokenizer(prompt).input_ids
    sub = tokenizer(text, add_special_tokens=False).input_ids
    for i in range(len(ids) - len(sub) + 1):
        if ids[i:i + len(sub)] == sub:
            return list(range(i, i + len(sub)))
    return []


class UCModels(Models):
    '''Models plus the Grad-ECLIP CLIP and the UnlearnCanvas classifiers, one on the GPU at a time.'''

    def __init__(self, args, device):
        super().__init__(args, device)
        self.geclip = None
        self.uc = None

    def free(self, keep: str = None):
        for name in ["geclip", "uc"]:
            obj = getattr(self, name)
            if name != keep and obj is not None:
                obj.model.to("cpu")
                setattr(self, name, None)
        super().free(keep)

    def get_geclip(self):
        self.free(keep="geclip")
        if self.geclip is None:
            self.geclip = GradEClip(self.args, self.device)
        return self.geclip

    def get_uc(self):
        self.free(keep="uc")
        if self.uc is None:
            self.uc = UCClassifiers(self.args.style_ckpt, self.args.class_ckpt, self.device)
        return self.uc


class GradEClip:
    def __init__(self, args, device):
        self.device = device
        self.model, self.tokenizer = load_clip(device, args.grad_eclip_model, args.grad_eclip_pretrained)

    def __call__(self, image: Image.Image, text: str) -> np.ndarray:
        return grad_eclip_pixel_map(self.model, self.tokenizer, image, text, self.device)


class UCClassifiers:
    '''
    UnlearnCanvas's style (51-way incl. Seed_Images) and object (20-way)
    ViT-L/16 classifiers, loaded and applied exactly like
    machine_unlearning/evaluation/quantitative/accuracy.py.
    '''

    def __init__(self, style_ckpt: str, class_ckpt: str, device):
        import timm
        from torchvision import transforms
        self.device = device

        def load(ckpt, n):
            # pretrained=False: every weight, head included, comes from the checkpoint
            m = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=False)
            m.head = torch.nn.Linear(1024, n)
            m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
            return m.to(device).eval()

        self.style = load(style_ckpt, len(THEMES))
        self.cls = load(class_ckpt, len(CLASSES))
        self.model = torch.nn.ModuleList([self.style, self.cls])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    @torch.no_grad()
    def batch(self, images, texts=None):
        x = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in images]).to(self.device)
        sp = F.softmax(self.style(x).float(), dim=-1).cpu().numpy()
        cp = F.softmax(self.cls(x).float(), dim=-1).cpu().numpy()
        return [{"score": THEMES[int(s.argmax())], "style_pred": THEMES[int(s.argmax())],
                 "class_pred": CLASSES[int(c.argmax())],
                 "style_probs": np.round(s, 5).tolist(), "class_probs": np.round(c, 5).tolist()}
                for s, c in zip(sp, cp)]


def uc_model_id(args) -> str:
    return f"{args.style_ckpt}|{args.class_ckpt}"


def load_uc(image_path: str):
    path = score_cache_path(image_path, "uc", "image")
    return load_json(path, None) if os.path.exists(path) else None


# ---------------------------------------------------------------- stage 1

class CrossAttnStore:
    '''Running sum of cross-attention probs (mean over heads) per attn2 resolution.'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums, self.counts = {}, {}

    def add(self, probs: torch.Tensor):
        hw = probs.shape[0]
        if hw not in self.sums:
            self.sums[hw] = torch.zeros_like(probs)
            self.counts[hw] = 0
        self.sums[hw] += probs
        self.counts[hw] += 1

    def token_map(self, span: list, size: int) -> np.ndarray:
        '''Summed attention to span's tokens, averaged over every layer/resolution at size x size, in [0, 1].'''
        if not span or not self.sums:
            return np.zeros((size, size), dtype=np.float32)
        maps = []
        for hw, total in self.sums.items():
            r = int(round(hw ** 0.5))
            m = (total[:, span].sum(-1) / self.counts[hw]).reshape(1, 1, r, r)
            maps.append(F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)[0, 0])
        m = torch.stack(maps).mean(0)
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return m.cpu().numpy().astype(np.float32)


class RecordingAttnProcessor:
    '''
    Wraps a UNet attn2 processor: records softmax(q k^T * scale) for the
    conditional (last) batch element, then runs the original processor
    unchanged so generation is untouched.
    '''

    def __init__(self, inner, store: CrossAttnStore):
        self.inner = inner
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        if encoder_hidden_states is not None:
            with torch.no_grad():
                q = attn.head_to_batch_dim(attn.to_q(hidden_states[-1:])).float()
                k = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states[-1:])).float()
                probs = torch.softmax(q @ k.transpose(-1, -2) * attn.scale, dim=-1)  # (heads, HW, tokens)
                self.store.add(probs.mean(0))
        return self.inner(attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs)


def discover_entries(args, objects: list, styles: list, style_targets: bool) -> list:
    '''
    "grid" entries (object x style x seed, the benchmark prompt) and, with
    --object_discover_prompt_file, "prompt" entries (template x object x seed,
    style None). Grid entries are only made when something uses them: always
    without the prompt file, else only for style targets.
    '''
    d = os.path.join(args.cache_dir, "discover")

    def entry(name, obj, style, seed, prompt, source):
        return {"name": name, "object": obj, "style": style, "seed": seed, "prompt": prompt, "source": source,
                "image": os.path.join(d, "images", f"{name}.jpg"),
                "embedding": os.path.join(d, "embeddings", f"{name}.npz"),
                "sparse": os.path.join(d, "sparse", f"{name}.npz"),
                "attn": os.path.join(d, "attn", f"{name}.npz")}

    entries = []
    if not args.object_discover_prompt_file or style_targets:
        for obj in objects:
            for style in styles:
                for seed in args.discover_seeds:
                    entries.append(entry(f"{obj}__{style}__s{seed}", obj, style, seed,
                                         fill(args, obj, style), "grid"))
    if args.object_discover_prompt_file:
        for j, template in enumerate(read_lines(args.object_discover_prompt_file)):
            for obj in objects:
                for seed in args.discover_seeds:
                    entries.append(entry(f"{obj}__prompt{j:02d}__s{seed}", obj, None, seed,
                                         fill_prompt(template, words(obj), args.placeholder), "prompt"))
    return entries


def concept_entries(args, entries: list, ctype: str, concept: str) -> list:
    '''Indices of the discovery images a concept is probed on.'''
    source = "prompt" if (ctype == "object" and args.object_discover_prompt_file) else "grid"
    return [n for n, e in enumerate(entries) if e["source"] == source and e[ctype] == concept]


@torch.no_grad()
def run_discover_generate(args, models: UCModels, entries: list, block_list: list):
    todo = [e for e in entries if not all(os.path.exists(e[k]) for k in ["image", "embedding", "attn"])]
    print(f"discover: {len(todo)} of {len(entries)} images to generate + cache")
    if not todo:
        return
    for sub in ["images", "embeddings", "attn"]:
        os.makedirs(os.path.join(args.cache_dir, "discover", sub), exist_ok=True)
    pipe = models.get_pipe()
    unet, tokenizer = pipe.pipe.unet, pipe.pipe.tokenizer
    store = CrossAttnStore()
    original = dict(unet.attn_processors)
    unet.set_attn_processor({n: RecordingAttnProcessor(p, store) if "attn2" in n else p for n, p in original.items()})
    positions = [f"unet.{block}" for block in block_list]
    try:
        for n, e in enumerate(todo):
            store.reset()
            output, cache = pipe.run_with_cache(
                prompt=e["prompt"], positions_to_cache=positions, save_input=True, save_output=True,
                num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                height=args.size, width=args.size, generator=torch.Generator().manual_seed(e["seed"]),
                output_type="pil",
            )
            save_image(output.images[0], e["image"])
            result = {}
            for block in block_list:
                pos = f"unet.{block}"
                result[f"saved_input.{block}"] = cache["input"][pos][:, -1].cpu().float().numpy()
                result[f"saved_output.{block}"] = cache["output"][pos][:, -1].cpu().float().numpy()
            save_npz(e["embedding"], **result)

            spans = {role: token_span(tokenizer, e["prompt"], words(e[role]))
                     for role in ["object", "style"] if e[role] is not None}
            for role, span in spans.items():
                if not span:
                    print(f"  ! no '{e[role]}' tokens found in '{e['prompt']}' - its attention map is all zero")
            save_npz(e["attn"], compressed=True, **{role: store.token_map(span, args.attn_map_size).astype(np.float16)
                                              for role, span in spans.items()})
            if n % 100 == 0:
                print(f"  discover {n}/{len(todo)}")
    finally:
        unet.set_attn_processor(original)


# ---------------------------------------------------------------- stage 2

def geclip_cache_path(image_path: str, text: str) -> str:
    return f"{image_path}.geclip.{safe(text)}.npz"


def ensure_grad_eclip_maps(models: UCModels, pairs: list):
    todo = [(p, t) for p, t in pairs if not os.path.exists(geclip_cache_path(p, t))]
    print(f"Grad-ECLIP: {len(todo)} of {len(pairs)} maps to compute")
    if not todo:
        return
    geclip = models.get_geclip()
    for n, (path, text) in enumerate(todo):
        pixel_map = geclip(Image.open(path).convert("RGB"), text)
        save_npz(geclip_cache_path(path, text), compressed=True, pixel_map=pixel_map.astype(np.float16))
        if n % 200 == 0:
            print(f"  Grad-ECLIP {n}/{len(todo)}")


def concept_methods(args, ctype: str) -> list:
    return args.object_mask_methods if ctype == "object" else args.style_mask_methods


def run_masks(args, models: UCModels, entries: list, targets: list, device):
    sam_pairs, geclip_pairs = set(), set()
    for ctype, concept in targets:
        methods = concept_methods(args, ctype)
        for n in concept_entries(args, entries, ctype, concept):
            e = entries[n]
            if "sam" in methods:
                sam_pairs.add((e["image"], sam_query(concept)))
            if "grad_eclip" in methods:
                geclip_pairs.add((e["image"], grad_eclip_text(args, ctype, concept)))
    ensure_sam_masks(models, sorted(sam_pairs), device)
    ensure_grad_eclip_maps(models, sorted(geclip_pairs))


def patch_labels(args, e: dict, ctype: str, concept: str, method: str, gh: int, gw: int) -> np.ndarray:
    if method == "attention":
        with np.load(e["attn"]) as d:
            return top_frac_patch_mask(d[ctype].astype(np.float32), gh, gw, args.frac)
    if method == "sam":
        mask, _ = load_sam(e["image"], sam_query(concept))
        return resize_mask_to_grid(mask, gh, gw)
    with np.load(geclip_cache_path(e["image"], grad_eclip_text(args, ctype, concept))) as d:
        return top_frac_patch_mask(d["pixel_map"].astype(np.float32), gh, gw, args.frac)


# ---------------------------------------------------------------- stage 3

def run_tag(args) -> str:
    '''Mask methods of this run, so per-method jobs sharing an out_dir write separate tables.'''
    tag = "_".join(sorted(set(args.object_mask_methods) | set(args.style_mask_methods)))
    return tag if args.top_k == 1 else f"{tag}_k{args.top_k}"


def features_path(args, concept: str, method: str) -> str:
    return os.path.join(args.out_dir, "features", f"{safe(concept)}__{method}.json")


def legacy_features_path(args, concept: str) -> str:
    # before per-method files: every method of a concept in one json
    return os.path.join(args.out_dir, "features", f"{safe(concept)}.json")


def load_features(args, targets: list) -> dict:
    '''
    {concept: {"type": ctype, "methods": {method: {block: result}}}} for this
    run's mask methods only, from the per-method files (falling back to the
    legacy combined file).
    '''
    features = {}
    for ctype, concept in targets:
        legacy = load_json(legacy_features_path(args, concept), {}).get("methods", {})
        methods = {}
        for method in concept_methods(args, ctype):
            per_block = load_json(features_path(args, concept, method), None)
            if per_block is None:
                per_block = legacy.get(method, {})
            methods[method] = per_block
        features[concept] = {"type": ctype, "methods": methods}
    return features


def run_probe(args, entries: list, targets: list, block_list: list) -> dict:
    '''{concept: {"type": ctype, "methods": {method: {block: select_bce_and_f1 result}}}}'''
    os.makedirs(os.path.join(args.out_dir, "features"), exist_ok=True)
    features = load_features(args, targets)

    for block in block_list:
        todo = [(t, c, m) for t, c in targets for m in concept_methods(args, t)
                if block not in features[c]["methods"].get(m, {})
                or features[c]["methods"][m][block].get("top_k", 1) < args.top_k]
        if not todo:
            continue
        idx_all, val_all, owner, (gh, gw), n_dirs = load_block_codes(entries, block)
        for ctype, concept, method in todo:
            own = concept_entries(args, entries, ctype, concept)
            labels = np.zeros(len(owner), dtype=bool)
            for n in own:
                labels[owner == n] = patch_labels(args, entries[n], ctype, concept, method, gh, gw).reshape(-1)
            rows = np.ones(len(owner), dtype=bool) if args.negatives == "all" else np.isin(owner, own)
            labels = labels[rows]
            n_pos = int(labels.sum())
            if n_pos == 0 or n_pos == len(labels):
                print(f"  skipped {ctype} '{concept}' ({method}) @ {block}: no positive/negative contrast")
                continue
            result = select_bce_and_f1(idx_all[rows], val_all[rows], labels, n_dirs,
                                       args.bce_ridge, args.bce_newton_steps, top_k=args.top_k)
            result["n_images"] = len(own)
            features[concept]["methods"].setdefault(method, {})[block] = result
            print(f"{ctype} '{concept}' ({method}) @ {block}: bce latent {result['bce']['idx']} "
                  f"(bce={result['bce']['bce']:.4f}, explained={result['bce']['loss_explained']:.3f}) | "
                  f"f1 latent {result['f1']['idx']} (f1={result['f1']['f1']:.3f})")
        for ctype, concept, method in todo:
            save_json(features_path(args, concept, method), features[concept]["methods"].get(method, {}))
    return features


# ---------------------------------------------------------------- stage 4

def answer_entries(args) -> list:
    base = os.path.join(args.cache_dir, "answers", "base")
    return [{"object": o, "style": s, "seed": seed, "prompt": fill(args, o, s),
             "file": f"{s}_{o}_seed{seed}.jpg", "image": os.path.join(base, f"{s}_{o}_seed{seed}.jpg")}
            for s in args.eval_styles for o in args.eval_objects for seed in args.eval_seeds]


def latent_dir(args, block: str, latents: list) -> str:
    # one latent keeps the old "latent{idx}" folder, so earlier top-1 images are reused
    name = f"latent{latents[0]}" if len(latents) == 1 else "latents" + "_".join(str(i) for i in sorted(latents))
    return os.path.join(args.cache_dir, "answers", safe(block.replace(".", "_")), name)


def resolve_random_latents(args, models: UCModels, block_list: list) -> dict:
    path = os.path.join(args.cache_dir, "random_latents.json")
    chosen = load_json(path, {})
    rng = np.random.default_rng(args.seed)
    for block in block_list:
        for r in range(args.n_random_controls):
            key = f"{block}__random{r}"
            if key not in chosen:
                chosen[key] = int(rng.integers(models.get_sae(block).n_dirs))
            if args.top_k > 1 and f"{key}__k{args.top_k}" not in chosen:
                # the top-1 random latent plus k-1 more, so k random latents per block
                n_dirs = models.get_sae(block).n_dirs
                pool = np.setdiff1d(np.arange(n_dirs), [chosen[key]])
                extra = np.random.default_rng([args.seed, r, args.top_k]).choice(pool, args.top_k - 1, replace=False)
                chosen[f"{key}__k{args.top_k}"] = [chosen[key]] + [int(i) for i in extra]
    os.makedirs(args.cache_dir, exist_ok=True)
    save_json(path, chosen)
    return chosen


def variants(args, features: dict, random_latents: dict, targets: list, block_list: list) -> list:
    '''
    Every (concept, method, block, rule, latent) that gets scored: "base"
    (the unedited answer set), each concept's bce/f1 latents, and the random
    controls (scored against every concept).
    '''
    out = []
    for ctype, concept in targets:
        common = {"subject": concept, "concept_type": ctype}
        out.append({**common, "method": "none", "block": "none", "kind": "base", "feature_idx": None,
                    "latents": [], "probe": {}})
        for method, per_block in features.get(concept, {}).get("methods", {}).items():
            if method not in concept_methods(args, ctype):
                continue  # only the mask methods this run was asked to test
            for block in block_list:
                info = per_block.get(block)
                if info is None:
                    continue
                if args.top_k == 1:
                    picks = {r: [info[r]["idx"]] for r in ["bce", "f1"]}
                else:
                    picks = {r: [d["idx"] for d in info[f"{r}_top"][:args.top_k]] for r in ["bce", "f1"]}
                same = sorted(picks["bce"]) == sorted(picks["f1"])
                rules = ["bce+f1"] if same else [r for r in ["bce", "f1"] if r in args.rules]
                for rule in rules:
                    key = "f1" if rule == "f1" else "bce"
                    chosen = info[key]  # top-1: its probe stats go in the tables
                    out.append({**common, "method": method, "block": block, "kind": rule,
                                "feature_idx": chosen["idx"], "latents": picks[key], "probe": chosen})
        for block in block_list:
            for r in range(args.n_random_controls):
                key = f"{block}__random{r}" + ("" if args.top_k == 1 else f"__k{args.top_k}")
                latents = random_latents[key] if args.top_k > 1 else [random_latents[key]]
                out.append({**common, "method": "random", "block": block, "kind": f"random{r}",
                            "feature_idx": latents[0], "latents": latents, "probe": {}})
    for v in out:
        v["top_k"] = args.top_k
    return out


def variant_image(args, v: dict, a: dict) -> str:
    return a["image"] if v["kind"] == "base" else os.path.join(latent_dir(args, v["block"], v["latents"]), a["file"])


def is_target(v: dict, a: dict) -> bool:
    return a[v["concept_type"]] == v["subject"]


def scored_answers(args, v: dict, answers: list) -> list:
    return answers if args.eval_scope == "all" else [a for a in answers if is_target(v, a)]


def run_answers_generate(args, models: UCModels, answers: list, var_list: list):
    needed = {}
    for v in var_list:
        if v["kind"] == "base":
            for a in scored_answers(args, v, answers):
                needed[a["image"]] = a
    todo = [a for path, a in needed.items() if not os.path.exists(path)]
    print(f"answers (base): {len(todo)} of {len(needed)} images to generate")
    if todo:
        os.makedirs(os.path.dirname(todo[0]["image"]), exist_ok=True)
        pipe = models.get_pipe()
        for a in todo:
            save_image(generate(pipe, a["prompt"], a["seed"], args), a["image"])

    jobs = {}
    for v in var_list:
        if v["kind"] == "base":
            continue
        for a in scored_answers(args, v, answers):
            path = variant_image(args, v, a)
            # make_zero_hook zeroes a list of latents as well as one
            jobs[path] = {"block": v["block"], "feature_idx": list(v["latents"]), "prompt": a["prompt"],
                          "seed": a["seed"], "image": path}
    n_sets = len({(j["block"], tuple(j["feature_idx"])) for j in jobs.values()})
    print(f"answers: {len(jobs)} zeroed-latent images over {n_sets} distinct latent sets (top_k={args.top_k})")
    run_remove_generate(args, models, list(jobs.values()))


# ---------------------------------------------------------------- stage 5

def run_scoring(args, models: UCModels, answers: list, var_list: list, device):
    images = sorted({variant_image(args, v, a) for v in var_list for a in scored_answers(args, v, answers)})
    images = [p for p in images if os.path.exists(p)]

    if not args.disable_answer_masks:
        pairs = {(variant_image(args, v, a), sam_query(v["subject"]))
                 for v in var_list if v["concept_type"] == "object" for a in answers if is_target(v, a)}
        ensure_sam_masks(models, sorted(p for p in pairs if os.path.exists(p[0])), device)

    if not args.disable_uc:
        if os.path.exists(args.style_ckpt) and os.path.exists(args.class_ckpt):
            ensure_text_scores(models.get_uc, "uc", [(p, "image") for p in images], args.score_batch_size,
                               model=uc_model_id(args))
        else:
            print(f"! UnlearnCanvas classifiers not found ({args.style_ckpt}, {args.class_ckpt}) - "
                  f"skipping UA/IRA/CRA")

    text_pairs = sorted({(variant_image(args, v, a), concept_text(args, v["concept_type"], v["subject"]))
                         for v in var_list for a in answers if is_target(v, a)})
    text_pairs = [p for p in text_pairs if os.path.exists(p[0])]
    if not args.disable_vqa:
        ensure_text_scores(models.get_vqa, "vqa", text_pairs, args.score_batch_size, model=args.vqa_model)
    if not args.disable_clip:
        ensure_text_scores(models.get_clip, "clip", text_pairs, args.score_batch_size, model=args.clip_model)


def build_results(args, answers: list, var_list: list):
    rows = []
    for v in var_list:
        ctype, concept = v["concept_type"], v["subject"]
        text = concept_text(args, ctype, concept)
        for a in scored_answers(args, v, answers):
            path = variant_image(args, v, a)
            if not os.path.exists(path):
                continue
            target = is_target(v, a)
            row = {
                "subject": concept, "concept_type": ctype, "method": v["method"], "block": v["block"],
                "kind": v["kind"], "feature_idx": v["feature_idx"], "top_k": v["top_k"],
                "latents": ";".join(str(i) for i in v["latents"]),
                "object": a["object"], "style": a["style"], "seed": a["seed"], "image": path,
                "is_target": float(target),
                "probe_bce": v["probe"].get("bce"), "probe_loss_explained": v["probe"].get("loss_explained"),
                "probe_f1": v["probe"].get("f1"),
            }
            uc = load_uc(path) if not args.disable_uc else None
            if uc is not None and uc.get("model") == uc_model_id(args):
                style_ok = float(uc["style_pred"] == a["style"])
                class_ok = float(uc["class_pred"] == a["object"])
                same, other = (style_ok, class_ok) if ctype == "style" else (class_ok, style_ok)
                target_pred = uc["style_pred"] if ctype == "style" else uc["class_pred"]
                probs = uc["style_probs"] if ctype == "style" else uc["class_probs"]
                labels = THEMES if ctype == "style" else CLASSES
                row.update({
                    "style_acc": style_ok, "object_acc": class_ok,
                    "UA": float(target_pred != concept) if target else np.nan,
                    "IRA": same if not target else np.nan,
                    "CRA": other if (not target or args.eval_scope == "target") else np.nan,
                    "CRA_target": other if (target and args.eval_scope == "all") else np.nan,
                    "p_target": float(probs[labels.index(concept)]) if target else np.nan,
                })
            if target and ctype == "object" and os.path.exists(sam_cache_path(path, sam_query(concept))):
                mask, score = load_sam(path, sam_query(concept))
                row.update({"sam_removed": float(not mask.any()), "sam_score": score,
                            "sam_area": float(mask.mean())})
            if target:
                row["vqa"] = load_text_score(path, "vqa", text)
                row["clip"] = load_text_score(path, "clip", text)
            if not args.disable_psnr and v["kind"] != "base" and os.path.exists(a["image"]):
                edited = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
                original = np.asarray(Image.open(a["image"]).convert("RGB"), dtype=np.float32) / 255.0
                row["psnr_target" if target else "psnr_retain"] = psnr(edited, original)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no scored rows yet")
        return df
    df.to_csv(os.path.join(args.out_dir, f"uc_results_{run_tag(args)}.csv.gz"), index=False)

    keys = ["subject", "concept_type", "method", "block", "kind", "top_k"]
    metrics = ["UA", "IRA", "CRA", "CRA_target", "p_target", "style_acc", "object_acc", "sam_removed", "sam_score", "sam_area",
               "vqa", "clip", "psnr_target", "psnr_retain",
               "probe_bce", "probe_loss_explained", "probe_f1"]
    metrics = [m for m in metrics if m in df]
    for m in metrics:
        df[m] = pd.to_numeric(df[m].replace([np.inf], np.nan), errors="coerce")
    df["block"] = df["block"].fillna("none")
    grouped = df.groupby(keys)
    summary = grouped[metrics].mean()
    summary.insert(0, "n_images", grouped.size())
    summary.insert(0, "latents", grouped["latents"].first())
    summary.insert(0, "feature_idx", grouped["feature_idx"].first())
    summary = summary.reset_index()

    # the unedited model's numbers next to every row of the same concept
    base_cols = [m for m in ["UA", "IRA", "CRA", "CRA_target", "p_target", "sam_removed", "vqa", "clip"] if m in summary]
    base = summary[summary["kind"] == "base"].set_index("subject")[base_cols].add_suffix("_base")
    summary = summary.join(base, on="subject")
    summary.to_csv(os.path.join(args.out_dir, f"uc_summary_{run_tag(args)}.csv"), index=False)

    shown = [m for m in ["UA", "IRA", "CRA", "CRA_target", "sam_removed", "vqa", "psnr_target", "psnr_retain"]
             if m in summary and summary[m].notna().any()]
    print(summary.groupby(["concept_type", "method", "kind"])[shown].mean().to_string())

    write_outputs_results(args, df.drop(columns=["seed", "latents"]), filename="uc_results.csv", keys=keys,
                          replace_on=["subject", "method", "top_k"], defaults={"top_k": 1})
    return df


def save_panels(args, answers: list, var_list: list):
    '''
    {out_dir}/panels/{concept}_{methods}.jpg: one row per target prompt (first
    --panel_rows), one column per variant - unedited, then each mask method x
    block x rule, then the random controls. Every cell is captioned with the
    UnlearnCanvas classifiers' object / style prediction (red = the removed
    concept is still predicted). Only uses images already in the cache.
    '''
    d = os.path.join(args.out_dir, "panels")
    os.makedirs(d, exist_ok=True)
    size, head, cap = args.panel_size, 34, 14
    by_concept = {}
    for v in var_list:
        by_concept.setdefault((v["concept_type"], v["subject"]), []).append(v)
    for (ctype, concept), vs in by_concept.items():
        order = {"none": 0, "random": 2}  # unedited first, random controls last
        vs = sorted(vs, key=lambda v: (order.get(v["method"], 1), v["method"], v["block"], v["kind"]))
        rows = [a for a in answers if a[ctype] == concept][:args.panel_rows]
        if not rows:
            continue
        grid = Image.new("RGB", (size * len(vs), head + (size + cap) * len(rows)), "white")
        draw = ImageDraw.Draw(grid)
        for c, v in enumerate(vs):
            label = "unedited" if v["kind"] == "base" else \
                f"{v['method']} {v['kind']}\n{v['block'].replace('_blocks', '').replace('.attentions', '')} " \
                f"#{','.join(str(i) for i in v['latents'])[:24]}"
            draw.text((c * size + 3, 2), label, fill="black")
            for r, a in enumerate(rows):
                x, y = c * size, head + r * (size + cap)
                path = variant_image(args, v, a)
                if not os.path.exists(path):
                    draw.rectangle([x, y, x + size - 1, y + size - 1], fill="lightgray")
                    continue
                grid.paste(Image.open(path).convert("RGB").resize((size, size)), (x, y))
                uc = load_uc(path) if not args.disable_uc else None
                if uc is not None and uc.get("model") == uc_model_id(args):
                    pred = uc["class_pred"] if ctype == "object" else uc["style_pred"]
                    draw.text((x + 3, y + size), f"{uc['class_pred']} / {uc['style_pred']}"[:30],
                              fill="red" if pred == concept else "black")
        # row labels: the other half of each prompt (the style for objects, the object for styles)
        other = "style" if ctype == "object" else "object"
        for r, a in enumerate(rows):
            draw.text((3, head + r * (size + cap) + 3), a[other], fill="yellow")
        save_image(grid, os.path.join(d, f"{safe(concept)}_{run_tag(args)}.jpg"))
    print(f"panels -> {d}")


# ---------------------------------------------------------------- stage 6

def inject_jobs(args, base_entries: list, var_list: list) -> list:
    '''
    One job per (variant, strength, masked base image). Images are keyed by
    (block, latent, strength, base) - plus the value with --inject_value
    pos_mean, since that depends on the concept - so concepts/rules that
    picked the same latent share them.
    '''
    usable = [e for e in base_entries if e.get("mask_area", 0) > 0]
    jobs = []
    for v in var_list:
        if v["kind"] == "base":
            continue
        pos_mean = v["probe"].get("pos_mean")
        folder = f"latent{v['feature_idx']}"
        if args.inject_value == "pos_mean" and pos_mean is not None:
            folder += f"_pos{pos_mean:.4g}"
        for strength in args.strength_list:
            for e in usable:
                jobs.append({
                    **{k: v[k] for k in ["subject", "concept_type", "method", "block", "kind", "feature_idx"]},
                    "probe": v["probe"], "pos_mean": pos_mean, "strength": strength, "base": e["name"],
                    "image": os.path.join(args.cache_dir, "inject", "images", safe(v["block"].replace(".", "_")),
                                          folder, f"s{strength:g}", f"{e['name']}.jpg"),
                })
    return jobs


def run_inject_scoring(args, models: UCModels, jobs: list, base_by_name: dict, device):
    done = [j for j in jobs if os.path.exists(j["image"])]
    base_images = sorted({base_by_name[j["base"]]["image"] for j in done})
    images = sorted({j["image"] for j in done}) + base_images

    if not args.disable_inject_masks:
        pairs = set()
        for j in done:
            base = base_by_name[j["base"]]
            pairs.add((j["image"], base["subject"]))
            if j["concept_type"] == "object":
                pairs.update([(j["image"], sam_query(j["subject"])), (base["image"], sam_query(j["subject"]))])
        ensure_sam_masks(models, sorted(pairs), device)

    if not args.disable_uc and os.path.exists(args.style_ckpt) and os.path.exists(args.class_ckpt):
        ensure_text_scores(models.get_uc, "uc", [(p, "image") for p in images], args.score_batch_size,
                           model=uc_model_id(args))

    text_pairs = set()
    for j in done:
        text = concept_text(args, j["concept_type"], j["subject"])
        text_pairs.update([(j["image"], text), (base_by_name[j["base"]]["image"], text)])
    text_pairs = sorted(text_pairs)
    if not args.disable_vqa:
        ensure_text_scores(models.get_vqa, "vqa", text_pairs, args.score_batch_size, model=args.vqa_model)
    if not args.disable_clip:
        ensure_text_scores(models.get_clip, "clip", text_pairs, args.score_batch_size, model=args.clip_model)


def uc_target(args, path: str, ctype: str, concept: str):
    '''(classified as concept, p(concept)) from the UnlearnCanvas classifiers, or (None, None).'''
    uc = load_uc(path) if not args.disable_uc else None
    if uc is None or uc.get("model") != uc_model_id(args):
        return None, None
    pred = uc["style_pred"] if ctype == "style" else uc["class_pred"]
    probs = uc["style_probs"] if ctype == "style" else uc["class_probs"]
    labels = THEMES if ctype == "style" else CLASSES
    return float(pred == concept), float(probs[labels.index(concept)])


def build_inject_results(args, jobs: list, base_by_name: dict):
    rows = []
    cache = {}

    def arr(path):
        if path not in cache:
            cache[path] = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return cache[path]

    for j in jobs:
        base = base_by_name[j["base"]]
        if not (os.path.exists(j["image"]) and os.path.exists(sam_cache_path(j["image"], base["subject"]))):
            continue
        ctype, concept = j["concept_type"], j["subject"]
        text = concept_text(args, ctype, concept)
        base_mask, _ = load_sam(base["image"], base["subject"])
        old_mask, old_score = load_sam(j["image"], base["subject"])
        edited, original = arr(j["image"]), arr(base["image"])
        diff = np.abs(edited - original).mean(axis=-1)

        row = {
            **{k: j[k] for k in ["subject", "concept_type", "method", "block", "kind", "feature_idx", "strength"]},
            "base": base["name"], "base_subject": base["subject"], "base_prompt": base["prompt"],
            "image": j["image"],
            "probe_bce": j["probe"].get("bce"), "probe_loss_explained": j["probe"].get("loss_explained"),
            "probe_f1": j["probe"].get("f1"),
            # is the original object still there?
            "base_subject_remaining": float((old_mask & base_mask).sum()) / float(base_mask.sum()),
            "base_subject_sam_score": old_score,
            # locality of the edit
            "background_psnr": psnr(edited[~base_mask], original[~base_mask]),
            "foreground_change": float(diff[base_mask].mean()),
            "background_change": float(diff[~base_mask].mean()) if (~base_mask).any() else 0.0,
            "vqa_subject": load_text_score(j["image"], "vqa", text),
            "vqa_subject_before": load_text_score(base["image"], "vqa", text),
            "clip_subject": load_text_score(j["image"], "clip", text),
            "clip_subject_before": load_text_score(base["image"], "clip", text),
        }
        if ctype == "object" and os.path.exists(sam_cache_path(j["image"], sam_query(concept))) \
                and os.path.exists(sam_cache_path(base["image"], sam_query(concept))):
            subj_mask, subj_score = load_sam(j["image"], sam_query(concept))
            _, before_score = load_sam(base["image"], sam_query(concept))
            inter, union = float((subj_mask & base_mask).sum()), float((subj_mask | base_mask).sum())
            row.update({
                "mask_iou": inter / union if union else 0.0,
                "mask_precision": inter / subj_mask.sum() if subj_mask.sum() else 0.0,
                "mask_recall": inter / base_mask.sum() if base_mask.sum() else 0.0,
                "subject_area": float(subj_mask.mean()),
                "subject_sam_score": subj_score, "subject_sam_score_before": before_score,
            })
        hit, p = uc_target(args, j["image"], ctype, concept)
        hit_before, p_before = uc_target(args, base["image"], ctype, concept)
        if hit is not None:
            row.update({"uc_classified_as": hit, "uc_classified_as_before": hit_before,
                        "uc_p_target": p, "uc_p_target_before": p_before})
        for k in ["vqa_subject", "clip_subject", "uc_p_target"]:
            a, b = row.get(k), row.get(f"{k}_before")
            if k in row:
                row[f"{k}_gain"] = (a - b) if (a is not None and b is not None) else None
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no injection rows yet")
        return df
    df.to_csv(os.path.join(args.out_dir, f"inject_results_{run_tag(args)}.csv"), index=False)

    keys = ["subject", "concept_type", "method", "block", "kind", "strength"]
    metrics = [c for c in df.columns if c not in keys and c != "feature_idx"
               and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
               and df[c].notna().any() and c not in ["base", "base_subject", "base_prompt", "image"]]
    for m in metrics:
        df[m] = pd.to_numeric(df[m].replace([np.inf], np.nan), errors="coerce")
    grouped = df.groupby(keys)
    summary = grouped[metrics].mean()
    summary.insert(0, "n_images", grouped.size())
    summary.insert(0, "feature_idx", grouped["feature_idx"].first())
    summary.reset_index().to_csv(os.path.join(args.out_dir, f"inject_summary_{run_tag(args)}.csv"), index=False)

    shown = [m for m in ["mask_iou", "base_subject_remaining", "vqa_subject_gain", "uc_classified_as",
                         "uc_p_target_gain", "background_psnr"] if m in df]
    print(df.groupby(["concept_type", "method", "kind", "strength"])[shown].mean().to_string())
    write_outputs_results(args, df, filename="uc_inject_results.csv", keys=keys, replace_on=["subject", "method"])
    return df


# ---------------------------------------------------------------- main

def main(args):
    api, accelerator, device = repo_api_init(args)
    args.cache_dir = args.cache_dir or args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    block_list = args.block_list if args.block_list else list(DEFAULT_BLOCK_LIST)
    if args.prepare_only:
        # every mask method, so any later --*_mask_methods finds its masks cached
        args.object_mask_methods = ["attention", "sam", "grad_eclip"]
        args.style_mask_methods = ["attention", "grad_eclip"]

    args.object_list = args.object_list or list(CLASSES)
    if args.n_objects > 0:
        args.object_list = args.object_list[:args.n_objects]
        print(f"--n_objects {args.n_objects}: objects {args.object_list}")
    args.style_list = args.style_list or list(STYLES)
    for name in args.object_list:
        assert name in CLASSES, f"'{name}' is not an UnlearnCanvas object: {CLASSES}"
    for name in args.style_list:
        assert name in THEMES, f"'{name}' is not an UnlearnCanvas style: {STYLES}"
    args.eval_objects = args.eval_objects or args.object_list
    args.eval_styles = args.eval_styles or args.style_list
    target_objects = args.object_list if args.target_objects is None else args.target_objects
    target_styles = args.style_list if args.target_styles is None else args.target_styles
    targets = [("object", o) for o in target_objects] + [("style", s) for s in target_styles]
    if args.limit > 0:
        targets = targets[:args.limit]
    print(f"{len(targets)} concepts to unlearn: {targets}")

    models = UCModels(args, device)

    # stages 1-3
    entries = discover_entries(args, args.object_list, args.style_list,
                               style_targets=any(t == "style" for t, _ in targets))
    os.makedirs(os.path.join(args.cache_dir, "discover", "sparse"), exist_ok=True)
    if not args.disable_discover_generate:
        run_discover_generate(args, models, entries, block_list)
    if not args.disable_sparsify:
        run_dream_sparsify(args, models, entries, block_list)
    if not args.disable_masks:
        run_masks(args, models, entries, targets, device)
    if args.prepare_only:
        features = {}  # -> only the unedited answers and the random controls below
    elif args.disable_probe:
        features = load_features(args, targets)
    else:
        features = run_probe(args, entries, targets, block_list)

    # stage 4
    answers = answer_entries(args)
    random_latents = resolve_random_latents(args, models, block_list)
    var_list = variants(args, features, random_latents, targets, block_list)
    if not args.disable_answers_generate:
        run_answers_generate(args, models, answers, var_list)

    # stage 5
    if not args.prepare_only:
        run_scoring(args, models, answers, var_list, device)
    if not args.disable_summary and not args.prepare_only:
        build_results(args, answers, var_list)
    if not args.disable_panels and not args.prepare_only:
        save_panels(args, answers, var_list)

    # stage 6
    if not args.disable_inject and args.top_k > 1:
        print(f"skipping injection (stage 6): it adds one latent's direction, and --top_k is {args.top_k}")
    elif not args.disable_inject:
        # run_base writes {out_dir}/base - point it at the shared cache
        cache_args = argparse.Namespace(**{**vars(args), "out_dir": args.cache_dir})
        if args.disable_base:
            base_entries = load_json(os.path.join(args.cache_dir, "base", "manifest.json"), [])
        else:
            base_entries = run_base(cache_args, models, device)
        base_by_name = {e["name"]: e for e in base_entries}
        i_jobs = inject_jobs(args, base_entries, var_list)
        if not args.disable_inject_generate:
            unique = list({j["image"]: j for j in i_jobs}.values())
            run_ablate_generate(args, models, unique, base_by_name, device)
        if args.prepare_only:
            print("prepared cache in", args.cache_dir)
            return
        run_inject_scoring(args, models, i_jobs, base_by_name, device)
        if not args.disable_inject_summary:
            build_inject_results(args, i_jobs, base_by_name)


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
