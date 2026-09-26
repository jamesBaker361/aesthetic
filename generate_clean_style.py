# style version of generate_clean_inference.py: instead of SAM3-localized
# objects, finds the SAE latent(s) for a *style* using same-seed pairs.
#
# pipeline: for every scene in --scene_file (one noun/scene per line), generate
# a plain image ("{scene}") plus one stylized image per --style_list entry
# (--style_template, default "{scene}, {style} style"), ALL from the same seed
# (the scene's line index) so each styled image is a controlled twin of its
# plain one - caching per-block activations in the same single sdxl-turbo
# run_with_cache pass generate_clean_inference.generate_and_cache uses ->
# sparsify.sparsify_embeddings -> per style, per block, rank latents by how
# much they separate styled from plain -> save into --npz_dict with the exact
# same "{query}__{block}" key layout generate_clean_inference.py writes, so
# generate_clean_ablate.py / generate_clean_swap.py read it unchanged (query
# = the style name, spaces -> underscores).
#
# Masking (--mask_top_frac, default 0.25): a style isn't necessarily spread
# evenly over the whole frame, so every mode below only looks at the patches
# where the style actually shows up. For each styled image, Grad-ECLIP
# (grad_eclip_mask.py - Zhao et al., ICML 2024, github.com/Cyang-Zhao/Grad-Eclip)
# explains its CLIP similarity to --grad_eclip_text (default "{style} style");
# at each SAE block's own grid, the top mask_top_frac of patches by that heat
# map are kept - in the styled image AND at the same positions in its
# same-seed plain twin, so positives and negatives stay a controlled pair.
# Everything else (both images) is dropped. --mask_top_frac 1.0 disables
# masking (whole image, no CLIP needed).
#
# The train/test partition is over SCENES, not images, so a plain image and
# its styled twins always land on the same side.
#
# --feature_selection:
#   paired modes (use the same-seed pairing directly; each selects the single
#   best latent and records its styled-patch mean/std, like bce/f1, so
#   generate_clean_ablate.py makes its mean/plus_std/minus_std variants):
#   "mean_diff" (default): per scene, subtract each plain patch's activation
#     from the matching styled patch's, average over all patches and scenes,
#     pick the largest. Note this is exactly equal to (mean styled-image
#     activation - mean plain-image activation), so it does NOT actually depend
#     on the patches lining up spatially.
#   "paired_t": same per-scene differences d (image-mean styled - plain), but
#     scored by the paired t-statistic mean(d) / (std(d)/sqrt(n_scenes)) -
#     rewards a latent that goes up on every scene over one that spikes
#     hugely on a few.
#   "patch_winrate": the one mode that really uses patch alignment - fraction
#     of aligned (scene, patch) positions where the styled activation is
#     higher than the plain one, minus the fraction where it's lower. Only
#     meaningful if the style mostly preserves the plain image's layout
#     (usually true-ish for 1-step sdxl-turbo with a shared seed).
#   label modes (generate_clean_inference.select_features, unchanged): every
#   (masked) styled patch is positive, every (masked) plain patch negative:
#   "bce", "f1", "saeuron", "auroc" - see generate_clean_inference.py.

import os
import json
import argparse
import time
import random

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from experiment_helpers.init_helpers import default_parser, repo_api_init

from sdxl_unbox.SDLens.hooked_sd_pipeline import HookedStableDiffusionXLPipeline
from sparsify import sparsify_embeddings
from attribution import DEFAULT_BLOCK_LIST

from generate_clean_inference import read_prompts, select_features, update_npz_dict
from grad_eclip_mask import load_clip, grad_eclip_pixel_map, top_frac_patch_mask

PAIRED_MODES = ["mean_diff", "paired_t", "patch_winrate"]
LABEL_MODES = ["bce", "f1", "saeuron", "auroc"]

parser = default_parser(
    {
        "repo_id": "jlbaker361/nsfw"
    }
)

parser.add_argument("--image_src_dir", type=str, default="style_images")
parser.add_argument("--embedding_dir", type=str, default="style_embeddings")
parser.add_argument("--sparse_embedding_dir", type=str, default="style_sparse_embeddings")
parser.add_argument("--eval_dir", type=str, default="style_eval_dir")
parser.add_argument("--mask_dir", type=str, default="style_mask_dir")
parser.add_argument("--npz_dict", type=str, default="style.npz")

parser.add_argument("--scene_file", type=str, default="prompt_dir/subjects.txt")  # one noun/scene per line
parser.add_argument("--style_list", nargs="*", default=[])  # e.g. anime watercolor "van gogh"
parser.add_argument("--style_template", type=str, default="{scene}, {style} style")

parser.add_argument("--partition_path", type=str, default="style_partition.json")
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=42)  # partition shuffle only - generation seeds are the scene index

# num_inference_steps=1, guidance_scale=0.0 match collect_latents_dataset.py
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--mode", type=str, default="diff")  # fed to sparsify_embeddings

parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--feature_selection", type=str, default="mean_diff", choices=PAIRED_MODES + LABEL_MODES)
parser.add_argument("--bce_ridge", type=float, default=1e-8)
parser.add_argument("--bce_newton_steps", type=int, default=30)
parser.add_argument("--saeuron_percentile", type=float, default=99.9)

parser.add_argument("--mask_top_frac", type=float, default=0.25,
                     help="fraction of each styled image's patches (ranked by Grad-ECLIP) kept for discovery/eval; "
                          "1.0 = no mask")
parser.add_argument("--grad_eclip_text", type=str, default="{style} style")
parser.add_argument("--clip_model", type=str, default="ViT-B-16")
parser.add_argument("--clip_pretrained", type=str, default="openai")
parser.add_argument("--n_visualize", type=int, default=5)  # mask overlays saved per style

parser.add_argument("--sae_source", type=str, default="local", choices=["local", "saeuron"])
parser.add_argument("--block_list", nargs="*", default=None)

parser.add_argument("--disable_generate", action="store_true")
parser.add_argument("--disable_sparsify_embeddings", action="store_true")
parser.add_argument("--disable_masks", action="store_true")
parser.add_argument("--disable_discover", action="store_true")


def safe_name(s: str) -> str:
    return s.replace(" ", "_")


def plain_name(i: int) -> str:
    return f"scene_{i}__plain.jpg"


def styled_name(i: int, style: str) -> str:
    return f"scene_{i}__{safe_name(style)}.jpg"


def generate_pairs_and_cache(image_src_dir: str, embedding_dir: str, scenes: list, style_list: list,
                              style_template: str, block_list: list, size: int, num_inference_steps: int,
                              guidance_scale: float, mixed_precision: str, device):
    '''
    Same single run_with_cache pass as generate_clean_inference.generate_and_cache,
    but per scene: the plain prompt and every styled prompt all get a fresh
    generator seeded with the scene index, so they start from identical noise.
    '''
    print("generate same-seed plain/styled pairs + cache block activations")
    os.makedirs(image_src_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    dtype = torch.float16 if (torch.cuda.is_available() and mixed_precision == "fp16") else torch.float32
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=dtype,
        variant=("fp16" if dtype == torch.float16 else None),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    positions = [f"unet.{block}" for block in block_list]

    for i, scene in enumerate(scenes):
        jobs = [(plain_name(i), scene)] + [
            (styled_name(i, style), style_template.format(scene=scene, style=style)) for style in style_list
        ]
        for name, prompt in jobs:
            image_path = os.path.join(image_src_dir, name)
            npz_path = os.path.join(embedding_dir, name + ".npz")
            if os.path.exists(image_path) and os.path.exists(npz_path):
                continue

            generator = torch.Generator(device="cpu").manual_seed(i)
            output, cache = pipe.run_with_cache(
                prompt=prompt,
                positions_to_cache=positions,
                save_input=True,
                save_output=True,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=size,
                width=size,
                generator=generator,
                output_type="pil",
            )
            output.images[0].save(image_path)

            result = {}
            for block in block_list:
                pos = f"unet.{block}"
                result[f"saved_input.{block}"] = cache["input"][pos][:, -1].cpu().float().numpy()
                result[f"saved_output.{block}"] = cache["output"][pos][:, -1].cpu().float().numpy()
            np.savez(npz_path, **result)


def get_or_make_scene_partition(n_scenes: int, partition_path: str, train_frac: float, seed: int) -> dict:
    if os.path.exists(partition_path):
        with open(partition_path) as f:
            return json.load(f)
    idx = list(range(n_scenes))
    random.Random(seed).shuffle(idx)
    n_train = int(round(n_scenes * train_frac))
    partition = {"train": sorted(idx[:n_train]), "test": sorted(idx[n_train:])}
    with open(partition_path, "w") as f:
        json.dump(partition, f, indent=2)
    return partition


def mask_path(mask_dir: str, name: str) -> str:
    return os.path.join(mask_dir, f"{name}.grad_eclip.npz")


def cache_grad_eclip_maps(image_src_dir: str, mask_dir: str, scene_idx: list, style_list: list,
                           grad_eclip_text: str, top_frac: float, n_visualize: int,
                           clip_model: str, clip_pretrained: str, device, viz_grid: int = 16):
    '''
    One Grad-ECLIP pixel heat map per styled image (text = grad_eclip_text
    formatted with the style), saved as float16 "pixel_map". The plain twin
    gets no map of its own - it reuses its styled partner's (see iter_pairs).
    Also saves an overlay of the top_frac pixels for the first n_visualize
    scenes of each style, as a sanity check on what the mask is picking up.
    '''
    print("computing Grad-ECLIP style masks")
    os.makedirs(mask_dir, exist_ok=True)
    model, tokenizer = None, None
    for style in style_list:
        text = grad_eclip_text.format(style=style)
        for n, i in enumerate(scene_idx):
            name = styled_name(i, style)
            out_path = mask_path(mask_dir, name)
            image_path = os.path.join(image_src_dir, name)
            if os.path.exists(out_path) or not os.path.exists(image_path):
                continue
            if model is None:
                model, tokenizer = load_clip(device, clip_model, clip_pretrained)
            image = Image.open(image_path).convert("RGB")
            pixel_map = grad_eclip_pixel_map(model, tokenizer, image, text, device)
            np.savez(out_path, pixel_map=pixel_map.astype(np.float16))

            if n < n_visualize:
                # drawn at viz_grid (16x16 = every DEFAULT_BLOCK_LIST block at
                # 512px) since that's the resolution the mask is actually used at
                h, w = pixel_map.shape
                patch_mask = top_frac_patch_mask(pixel_map, viz_grid, viz_grid, top_frac)
                pixel_mask = np.kron(patch_mask, np.ones((h // viz_grid, w // viz_grid), dtype=bool))
                pixel_mask = np.pad(pixel_mask, ((0, h - pixel_mask.shape[0]), (0, w - pixel_mask.shape[1])))
                img_np = np.array(image).astype(np.float32)
                img_np[~pixel_mask] *= 0.25  # darken everything outside the mask
                plain = Image.open(os.path.join(image_src_dir, plain_name(i))).convert("RGB")
                panel = Image.new("RGB", (w * 2, h))
                panel.paste(plain, (0, 0))
                panel.paste(Image.fromarray(np.uint8(img_np)), (w, 0))
                panel.save(os.path.join(mask_dir, f"_viz_{name}"))


def load_block_patches(sparse_embedding_dir: str, name: str, block: str):
    path = os.path.join(sparse_embedding_dir, name + ".npz")
    if not os.path.exists(path):
        return None
    return np.load(path)[block]  # (h,w,n_dirs)


def iter_pairs(scene_idx: list, sparse_embedding_dir: str, style: str, block: str,
                mask_dir: str = None, top_frac: float = 1.0):
    # yields (plain_patches, styled_patches), each (n_kept, n_dirs), for every
    # scene where both halves of the pair exist. With top_frac < 1, only the
    # patches in the styled image's Grad-ECLIP top_frac are kept - the same
    # positions in both images, so row j of each is the same spatial patch.
    for i in scene_idx:
        plain = load_block_patches(sparse_embedding_dir, plain_name(i), block)
        styled = load_block_patches(sparse_embedding_dir, styled_name(i, style), block)
        if plain is None or styled is None:
            continue
        grid_h, grid_w, n_dirs = styled.shape
        if top_frac < 1.0:
            path = mask_path(mask_dir, styled_name(i, style))
            if not os.path.exists(path):
                continue
            pixel_map = np.load(path)["pixel_map"].astype(np.float32)
            keep = top_frac_patch_mask(pixel_map, grid_h, grid_w, top_frac).reshape(-1)
        else:
            keep = np.ones(grid_h * grid_w, dtype=bool)
        yield plain.reshape(-1, n_dirs)[keep], styled.reshape(-1, n_dirs)[keep]


def discover_paired(train_idx: list, sparse_embedding_dir: str, style: str, block: str, top_k: int,
                     feature_selection: str, mask_dir: str = None, top_frac: float = 1.0):
    '''
    Streams over train pairs (never holds more than one pair in memory),
    accumulating per-latent: per-scene image-mean differences (sum and sum of
    squares, for mean_diff/paired_t), aligned-patch win/loss counts (for
    patch_winrate), and styled-patch sum/sum-of-squares (for the chosen
    latent's mean/std, which generate_clean_ablate.py uses for plus/minus_std).
    '''
    n_scenes = 0
    n_patches = 0
    sum_d = sum_d2 = win = lose = styled_sum = styled_sq = None
    for plain, styled in iter_pairs(train_idx, sparse_embedding_dir, style, block, mask_dir, top_frac):
        plain = plain.astype(np.float64)
        styled = styled.astype(np.float64)
        d = styled.mean(axis=0) - plain.mean(axis=0)
        if sum_d is None:
            n_dirs = plain.shape[1]
            sum_d, sum_d2 = np.zeros(n_dirs), np.zeros(n_dirs)
            win, lose = np.zeros(n_dirs), np.zeros(n_dirs)
            styled_sum, styled_sq = np.zeros(n_dirs), np.zeros(n_dirs)
        sum_d += d
        sum_d2 += d * d
        win += (styled > plain).sum(axis=0)
        lose += (styled < plain).sum(axis=0)
        styled_sum += styled.sum(axis=0)
        styled_sq += (styled * styled).sum(axis=0)
        n_scenes += 1
        n_patches += plain.shape[0]

    if n_scenes == 0:
        return None

    mean_d = sum_d / n_scenes
    if feature_selection == "paired_t" and n_scenes >= 2:
        var_d = np.maximum(sum_d2 - n_scenes * mean_d * mean_d, 0.0) / (n_scenes - 1)
        score = mean_d / (np.sqrt(var_d / n_scenes) + 1e-8)
    elif feature_selection == "patch_winrate":
        score = (win - lose) / n_patches
    else:
        if feature_selection == "paired_t":
            print("  only 1 train scene - paired_t falls back to mean_diff")
        score = mean_d

    top_idx = np.argsort(score)[::-1][:top_k].astype(np.int64)
    best_idx = int(top_idx[0])
    styled_mean = styled_sum / n_patches
    styled_std = np.sqrt(np.maximum(styled_sq / n_patches - styled_mean ** 2, 0.0))
    pos_mean, pos_std = float(styled_mean[best_idx]), float(styled_std[best_idx])

    print(f"  {n_scenes} train pairs | top latents {top_idx.tolist()}")
    print(f"  scores {np.round(score[top_idx], 4).tolist()}")
    print(f"  best latent {best_idx}: mean diff={mean_d[best_idx]:.4f} "
          f"| styled activation mean={pos_mean:.4f} std={pos_std:.4f}")

    # same layout as generate_clean_inference's bce/f1: single-latent vector,
    # top_idx[0] is the chosen latent (generate_clean_ablate reads it that way)
    mean_vec = np.zeros(len(score), dtype=np.float32)
    mean_vec[best_idx] = pos_mean
    return {
        "mean_vec": mean_vec,
        "top_idx": top_idx,
        "top_auc": score[top_idx].astype(np.float32),
        "n_pos": n_scenes,
        "n_neg": n_scenes,
        "chosen_pos_mean": pos_mean,
        "chosen_pos_std": pos_std,
    }


def discover_labeled(train_idx: list, sparse_embedding_dir: str, style: str, block: str, args):
    # styled patches positive, plain patches negative, then the exact same
    # selection generate_clean_inference.py uses
    feats_list, labels_list = [], []
    for plain, styled in iter_pairs(train_idx, sparse_embedding_dir, style, block, args.mask_dir,
                                    args.mask_top_frac):
        feats_list += [plain, styled]
        labels_list += [np.zeros(len(plain), dtype=bool), np.ones(len(styled), dtype=bool)]
    if not feats_list:
        return None
    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return select_features(feats, labels, args.top_k, args.feature_selection, args.bce_ridge,
                            args.bce_newton_steps, args.saeuron_percentile)


def evaluate_style_block(test_idx: list, sparse_embedding_dir: str, style: str, block: str,
                          mean_vec: np.ndarray, mask_dir: str = None, top_frac: float = 1.0) -> dict:
    '''
    Projects every held-out patch onto the (normalized) saved vector:
    patch-level AUROC/AP for styled-vs-plain, plus pair_accuracy - the
    fraction of held-out same-seed pairs where the styled image's mean
    projection beats its plain twin's.
    '''
    direction = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
    scores, labels = [], []
    n_pairs = n_correct = 0
    for plain, styled in iter_pairs(test_idx, sparse_embedding_dir, style, block, mask_dir, top_frac):
        s_plain, s_styled = plain @ direction, styled @ direction
        scores += [s_plain, s_styled]
        labels += [np.zeros(len(s_plain), dtype=bool), np.ones(len(s_styled), dtype=bool)]
        n_pairs += 1
        n_correct += int(s_styled.mean() > s_plain.mean())

    metrics = {"n_test_pairs": n_pairs, "pair_accuracy": None, "auroc": None, "ap": None}
    if n_pairs == 0:
        return metrics
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    metrics["pair_accuracy"] = n_correct / n_pairs
    metrics["auroc"] = float(roc_auc_score(labels, scores))
    metrics["ap"] = float(average_precision_score(labels, scores))
    return metrics


def main(args):
    api, accelerator, device = repo_api_init(args)

    for d in [args.image_src_dir, args.embedding_dir, args.sparse_embedding_dir, args.eval_dir, args.mask_dir]:
        os.makedirs(d, exist_ok=True)

    style_list: list = args.style_list
    style_list=[s.replace("_"," ") for s in style_list]
    if not style_list:
        raise ValueError("--style_list is required (e.g. --style_list anime watercolor)")

    block_list = args.block_list if args.block_list else list(DEFAULT_BLOCK_LIST)
    scenes = read_prompts(args.scene_file)
    if args.limit >= 0:
        scenes = scenes[:args.limit]
    print(f"{len(scenes)} scenes x (1 plain + {len(style_list)} styles)")

    if not args.disable_generate:
        generate_pairs_and_cache(args.image_src_dir, args.embedding_dir, scenes, style_list, args.style_template,
                                  block_list, args.size, args.num_inference_steps, args.guidance_scale,
                                  args.mixed_precision, device)

    partition = get_or_make_scene_partition(len(scenes), args.partition_path, args.train_frac, args.seed)
    train_idx, test_idx = partition["train"], partition["test"]
    print(f"partition: {len(train_idx)} train scenes, {len(test_idx)} test scenes")

    if not args.disable_sparsify_embeddings:
        sparsify_embeddings(args.sparse_embedding_dir, args.embedding_dir, args.mode, block_list, args.sae_source)

    if not args.disable_masks and args.mask_top_frac < 1.0:
        cache_grad_eclip_maps(args.image_src_dir, args.mask_dir, list(range(len(scenes))), style_list,
                               args.grad_eclip_text, args.mask_top_frac, args.n_visualize,
                               args.clip_model, args.clip_pretrained, device)

    if not args.disable_discover:
        for style in style_list:
            query = safe_name(style)
            for block in block_list:
                print(f"discovering '{style}' style latent / {block} ({args.feature_selection}) ...")
                if args.feature_selection in PAIRED_MODES:
                    result = discover_paired(train_idx, args.sparse_embedding_dir, style, block, args.top_k,
                                              args.feature_selection, args.mask_dir, args.mask_top_frac)
                else:
                    result = discover_labeled(train_idx, args.sparse_embedding_dir, style, block, args)
                if result is None:
                    print(f"  skipped: no train pairs for '{style}' at {block}")
                    continue
                update_npz_dict(args.npz_dict, query, block, result, args.mode)

                metrics = evaluate_style_block(test_idx, args.sparse_embedding_dir, style, block, result["mean_vec"],
                                                args.mask_dir, args.mask_top_frac)
                metrics["top_idx"] = result["top_idx"].tolist()
                metrics["top_score"] = result["top_auc"].tolist()
                print(f"  test pair_accuracy={metrics['pair_accuracy']} auroc={metrics['auroc']} ap={metrics['ap']}")
                with open(os.path.join(args.eval_dir, f"{query}_{block}_{args.feature_selection}_metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    print_args(parser)
    print_details()
    start = time.time()
    args = parser.parse_args()
    print_args(parser)
    print(args)
    main(args)
    end = time.time()
    seconds = end - start
    hours = seconds / (60 * 60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")
