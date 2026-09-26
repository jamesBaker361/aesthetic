# this trains the sae and then does SAEURON stuyle removal I think?
#
# pipeline: generate images from a prompt file, extracting per-block SDXL UNet
# activations the same way sdxl_unbox/scripts/collect_latents_dataset.py built
# the SAE training data - a single stabilityai/sdxl-turbo run_with_cache call
# per prompt (not a separate generator + sdxl_extract's fake-noise re-encode,
# which is only valid for pre-existing real photos) - then sparsify.sparsify_embeddings
# (same as generate_clean.py) -> for each query in query_list, use SAM3 to find
# which patches of each generated image depict that query, at EVERY block's
# own spatial resolution -> discover the most discriminative SAE latents for
# that query per block and build a per-block "SAE embedding" vector for it -
# saved into a single npz_dict keyed by "{query}__{block}" so generate_clean_swap.py
# can later subtract it (remove the query) or subtract-and-inject another
# query's vector (replace one query with another) at generation time.
#
# --feature_selection controls how that per-block vector is built:
#   "auroc" (default): same technique as generate_clean_patch.py - rank latents
#     by AUROC, keep the top_k, and average the (relu'd) sparse activation over
#     the query's positive patches across all latents.
#   "bce": Eq. (7) / p.5 of "Rediscovering SAEs" (arXiv:2511.17735) - fit a
#     per-latent ridge-regularized 1D logistic regression (ridge=1e-8, 30
#     Newton steps, w0=0, b0=prevalence - the appendix's "1D Probe Training"
#     hyperparameters) and pick the single latent with lowest training binary
#     cross-entropy for the query ("unsupervised feature-to-concept matching").
#     The saved vector is then all-zero except at that one latent.
#   "f1": same per-latent probes as "bce", but pick whichever latent's
#     thresholded (p>=0.5) predictions score highest training F1 instead of
#     lowest BCE - rewards precision/recall balance directly rather than
#     calibrated probability quality.
#   "saeuron": the feature-selection heuristic from "SAeUron: Interpretable
#     Concept Unlearning in Diffusion Models with Sparse Autoencoders"
#     (arXiv:2501.18052), adapted to this repo's patch-level labels instead
#     of their whole-image ones. Scores every latent by normalized mean
#     activation difference, score(i) = mean_pos[i]/sum(mean_pos) -
#     mean_neg[i]/sum(mean_neg), then keeps every latent above the
#     --saeuron_percentile threshold with above-average positive activation
#     (their paper reports this usually converges to ~1-2 features per
#     concept). Unlike "bce"/"f1", this is a heuristic activation-difference
#     ranking, not a fitted probe, and can select more than one latent.
#
# --mask_function controls how each image's positive/negative patches are
# labeled for a given query:
#   "sam3" (default): SAM3 text-prompted segmentation finds which patches of
#     the image actually depict the query concept.
#   "prompt": no segmentation at all - every patch of an image is labeled
#     positive for whichever query's text literally appears in the prompt
#     that generated it (via --prompt_file), and negative for every other
#     query. Meant for concepts SAM3 can't spatially localize, like a style
#     ("anime_style"), where the honest ground truth is "the whole image
#     is/isn't this". Only valid for images generate_and_cache produced
#     itself (relies on the "prompt_{i}.jpg" naming to recover the prompt).

import os
import json
import argparse
import time

import numpy as np
import torch
import random
from PIL import Image
from scipy.stats import rankdata
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from experiment_helpers.init_helpers import default_parser, repo_api_init

from sdxl_unbox.SDLens.hooked_sd_pipeline import HookedStableDiffusionXLPipeline
from sparsify import sparsify_embeddings
from attribution import DEFAULT_BLOCK_LIST

from sam3_repo.sam3.model_builder import build_sam3_image_model
from sam3_repo.sam3.model.sam3_image_processor import Sam3Processor

parser = default_parser(
    {
        "repo_id": "jlbaker361/nsfw"
    }
)

parser.add_argument("--image_src_dir", type=str, default="real_fruit")
parser.add_argument("--embedding_dir", type=str, default="embeddings")
parser.add_argument("--sparse_embedding_dir", type=str, default="sparse_embeddings")
parser.add_argument("--mask_dir", type=str, default="mask_dir")
parser.add_argument("--eval_dir", type=str, default="eval_dir")
parser.add_argument("--npz_dict", type=str, default="platonic.npz")

parser.add_argument("--prompt_file", type=str, default="prompts.txt")

parser.add_argument("--query_list", nargs="*", default=[])

parser.add_argument("--partition_path", type=str, default="partition.json")
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=42)

# num_inference_steps=1, guidance_scale=0.0 match collect_latents_dataset.py
# exactly - that's the distribution the SAE checkpoints were trained on
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--mode", type=str, default="diff")  # fed to sparsify_embeddings - must match what generate_clean_swap.py is told to use

parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--n_visualize", type=int, default=5)

parser.add_argument("--feature_selection", type=str, default="auroc", choices=["auroc", "bce", "f1", "saeuron"],
                     help="'auroc': top_k-AUROC latents + positive-patch averaging (default, generate_clean_patch style). "
                          "'bce': Eq. (7) of arXiv:2511.17735 - single lowest-training-BCE latent per query/block "
                          "via a ridge-regularized 1D logistic regression (see appendix hyperparameters below). "
                          "'f1': same per-latent probes as 'bce', but pick the single highest-training-F1 latent instead. "
                          "'saeuron': SAeUron's (arXiv:2501.18052) normalized-activation-difference percentile "
                          "thresholding instead of a fitted probe - see --saeuron_percentile.")
parser.add_argument("--bce_ridge", type=float, default=1e-8,
                     help="ridge regularization strength for --feature_selection=bce (paper appendix default)")
parser.add_argument("--bce_newton_steps", type=int, default=30,
                     help="number of Newton-method steps for --feature_selection=bce (paper appendix default)")
parser.add_argument("--saeuron_percentile", type=float, default=99.9,
                     help="percentile threshold tau_c for --feature_selection=saeuron (SAeUron's paper uses up to "
                          "99.999 for styles) - only latents above this percentile of the activation-difference "
                          "score, and with above-average positive activation, get selected")

parser.add_argument("--mask_function", type=str, default="sam3", choices=["sam3", "prompt"],
                     help="'sam3' (default): SAM3 text-prompted segmentation finds the query's patches. "
                          "'prompt': whole image is positive/negative based on whether the query's text appears "
                          "in the prompt that generated it - for concepts SAM3 can't spatially localize (e.g. a style).")

parser.add_argument("--sae_source", type=str, default="local", choices=["local", "saeuron"],
                     help="'local': this repo's own trained checkpoints (default). 'saeuron': checkpoints converted "
                          "by convert_saeuron_checkpoint.py from github.com/cywinski/SAeUron - pair with "
                          "--block_list since they don't overlap DEFAULT_BLOCK_LIST.")
parser.add_argument("--block_list", nargs="*", default=None,
                     help="overrides attribution.DEFAULT_BLOCK_LIST - required when --sae_source=saeuron")

parser.add_argument("--disable_generate", action="store_true")
parser.add_argument("--disable_sparsify_embeddings", action="store_true")
parser.add_argument("--disable_masks", action="store_true")
parser.add_argument("--disable_discover", action="store_true")


def read_prompts(prompt_file: str) -> list:
    with open(prompt_file) as f:
        return [line.strip() for line in f if line.strip()]
    
IMAGE_EXTENSIONS=(".jpg",".jpeg",".png",".bmp",".webp")


def list_images(image_src_dir:str)->list:
    return sorted(f for f in os.listdir(image_src_dir) if f.lower().endswith(IMAGE_EXTENSIONS))


    
def get_or_make_partition(image_src_dir:str,partition_path:str,train_frac:float,seed:int)->dict:
    if os.path.exists(partition_path):
        with open(partition_path) as f:
            return json.load(f)

    images=list_images(image_src_dir)
    rng=random.Random(seed)
    rng.shuffle(images)
    n_train=int(round(len(images)*train_frac))
    partition={"train":images[:n_train],"test":images[n_train:]}
    with open(partition_path,"w") as f:
        json.dump(partition,f,indent=2)
    return partition


def generate_and_cache(image_src_dir: str, embedding_dir: str, prompt_file: str, block_list: list,
                        size: int, num_inference_steps: int, guidance_scale: float, mixed_precision: str, device,
                        limit: int = -1):
    '''
    Generates one image per prompt AND caches each block's UNet input/output
    for that same generation, in a single stabilityai/sdxl-turbo pass - exactly
    how sdxl_unbox/scripts/collect_latents_dataset.py built the data the SAE
    checkpoints were trained on (run_with_cache with num_inference_steps=1,
    guidance_scale=0.0), rather than generating with an unrelated model and
    re-deriving activations via a fake-noise re-encode (that trick is only for
    pre-existing real photos - see sdxl_extract.extract_vanilla).
    '''
    print("generate images from prompts + cache block activations")
    os.makedirs(image_src_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)
    prompts = read_prompts(prompt_file)

    dtype = torch.float16 if (torch.cuda.is_available() and mixed_precision == "fp16") else torch.float32
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=dtype,
        variant=("fp16" if dtype == torch.float16 else None),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    positions = [f"unet.{block}" for block in block_list]
    if limit >= 0:
        prompts = prompts[:limit]

    for i, prompt in enumerate(prompts):
        image_path = os.path.join(image_src_dir, f"prompt_{i}.jpg")
        npz_path = os.path.join(embedding_dir, f"prompt_{i}.jpg.npz")
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
            # cache tensors are stacked over denoising steps at dim=1:
            # (batch, steps, C, H, W) - keep the last step, closest to the
            # finished image, matching the (1,C,H,W) shape sparsify.py expects
            result[f"saved_input.{block}"] = cache["input"][pos][:, -1].cpu().float().numpy()
            result[f"saved_output.{block}"] = cache["output"][pos][:, -1].cpu().float().numpy()
        np.savez(npz_path, **result)


def get_query_pixel_mask(image: Image.Image, query: str, sam3_processor) -> np.ndarray:
    inference_state = sam3_processor.set_image(image)
    output = sam3_processor.set_text_prompt(state=inference_state, prompt=query)
    masks, scores = output["masks"], output["scores"]
    w, h = image.size
    if len(scores) == 0:
        return np.zeros((h, w), dtype=bool)
    masks_np = masks.squeeze(1).cpu().numpy()
    return np.any(masks_np, axis=0)  # union of every returned mask

def mask_out_path(mask_dir:str,name:str,target_query:str)->str:
    safe_query=target_query.replace(" ","_")
    return os.path.join(mask_dir,f"{name}.{safe_query}.npz")

def cache_query_masks(images: list, image_src_dir: str, mask_dir: str, query: str, sam3_processor):
    for name in images:
        out_path = mask_out_path(mask_dir, name, query)
        if os.path.exists(out_path):
            continue
        image = Image.open(os.path.join(image_src_dir, name)).convert("RGB")
        pixel_mask = get_query_pixel_mask(image, query, sam3_processor)
        np.savez(out_path, pixel_mask=pixel_mask)


def image_name_to_prompt_index(name: str) -> int:
    # generate_and_cache names every image "prompt_{i}.jpg" (i = its line
    # number in --prompt_file) - recovering the prompt just means parsing
    # that index back out of the filename
    stem = os.path.splitext(name)[0]
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        raise ValueError(
            f"'{name}' doesn't look like a generate_and_cache output (\"prompt_{{i}}.jpg\") - "
            "--mask_function=prompt only works on images this script generated itself"
        )


def get_whole_image_mask(image: Image.Image, prompt: str, query: str) -> np.ndarray:
    '''
    Alternative to SAM3 segmentation: labels every patch of the image the
    same way, based solely on whether `query` was the concept that actually
    prompted it - all-positive if `query` (read with underscores as spaces)
    appears in the prompt text, all-negative otherwise. Useful for concepts
    SAM3 can't spatially localize, like a style ("anime_style"), where the
    honest ground truth is "the whole image is/isn't this".
    '''
    w, h = image.size
    is_positive = query.replace("_", " ").lower() in prompt.lower()
    return np.full((h, w), is_positive, dtype=bool)


def cache_whole_image_masks(images: list, image_src_dir: str, mask_dir: str, query: str, prompts: list):
    for name in images:
        out_path = mask_out_path(mask_dir, name, query)
        if os.path.exists(out_path):
            continue
        image = Image.open(os.path.join(image_src_dir, name)).convert("RGB")
        prompt = prompts[image_name_to_prompt_index(name)]
        pixel_mask = get_whole_image_mask(image, prompt, query)
        np.savez(out_path, pixel_mask=pixel_mask)


def resize_mask_to_grid(pixel_mask: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    # exact-divide reshape+mean (a patch counts "on-target" if the majority of
    # its pixels are inside the SAM3 mask) - mirrors both generate_clean_patch's
    # DINOv2 patch_mask calc and sdxl_extract's own scale=pixels//grid math
    h, w = pixel_mask.shape
    scale_h, scale_w = h // grid_h, w // grid_w
    cropped = pixel_mask[:grid_h * scale_h, :grid_w * scale_w]
    return cropped.reshape(grid_h, scale_h, grid_w, scale_w).mean(axis=(1, 3)) > 0.5


def load_block_feats_and_labels(images: list, sparse_embedding_dir: str, mask_dir: str, query: str, block: str):
    feats_list, labels_list = [], []
    for name in images:
        sparse_path = os.path.join(sparse_embedding_dir, name + ".npz")
        mask_path = mask_out_path(mask_dir, name, query)
        if not (os.path.exists(sparse_path) and os.path.exists(mask_path)):
            continue
        block_feats = np.load(sparse_path)[block]  # (h,w,n_dirs)
        grid_h, grid_w, n_dirs = block_feats.shape
        pixel_mask = np.load(mask_path)["pixel_mask"]
        patch_mask = resize_mask_to_grid(pixel_mask, grid_h, grid_w)
        feats_list.append(block_feats.reshape(-1, n_dirs))
        labels_list.append(patch_mask.reshape(-1))
    if not feats_list:
        return None, None
    return np.concatenate(feats_list, axis=0), np.concatenate(labels_list, axis=0)


def fit_1d_ridge_logistic(feats: np.ndarray, labels: np.ndarray, ridge: float, n_newton_steps: int):
    """
    Per-latent 1D ridge-regularized logistic regression, fit for every latent
    at once via Newton's method (closed-form 2x2 Hessian inverse per latent).
    Matches Eq. (7) and the "1D Probe Training" hyperparameters in the
    appendix of "Rediscovering SAEs" (arXiv:2511.17735): w0=0, b0=prevalence,
    solved with a handful of Newton steps under a small ridge penalty on w.

    feats: (n_samples, n_dirs); labels: (n_samples,) 0/1, shared across latents.
    Returns w, b, loss - each (n_dirs,) - loss is the unregularized mean
    binary cross-entropy of the fitted probe on the training set (Eq. (7)).
    """
    y = labels.astype(np.float64)
    z = feats.astype(np.float64)
    n_dirs = z.shape[1]
    prevalence = y.mean()

    w = np.zeros(n_dirs, dtype=np.float64)
    b = np.full(n_dirs, prevalence, dtype=np.float64)

    for _ in range(n_newton_steps):
        p = expit(w[None, :] * z + b[None, :])
        resid = p - y[:, None]
        s = p * (1.0 - p)

        g_w = np.einsum("ij,ij->j", resid, z) + 2.0 * ridge * w
        g_b = resid.sum(axis=0)

        h_ww = np.einsum("ij,ij->j", s, z * z) + 2.0 * ridge
        h_wb = np.einsum("ij,ij->j", s, z)
        h_bb = s.sum(axis=0)

        det = h_ww * h_bb - h_wb * h_wb
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)

        w = w - (h_bb * g_w - h_wb * g_b) / det
        b = b - (h_ww * g_b - h_wb * g_w) / det

    p = np.clip(expit(w[None, :] * z + b[None, :]), 1e-12, 1 - 1e-12)
    loss = -(y[:, None] * np.log(p) + (1 - y[:, None]) * np.log(1 - p)).mean(axis=0)

    return w, b, loss


def per_latent_f1_at_threshold(p: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Vectorized per-latent F1 at a fixed probability threshold. p is
    (n_samples, n_dirs) (e.g. fit_1d_ridge_logistic's fitted per-latent
    probes), labels is (n_samples,) shared across latents. Zero-division (no
    predicted positives, or no actual positives) scores 0, matching sklearn's
    zero_division=0 convention.
    """
    pred = p >= threshold
    y = labels[:, None]
    tp = (pred & y).sum(axis=0).astype(np.float64)
    fp = (pred & ~y).sum(axis=0).astype(np.float64)
    fn = (~pred & y).sum(axis=0).astype(np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return f1


def _best_latent_mean_vec(feats: np.ndarray, labels: np.ndarray, w: np.ndarray, b: np.ndarray,
                           best_idx: int) -> tuple:
    """
    Shared by the "bce" and "f1" feature_selection modes (both fit the same
    per-latent probes via fit_1d_ridge_logistic, just picking best_idx by a
    different criterion): prints precision/recall/F1 + activation stats for
    the chosen latent, and returns (mean_vec, pos_mean, pos_std) - mean_vec is
    the SAE embedding, all-zero except at best_idx, set to its mean activation
    over the positive (on-target) patches; pos_mean/pos_std are that same
    latent's activation mean/std over the positive patches, for the caller to
    persist alongside it.
    """
    best_feat = feats[:, best_idx].astype(np.float64)
    p_best = expit(w[best_idx] * best_feat + b[best_idx])
    pred_best = p_best >= 0.5
    precision = precision_score(labels, pred_best, zero_division=0)
    recall = recall_score(labels, pred_best, zero_division=0)
    f1 = f1_score(labels, pred_best, zero_division=0)
    pos_mean = float(feats[labels, best_idx].mean())
    pos_std = float(feats[labels, best_idx].std())
    print(f"  best latent {best_idx}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
          f"| positive activation mean={pos_mean:.4f} std={pos_std:.4f}")

    mean_vec = np.zeros(feats.shape[1], dtype=np.float32)
    mean_vec[best_idx] = pos_mean
    return mean_vec, pos_mean, pos_std


def discover_query_block(train_images: list, sparse_embedding_dir: str, mask_dir: str, query: str, block: str,
                          top_k: int, feature_selection: str = "auroc", bce_ridge: float = 1e-8,
                          bce_newton_steps: int = 30, saeuron_percentile: float = 99.9):
    feats, labels = load_block_feats_and_labels(train_images, sparse_embedding_dir, mask_dir, query, block)
    if feats is None:
        return None
    return select_features(feats, labels, top_k, feature_selection, bce_ridge, bce_newton_steps, saeuron_percentile)


def select_features(feats: np.ndarray, labels: np.ndarray, top_k: int, feature_selection: str = "auroc",
                     bce_ridge: float = 1e-8, bce_newton_steps: int = 30, saeuron_percentile: float = 99.9):
    # feats: (n_patches, n_dirs), labels: (n_patches,) bool - split out of
    # discover_query_block so generate_clean_style.py can reuse it on its own
    # paired plain/styled labels
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    chosen_pos_mean = None
    chosen_pos_std = None

    if feature_selection == "bce":
        # Eq. (7): fit every latent's 1D probe and keep only the single
        # feature with lowest training BCE for this concept - "unsupervised
        # feature-to-concept matching" - instead of a top_k AUROC shortlist.
        w, b, loss = fit_1d_ridge_logistic(feats, labels, bce_ridge, bce_newton_steps)
        print(f"  per-latent BCE loss: mean={loss.mean():.4f} median={np.median(loss):.4f} "
              f"min={loss.min():.4f} max={loss.max():.4f} std={loss.std():.4f}")

        best_idx = int(np.argmin(loss))
        top_idx = np.array([best_idx], dtype=np.int64)
        top_score = loss[[best_idx]].astype(np.float32)  # lower is better (this is a loss, not an AUROC)
        mean_vec, chosen_pos_mean, chosen_pos_std = _best_latent_mean_vec(feats, labels, w, b, best_idx)
    elif feature_selection == "f1":
        # same per-latent probes as "bce", but pick whichever latent's
        # thresholded (p>=0.5) predictions score highest F1 instead of
        # lowest training BCE
        w, b, _ = fit_1d_ridge_logistic(feats, labels, bce_ridge, bce_newton_steps)
        p = expit(w[None, :] * feats.astype(np.float64) + b[None, :])
        f1_per_latent = per_latent_f1_at_threshold(p, labels)
        print(f"  per-latent F1: mean={f1_per_latent.mean():.4f} median={np.median(f1_per_latent):.4f} "
              f"min={f1_per_latent.min():.4f} max={f1_per_latent.max():.4f} std={f1_per_latent.std():.4f}")

        best_idx = int(np.argmax(f1_per_latent))
        top_idx = np.array([best_idx], dtype=np.int64)
        top_score = f1_per_latent[[best_idx]].astype(np.float32)  # higher is better (this is F1, not a loss)
        mean_vec, chosen_pos_mean, chosen_pos_std = _best_latent_mean_vec(feats, labels, w, b, best_idx)
    elif feature_selection == "saeuron":
        # SAeUron (arXiv:2501.18052): normalized mean-activation difference
        # between positive and negative patches, rather than a fitted probe.
        # score(i) = mean_pos[i]/sum(mean_pos) - mean_neg[i]/sum(mean_neg)
        mean_pos = feats[labels].mean(axis=0).astype(np.float64)
        mean_neg = feats[~labels].mean(axis=0).astype(np.float64)
        sum_pos, sum_neg = mean_pos.sum(), mean_neg.sum()
        norm_pos = mean_pos / sum_pos if sum_pos > 0 else np.zeros_like(mean_pos)
        norm_neg = mean_neg / sum_neg if sum_neg > 0 else np.zeros_like(mean_neg)
        score = norm_pos - norm_neg
        print(f"  SAeUron score: mean={score.mean():.6f} median={np.median(score):.6f} "
              f"min={score.min():.6f} max={score.max():.6f} std={score.std():.6f}")

        # the paper's selection condition: above the percentile threshold AND
        # above-average positive activation (filters out latents whose score
        # is only high because they're near-inactive on both sides)
        threshold = np.percentile(score, saeuron_percentile)
        candidates = np.where((score >= threshold) & (mean_pos > mean_pos.mean()))[0]
        if len(candidates) == 0:
            # threshold too strict for this query/block - fall back to the
            # single highest-scoring latent so a feature always gets picked,
            # matching bce/f1's guarantee of never returning empty-handed
            candidates = np.array([int(np.argmax(score))])

        order = np.argsort(score[candidates])[::-1]
        top_idx = candidates[order].astype(np.int64)
        top_score = score[top_idx].astype(np.float32)
        print(f"  selected {len(top_idx)} feature(s) above the {saeuron_percentile} percentile: "
              f"{top_idx.tolist()} (scores {top_score.tolist()})")

        # the paper ablates every selected feature together (a set, not
        # necessarily just one) - mean_vec is zero except at those latents
        mean_vec = np.zeros(feats.shape[1], dtype=np.float32)
        mean_vec[top_idx] = mean_pos[top_idx]
    else:
        # per-latent AUROC via rank-sum form of Mann-Whitney U (same technique
        # as generate_clean_patch.discover_top_features) - avoids one
        # roc_auc_score call per latent when there can be thousands of them
        ranks = rankdata(feats, axis=0)
        sum_ranks_pos = ranks[labels].sum(axis=0)
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

        top_idx = np.argsort(auc)[::-1][:top_k].astype(np.int64)
        top_score = auc[top_idx].astype(np.float32)

        # "the SAE embedding" for this query/block: mean of the (already
        # relu'd, top-k-sparse) latent vectors over the positive (on-target)
        # patches - this is what generate_clean_swap.py subtracts/injects at
        # generation time
        mean_vec = feats[labels].mean(axis=0).astype(np.float32)

    return {
        "mean_vec": mean_vec,
        "top_idx": top_idx,
        "top_auc": top_score,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "chosen_pos_mean": chosen_pos_mean,
        "chosen_pos_std": chosen_pos_std,
    }


def update_npz_dict(npz_dict_path: str, query: str, block: str, result: dict, mode: str):
    existing = {}
    if os.path.exists(npz_dict_path):
        with np.load(npz_dict_path, allow_pickle=True) as data:
            existing = {k: data[k] for k in data.files}
    key = f"{query}__{block}"
    existing[key] = result["mean_vec"]
    existing[f"{key}__topk_idx"] = result["top_idx"]
    existing[f"{key}__topk_auc"] = result["top_auc"]
    # only "bce"/"f1" feature_selection picks a single chosen latent - "auroc"
    # averages over a top_k shortlist, so there's no one feature's mean/std to save
    if result["chosen_pos_mean"] is not None:
        existing[f"{key}__chosen_pos_mean"] = np.float32(result["chosen_pos_mean"])
        existing[f"{key}__chosen_pos_std"] = np.float32(result["chosen_pos_std"])
    existing["__meta_mode__"] = np.array(mode)
    np.savez(npz_dict_path, **existing)


def evaluate_query_block(test_images: list, sparse_embedding_dir: str, mask_dir: str, query: str, block: str,
                          mean_vec: np.ndarray, eval_dir: str) -> dict:
    feats, labels = load_block_feats_and_labels(test_images, sparse_embedding_dir, mask_dir, query, block)
    metrics = {"n_test_images": len(test_images), "auroc": None, "ap": None}
    if feats is None or labels.sum() == 0 or labels.sum() == len(labels):
        return metrics

    norm = np.linalg.norm(mean_vec) + 1e-8
    scores = feats @ (mean_vec / norm)

    metrics["auroc"] = float(roc_auc_score(labels, scores))
    metrics["ap"] = float(average_precision_score(labels, scores))
    return metrics


def main(args):
    api, accelerator, device = repo_api_init(args)

    image_src_dir: str = args.image_src_dir
    embedding_dir: str = args.embedding_dir
    sparse_embedding_dir: str = args.sparse_embedding_dir
    mask_dir: str = args.mask_dir
    eval_dir: str = args.eval_dir

    for d in [image_src_dir, embedding_dir, sparse_embedding_dir, mask_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    query_list: list = args.query_list
    if not query_list:
        raise ValueError("--query_list is required (e.g. --query_list banana orange)")

    block_list = args.block_list if args.block_list else list(DEFAULT_BLOCK_LIST)

    if not args.disable_generate:
        generate_and_cache(image_src_dir, embedding_dir, args.prompt_file, block_list,
                            args.size, args.num_inference_steps, args.guidance_scale, args.mixed_precision, device,
                            args.limit)

    partition = get_or_make_partition(image_src_dir, args.partition_path, args.train_frac, args.seed)
    train_images, test_images = partition["train"], partition["test"]
    all_images = train_images + test_images
    print(f"partition: {len(train_images)} train images, {len(test_images)} test images")

    if not args.disable_sparsify_embeddings:
        sparsify_embeddings(sparse_embedding_dir, embedding_dir, args.mode, block_list, args.sae_source)

    if not args.disable_masks:
        if args.mask_function == "prompt":
            prompts = read_prompts(args.prompt_file)
            for query in query_list:
                print(f"labeling whole-image masks for '{query}' from prompt text...")
                cache_whole_image_masks(all_images, image_src_dir, mask_dir, query, prompts)
        else:
            if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            sam3_model = build_sam3_image_model()
            sam3_processor = Sam3Processor(sam3_model, device=device)
            for query in query_list:
                print(f"extracting SAM3 masks for '{query}'...")
                cache_query_masks(all_images, image_src_dir, mask_dir, query, sam3_processor)

    if not args.disable_discover:
        for query in query_list:
            for block in block_list:
                print(f"discovering SAE embedding for '{query}' / {block} ...")
                result = discover_query_block(train_images, sparse_embedding_dir, mask_dir, query, block, args.top_k,
                                               args.feature_selection, args.bce_ridge, args.bce_newton_steps,
                                               args.saeuron_percentile)
                if result is None:
                    print(f"  skipped: no positive/negative patch contrast for '{query}' at {block}")
                    continue
                print(f"  {result['n_pos']} positive / {result['n_neg']} negative patches, "
                      f"top latents {result['top_idx'].tolist()}")
                update_npz_dict(args.npz_dict, query, block, result, args.mode)

                metrics = evaluate_query_block(test_images, sparse_embedding_dir, mask_dir, query, block,
                                                result["mean_vec"], eval_dir)
                print(f"  test auroc={metrics['auroc']} ap={metrics['ap']}")
                safe_query = query.replace(" ", "_")
                with open(os.path.join(eval_dir, f"{safe_query}_{block}_metrics.json"), "w") as f:
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
