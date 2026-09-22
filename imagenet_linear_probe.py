# Investigates which ImageNet classes are most linearly separable in each
# UNet block's SAE feature space. Pipeline (each stage toggleable via
# --disable_X, and each stage skips work that's already cached on disk -
# same conventions as generate_clean_inference.py):
#
# 1. fetch_imagenet_classes: download the 1000 ImageNet class names from the
#    Waikato class-map page into --class_list_path (skip if it exists).
# 2. generate_and_cache_classes: generate --n_per_class images per class
#    ("a photo of a {class}") with SDXL-turbo, caching each of
#    attribution.DEFAULT_BLOCK_LIST's raw UNet block input/output the same
#    way generate_clean_inference.generate_and_cache does - then
#    sparsify.sparsify_embeddings turns those into SAE features.
# 3. cache_query_masks (reused from generate_clean_inference.py) finds,
#    per class per image, which patches SAM3 says actually depict it.
# 3.5. filter_classes_by_patch_count: per block, drop any class with fewer
#    than --min_patches positive patches across all its images.
# 4. train_class_block_probe: per (class, block) that survives filtering,
#    fit a ridge-regularized multi-dimensional logistic regression (sklearn
#    - Newton's method doesn't scale to hidden_dim~5120 features) on that
#    class's positive/negative patches, plus --n_extra_negatives patches
#    randomly sampled from OTHER classes' images as additional negatives
#    (patches that don't depict this class at all, not just "background of
#    this class's own images").
# 5. write_report: rank every (class, block) pair by held-out probe
#    accuracy - "most linearly separable" = highest accuracy.

import os
import re
import ast
import csv
import json
import time
import urllib.request

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from experiment_helpers.init_helpers import default_parser, repo_api_init

from sdxl_unbox.SDLens.hooked_sd_pipeline import HookedStableDiffusionXLPipeline
from sparsify import sparsify_embeddings
from attribution import DEFAULT_BLOCK_LIST

from sam3_repo.sam3.model_builder import build_sam3_image_model
from sam3_repo.sam3.model.sam3_image_processor import Sam3Processor

from generate_clean_inference import (
    get_query_pixel_mask, cache_query_masks, resize_mask_to_grid, load_block_feats_and_labels,
)

# the Waikato class-map page 403s from this cluster's network (bot-blocked);
# this gist's raw file is a plain Python dict literal {idx: "name, synonym, ..."}
# for all 1000 classes and fetches fine - see fetch_imagenet_classes
IMAGENET_CLASSMAP_URL = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/imagenet1000_clsidx_to_labels.txt"

parser = default_parser(
    {
        "repo_id": "jlbaker361/nsfw"
    }
)

parser.add_argument("--class_list_path", type=str, default="imagenet_classes.txt")
parser.add_argument("--image_src_dir", type=str, default="imagenet_probe_images")
parser.add_argument("--embedding_dir", type=str, default="imagenet_probe_embeddings")
parser.add_argument("--sparse_embedding_dir", type=str, default="imagenet_probe_sparse")
parser.add_argument("--mask_dir", type=str, default="imagenet_probe_masks")
parser.add_argument("--probe_dir", type=str, default="imagenet_probe_results")
parser.add_argument("--report_path", type=str, default="imagenet_linear_separability.csv")

parser.add_argument("--n_per_class", type=int, default=10)
parser.add_argument("--min_patches", type=int, default=50)  # step 3.5 threshold
parser.add_argument("--n_extra_negatives", type=int, default=500)  # cross-class negatives per probe
parser.add_argument("--test_frac", type=float, default=0.25)
parser.add_argument("--probe_C", type=float, default=1.0)  # inverse ridge strength (sklearn convention)
parser.add_argument("--probe_max_iter", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)

# num_inference_steps=1, guidance_scale=0.0 match collect_latents_dataset.py
# exactly - that's the distribution the SAE checkpoints were trained on
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--mode", type=str, default="diff")  # fed to sparsify_embeddings

parser.add_argument("--disable_class_list", action="store_true")
parser.add_argument("--disable_generate", action="store_true")
parser.add_argument("--disable_sparsify_embeddings", action="store_true")
parser.add_argument("--disable_masks", action="store_true")
parser.add_argument("--disable_probe", action="store_true")


def fetch_imagenet_classes(class_list_path: str) -> list:
    '''
    Step 1: the 1000 ImageNet class names, in class-index order, parsed from
    the gist's raw {idx: "name, synonym, ..."} dict literal - cached to
    class_list_path (one class per line) so this only ever downloads once.
    The primary (first, comma-separated) name is used as both the generation
    prompt's subject and the SAM3/probe query text.
    '''
    if os.path.exists(class_list_path):
        return read_class_list(class_list_path)

    print(f"downloading ImageNet class list from {IMAGENET_CLASSMAP_URL}")
    req = urllib.request.Request(IMAGENET_CLASSMAP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    # the gist is a literal Python dict {int: str} - ast.literal_eval parses
    # only literals (unlike eval), so this is safe even though the content
    # comes from a URL rather than a fixed local file
    by_index = ast.literal_eval(text)
    if len(by_index) != 1000:
        raise ValueError(f"expected 1000 ImageNet classes, parsed {len(by_index)} from {IMAGENET_CLASSMAP_URL}")

    classes = [by_index[i].split(",")[0].strip() for i in range(1000)]

    with open(class_list_path, "w") as f:
        f.write("\n".join(classes) + "\n")
    return classes


def read_class_list(class_list_path: str) -> list:
    with open(class_list_path) as f:
        return [line.strip() for line in f if line.strip()]


def image_name(class_idx: int, k: int) -> str:
    return f"class{class_idx:04d}_img{k:02d}.jpg"


_IMAGE_NAME_RE = re.compile(r"class(\d+)_img(\d+)")


def parse_image_name(name: str):
    m = _IMAGE_NAME_RE.match(name)
    if not m:
        raise ValueError(f"'{name}' doesn't look like an imagenet_linear_probe output (\"class{{i}}_img{{k}}.jpg\")")
    return int(m.group(1)), int(m.group(2))


def generate_and_cache_classes(image_src_dir: str, embedding_dir: str, classes: list, block_list: list,
                                n_per_class: int, size: int, num_inference_steps: int, guidance_scale: float,
                                mixed_precision: str, device, seed: int):
    '''
    Step 2: n_per_class images of "a photo of a {class}" per ImageNet class,
    caching each block's raw UNet input/output for that same generation -
    identical technique to generate_clean_inference.generate_and_cache, just
    looping over (class, k) instead of a flat prompt list.
    '''
    print("generate per-class images + cache block activations")
    os.makedirs(image_src_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    dtype = torch.float16 if (torch.cuda.is_available() and mixed_precision == "fp16") else torch.float32
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=dtype,
        variant=("fp16" if dtype == torch.float16 else None),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    positions = [f"unet.{block}" for block in block_list]

    for class_idx, class_name in enumerate(classes):
        prompt = f"a photo of a {class_name}"
        for k in range(n_per_class):
            name = image_name(class_idx, k)
            image_path = os.path.join(image_src_dir, name)
            npz_path = os.path.join(embedding_dir, name + ".npz")
            if os.path.exists(image_path) and os.path.exists(npz_path):
                continue

            generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_000 + class_idx * 1000 + k)
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


def list_class_images(image_src_dir: str, classes: list) -> dict:
    '''class_idx -> sorted list of its own image filenames actually on disk'''
    by_class = {idx: [] for idx in range(len(classes))}
    if not os.path.isdir(image_src_dir):
        return by_class
    for name in sorted(os.listdir(image_src_dir)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            class_idx, _k = parse_image_name(name)
        except ValueError:
            continue
        if class_idx in by_class:
            by_class[class_idx].append(name)
    return by_class


def cache_all_class_masks(images_by_class: dict, classes: list, image_src_dir: str, mask_dir: str, sam3_processor,
                           device):
    '''
    Step 3: SAM3 patch masks, per class, over that class's own images
    (reuses generate_clean_inference.cache_query_masks). Some of SAM3's own
    layers run in bf16 internally while its weights stay float32, so calls
    into it need to run under bf16 autocast - without it, F.linear raises
    "mat1 and mat2 must have the same dtype, but got BFloat16 and Float".
    '''
    on_cuda = device == "cuda" or (hasattr(device, "type") and device.type == "cuda")
    for class_idx, images in images_by_class.items():
        if not images:
            continue
        print(f"extracting SAM3 masks for '{classes[class_idx]}' ({len(images)} images)...")
        if on_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cache_query_masks(images, image_src_dir, mask_dir, classes[class_idx], sam3_processor)
        else:
            cache_query_masks(images, image_src_dir, mask_dir, classes[class_idx], sam3_processor)


def filter_classes_by_patch_count(images_by_class: dict, classes: list, sparse_embedding_dir: str, mask_dir: str,
                                   block_list: list, min_patches: int, counts_path: str) -> dict:
    '''
    Step 3.5: per block, drop any class with fewer than min_patches total
    positive (SAM3-matched) patches across all its own images - not enough
    signal to train or trust a probe on. Cached to counts_path so re-runs
    don't redo the patch counting.
    '''
    if os.path.exists(counts_path):
        with open(counts_path) as f:
            counts = json.load(f)
    else:
        counts = {}
        for block in block_list:
            counts[block] = {}
            for class_idx, images in images_by_class.items():
                if not images:
                    counts[block][str(class_idx)] = 0
                    continue
                _, labels = load_block_feats_and_labels(images, sparse_embedding_dir, mask_dir,
                                                          classes[class_idx], block)
                counts[block][str(class_idx)] = 0 if labels is None else int(labels.sum())
        with open(counts_path, "w") as f:
            json.dump(counts, f, indent=2)

    kept = {
        block: [idx for idx in images_by_class if counts[block].get(str(idx), 0) >= min_patches]
        for block in block_list
    }
    for block in block_list:
        print(f"{block}: {len(kept[block])}/{len(images_by_class)} classes have >= {min_patches} positive patches")
    return kept


def sample_cross_class_negatives(class_idx: int, images_by_class: dict, sparse_embedding_dir: str, block: str,
                                  n_extra_negatives: int, rng: np.random.Generator) -> np.ndarray:
    '''
    Step 4's extra negative samples: patches from OTHER classes' images (any
    class != class_idx) - none of them depict class_idx at all regardless of
    whether they were positive for their OWN class, so every one of their
    patches is a valid negative here.
    '''
    other_indices = [idx for idx, images in images_by_class.items() if idx != class_idx and images]
    rng.shuffle(other_indices)

    pool = []
    pool_size = 0
    for idx in other_indices:
        for name in images_by_class[idx]:
            sparse_path = os.path.join(sparse_embedding_dir, name + ".npz")
            if not os.path.exists(sparse_path):
                continue
            block_feats = np.load(sparse_path)[block]
            pool.append(block_feats.reshape(-1, block_feats.shape[-1]))
            pool_size += pool[-1].shape[0]
        if pool_size >= n_extra_negatives * 4:  # gathered a large-enough pool to subsample from
            break

    if not pool:
        return np.empty((0, 0), dtype=np.float32)
    all_feats = np.concatenate(pool, axis=0)
    if all_feats.shape[0] > n_extra_negatives:
        sel = rng.choice(all_feats.shape[0], size=n_extra_negatives, replace=False)
        all_feats = all_feats[sel]
    return all_feats


def probe_result_path(probe_dir: str, class_idx: int, block: str) -> str:
    safe_block = block.replace(".", "_")
    return os.path.join(probe_dir, f"class{class_idx:04d}_{safe_block}.json")


def train_class_block_probe(class_idx: int, classes: list, images_by_class: dict, sparse_embedding_dir: str,
                             mask_dir: str, block: str, n_extra_negatives: int, test_frac: float, probe_C: float,
                             probe_max_iter: int, rng: np.random.Generator):
    '''
    Step 4: a ridge-regularized (L2) multi-dimensional logistic regression
    over the full SAE feature vector - "is this patch class_idx or not" -
    trained on this class's own positive/negative patches plus
    n_extra_negatives patches sampled from other classes' images. Full
    dimensionality (hidden_dim~5120) rules out the per-latent closed-form
    Newton's method generate_clean_inference.fit_1d_ridge_logistic uses for
    concept matching (that's hidden_dim independent 2x2 solves); this needs
    one hidden_dim x hidden_dim problem, which is what sklearn's solver is for.
    '''
    query = classes[class_idx]
    feats_in, labels_in = load_block_feats_and_labels(images_by_class[class_idx], sparse_embedding_dir, mask_dir,
                                                        query, block)
    if feats_in is None:
        return None

    extra_neg = sample_cross_class_negatives(class_idx, images_by_class, sparse_embedding_dir, block,
                                              n_extra_negatives, rng)
    if extra_neg.size:
        X = np.concatenate([feats_in, extra_neg], axis=0)
        y = np.concatenate([labels_in, np.zeros(extra_neg.shape[0], dtype=bool)])
    else:
        X, y = feats_in, labels_in

    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=int(rng.integers(0, 2 ** 31 - 1)), stratify=y
    )

    clf = LogisticRegression(C=probe_C, max_iter=probe_max_iter, class_weight="balanced")  # l2/ridge is the default penalty
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.decision_function(X_test)
    multi_class_test = len(set(y_test.tolist())) > 1

    return {
        "class": query,
        "class_idx": class_idx,
        "block": block,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_extra_negatives": int(extra_neg.shape[0]) if extra_neg.size else 0,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auroc": float(roc_auc_score(y_test, y_score)) if multi_class_test else None,
        "ap": float(average_precision_score(y_test, y_score)) if multi_class_test else None,
    }


def write_report(probe_dir: str, block_list: list, classes: list, report_path: str):
    '''Step 5: gather every cached per-(class,block) probe result and rank by held-out accuracy - highest first.'''
    rows = []
    for block in block_list:
        for class_idx in range(len(classes)):
            path = probe_result_path(probe_dir, class_idx, block)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                rows.append(json.load(f))

    rows.sort(key=lambda r: r["accuracy"], reverse=True)

    fieldnames = ["class", "class_idx", "block", "accuracy", "auroc", "ap", "n_pos", "n_neg", "n_extra_negatives",
                  "n_train", "n_test"]
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} (class, block) probe results to {report_path}")
    print("most linearly separable:")
    for row in rows[:20]:
        print(f"  acc={row['accuracy']:.3f}  '{row['class']}' @ {row['block']}")


def main(args):
    api, accelerator, device = repo_api_init(args)
    rng = np.random.default_rng(args.seed)

    for d in [args.image_src_dir, args.embedding_dir, args.sparse_embedding_dir, args.mask_dir, args.probe_dir]:
        os.makedirs(d, exist_ok=True)

    block_list = list(DEFAULT_BLOCK_LIST)

    classes = (fetch_imagenet_classes(args.class_list_path) if not args.disable_class_list
               else read_class_list(args.class_list_path))
    if args.limit >= 0:
        classes = classes[:args.limit]
    print(f"{len(classes)} ImageNet classes")

    if not args.disable_generate:
        generate_and_cache_classes(args.image_src_dir, args.embedding_dir, classes, block_list, args.n_per_class,
                                    args.size, args.num_inference_steps, args.guidance_scale, args.mixed_precision,
                                    device, args.seed)

    if not args.disable_sparsify_embeddings:
        sparsify_embeddings(args.sparse_embedding_dir, args.embedding_dir, args.mode)

    images_by_class = list_class_images(args.image_src_dir, classes)

    if not args.disable_masks:
        if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        sam3_model = build_sam3_image_model()
        sam3_processor = Sam3Processor(sam3_model, device=device)
        cache_all_class_masks(images_by_class, classes, args.image_src_dir, args.mask_dir, sam3_processor, device)

    if not args.disable_probe:
        counts_path = os.path.join(args.probe_dir, "class_patch_counts.json")
        kept_by_block = filter_classes_by_patch_count(images_by_class, classes, args.sparse_embedding_dir,
                                                        args.mask_dir, block_list, args.min_patches, counts_path)

        for block in block_list:
            for class_idx in kept_by_block[block]:
                out_path = probe_result_path(args.probe_dir, class_idx, block)
                if os.path.exists(out_path):
                    continue
                print(f"training probe: '{classes[class_idx]}' @ {block} ...")
                result = train_class_block_probe(class_idx, classes, images_by_class, args.sparse_embedding_dir,
                                                  args.mask_dir, block, args.n_extra_negatives, args.test_frac,
                                                  args.probe_C, args.probe_max_iter, rng)
                if result is None:
                    print("  skipped: no positive/negative contrast")
                    continue
                print(f"  accuracy={result['accuracy']:.3f} auroc={result['auroc']} "
                      f"n_pos={result['n_pos']} n_neg={result['n_neg']}")
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

        write_report(args.probe_dir, block_list, classes, args.report_path)


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
