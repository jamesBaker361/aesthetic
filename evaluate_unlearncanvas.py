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
#   the same prompts as a control.
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
# Injection (stage 6) adds a concept's whole latent set (one latent, or the --auto_k set) inside the
# base subject's SAM mask: each latent at --inject_value x strength. Images in {out_dir}/inject/...,
# panels (base | each block x rule x strength) in {out_dir}/panels_inject/{concept}_{tag}.jpg.
# --inject_top_k 1 3 5 (with --auto_k) also injects the first k latents of the ranking the auto-k
# search ran over, for each k: images in {out_dir}/inject/{method}/{rule}_top{k}/..., rows with
# k_metric "top{k}" next to the searched set's (k_metric = --auto_k_metric). uc_inject_compare_k.ipynb
# compares them.
#
# Edited images: {out_dir}/answers/{method}/{rule}_{metric}/{concept}/{block}/
# {style}_{object}_seed{seed}.jpg - rule = the variant's kind (bce, f1,
# bce+f1, lasso, random0), metric = --auto_k_metric with --auto_k, else
# "single"; a gamma sweep adds a g{gamma}/ level. Each folder's latents.json
# records the latents and scale it was made with; if a rerun picks different
# ones, that folder's images are deleted and regenerated. Injection images go
# to {out_dir}/inject/ the same way.
#
# --rules lasso: instead of ranking per-latent probes, fit ONE L1-penalised
# logistic regression over every latent jointly (sparse_probe.lasso_select;
# latents scaled to max 1, liblinear); without --auto_k the latent with the
# largest weight at the strongest penalty that keeps one is used (kind
# "lasso"). Per-run files get a _lasso tag.
#
# FID (stage 5, --disable_fid to skip): pytorch-fid features of every scored image are cached next to
# it ({image}.fid.npy); per edit, fid / fid_target / fid_retain compare its images with the unedited
# images of the same prompts (all / with the concept / other objects - the last needs --eval_scope all).
#
# --joint_blocks: instead of choosing latents block by block (and then needing a per-concept block
# choice), every rule works on one feature space - all --block_list blocks side by side (at 512 px the
# four default blocks share a 16x16 patch grid, so each patch has a code at every block; joint latent id
# = block offset + latent). bce / f1 rank all of them, lasso / auto-k fit across them, saeuron scores
# them together, attribution merges the per-block rankings. The chosen set is edited at every block it
# spans in the same pass (one hook per block); auto-gamma pools its probe check over those blocks.
# Tables show block "joint"; per-run files get a _joint tag.
#
# --rules saeuron: SAeUron's own feature selection as a baseline (arXiv:2501.18052,
# SAE/unlearning_utils.compute_feature_importance): each discovery image's SAE code is averaged over
# its patches, and a latent's score is its share of the total mean activation on the concept's images
# minus its share on all other concepts' images. The top tau_c latents are kept (Table 5, SAEURON_TAU;
# --saeuron_tau to override; with --auto_k the score order feeds the smallest-k search instead).
# The full SAeUron recipe adds --remove_scale_preset saeuron --remove_mode saeuron --saeuron_mask:
# their per-object multiplier, scaled by the latent's mean concept activation, applied only where the
# latent beats its mean over all concepts. (Their tau / gamma were tuned for their SD-1.5 SAE.)
#
# --rules attribution: rank latents by their effect on the UnlearnCanvas
# classifier instead of by a mask. For every concept x block, the concept's
# discovery prompts are regenerated with the block's SAE code spliced in as a
# differentiable leaf (forward pass unchanged), and gradient x activation of
# log p(concept) under the object (or style) classifier is summed over patches
# and averaged over images (run_attribution -> {out_dir}/attribution/). The
# top latent is used (kind "attribution"); with --auto_k the attribution order
# feeds the same smallest-k search. Needs the classifier checkpoints; SDXL, the
# SAE and the classifier share the GPU for that step only.
#
# --auto_k: instead of a fixed k, per concept x method x block x rule the
# latents are ordered by importance (the rule: per-latent BCE, per-latent F1,
# or lasso weight) and a binary search finds the fewest leading ones whose
# joint logistic regression classifies the mask patches at least
# --auto_k_frac (0.95) as well as one on every latent (sparse_probe.smallest_k;
# --auto_k_metric on held-out discovery images, ~20% of them). That set is
# zeroed; n_latents says how many, probe_score_all_latents / _k_latents the
# two scores, and the log prints every k the search tried. The k_metric column
# is --auto_k_metric ("single" without --auto_k); per-run files get an
# _auto{frac} tag; no random controls or injection.
#
# Removal, as in SAeUron (arXiv:2501.18052, utils/hooks.py): each chosen
# latent j is set to activation x (gamma x m_j), where m_j is j's mean
# activation over every patch of the concept's own discovery images (zeros
# included) and gamma is --remove_scale (the remove_scale column). gamma 0
# (default) zeroes the latents; negative gamma pushes them the other way, in
# proportion to how strongly they fire on the concept. Several gammas are a
# sweep. Unlike SAeUron it applies at every patch where the latent is active
# (no mask against its mean over other concepts).
#
# --remove_mode direct: instead, each chosen latent becomes activation x gamma
# (the same for every concept). Both modes
# zero the latents at gamma 0. The remove_mode column says which was used.
#
# --auto_gamma: gamma per edit instead of --remove_scale. A dense probe
# (StandardScaler + L2 logistic regression) is fit on the block's un-encoded
# activations of ~80% of the concept's discovery images (labels = its mask);
# on the held-out ~20% the edit is applied offline exactly as the hook does,
# and gamma in [--auto_gamma_min, 0] is bisected for the weakest one after
# which at most --auto_gamma_target (5%) of the held-out mask patches the probe
# called positive still are. No images are generated for the search. Random
# controls get their concept's mean gamma at that block. remove_scale is NaN
# for these rows; gamma holds the value used, dense_* the probe numbers
# (recall / false positives before and after, relative change on / off the
# mask), and the log prints each search curve. Cached in
# {out_dir}/auto_gamma/{concept}__{method}.json. It only checks this block:
# later blocks and the prompt can restore the concept, so confirm with UA.
#
# --remove_scale_preset saeuron: gamma per concept from SAeUron's Table 5
# (App. G, p. 18; SAEURON_MULTIPLIERS): objects -5 to -30, styles -1.
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
# The probe results (features/), the edited / injected images and the result
# tables live under --out_dir. Everything else - discovery images,
# activations, attention/SAM3/Grad-ECLIP maps, base and unedited answer
# images, random control latents, and the classifier/VQA/CLIP scores of
# those - lives under --cache_dir (default: --out_dir), so runs with
# different methods/rules can share it. Files are
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
#   epr_target_change       classifier logits         target images vs unedited        |change| of the concept's logit
#   epr_nontarget_change    classifier logits         target images vs unedited        mean |change| of every other logit of
#                                                                                      both classifiers (side effects)
#   EPR (summary only)      CASL (arXiv:2601.15441,   ratio of the two means above     Editing Precision Ratio: target change
#                           Eq. 14)                   (+1e-8)                          per unit of collateral change
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
#   epr_target_change,                    classifier logits      CASL's EPR terms vs the unedited base image, as in
#     epr_nontarget_change, EPR (summary)                        the removal tables
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
import hashlib
import json

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
from sparse_probe import select_bce_and_f1, smallest_k, latent_activation_stats, image_mean_codes
from evaluate_sae_features import (
    Models, safe, load_json, save_json, generate, ensure_sam_masks, load_sam, sam_cache_path,
    ensure_text_scores, score_cache_path, load_text_score, run_dream_sparsify, load_block_codes,
    make_zero_hook, run_remove_generate, psnr, write_outputs_results,
    run_base, run_ablate_generate, save_image, save_npz, fill_prompt, read_lines,
)
from evaluate_nsfw import fid_features, fid_cache_path, frechet_distance  # pytorch-fid, features cached per image


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
parser.add_argument("--rules", nargs="*", default=["bce", "f1"], choices=["bce", "f1", "lasso", "attribution", "saeuron"],
                    help="which probe rule(s) pick the latent(s) that get zeroed/injected: lowest per-latent "
                         "BCE, highest per-latent F1, or 'lasso' - the latents a joint L1 logistic "
                         "regression over every latent keeps (sparse_probe.lasso_select)")
parser.add_argument("--joint_blocks", action="store_true",
                    help="choose latents from all --block_list blocks at once (one feature space of block x latent, "
                         "patches aligned - every block must share the patch grid): every rule ranks / searches "
                         "across blocks, so no per-block choice is needed; the chosen set is edited at every "
                         "block it spans together (block 'joint' in the tables)")
parser.add_argument("--saeuron_tau", type=int, default=0,
                    help="--rules saeuron: latents kept per concept; 0 = SAeUron's Table 5 value per object "
                         "(SAEURON_TAU; styles 1). Ignored with --auto_k")
parser.add_argument("--saeuron_mask", action="store_true",
                    help="SAeUron's patch mask for removal: only edit a latent where it is above its mean activation "
                         "over all concepts' discovery images (works with any rule)")
parser.add_argument("--attribution_keep", type=int, default=512,
                    help="--rules attribution: how many top-attribution latents per concept x block to store")
parser.add_argument("--attribution_images", type=int, default=0,
                    help="--rules attribution: discovery images per concept to average over (0 = all)")
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
parser.add_argument("--auto_k", action="store_true",
                    help="per concept x method x block x rule, the fewest most-important latents whose joint "
                         "classifier reaches --auto_k_frac of the all-latent one (binary search; default: 1 latent)")
parser.add_argument("--auto_k_frac", type=float, default=0.95)
parser.add_argument("--auto_k_max", type=int, default=64, help="most latents the search may keep")
parser.add_argument("--auto_k_metric", type=str, default="accuracy",
                    choices=["accuracy", "balanced_accuracy", "f1"],
                    help="held-out patch classification score the search compares")
parser.add_argument("--remove_scale", nargs="*", type=float, default=[0.0],
                    help="gamma: removal sets each chosen latent to activation x (gamma x its mean activation on "
                         "the concept), as SAeUron does. 0 = zero it (default); negative = push it the other way. "
                         "Several values = a sweep, each scored separately (the remove_scale column)")
parser.add_argument("--remove_mode", type=str, default="saeuron", choices=["saeuron", "direct"],
                    help="how gamma (--remove_scale) is applied to each chosen latent: 'saeuron' = activation x "
                         "(gamma x the latent's mean activation on the concept), as SAeUron; 'direct' = "
                         "activation x gamma. Identical for gamma 0")
parser.add_argument("--auto_gamma", action="store_true",
                    help="per edit, binary-search the weakest gamma after which at most --auto_gamma_target of "
                         "the held-out mask patches a dense probe called positive still are (overrides "
                         "--remove_scale / --remove_scale_preset)")
parser.add_argument("--auto_gamma_target", type=float, default=0.05)
parser.add_argument("--auto_gamma_min", type=float, default=-50.0, help="strongest gamma the search may use")
parser.add_argument("--auto_gamma_steps", type=int, default=12, help="bisection steps")
parser.add_argument("--remove_scale_preset", type=str, default="none", choices=["none", "saeuron"],
                    help="saeuron: each concept's gamma is its multiplier from SAeUron's Table 5 "
                         "(arXiv:2501.18052, App. G, p.18; SAEURON_MULTIPLIERS) - objects -5 to -30, styles -1 - "
                         "instead of --remove_scale")
parser.add_argument("--seed", type=int, default=0, help="picks the random control latents")
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--end_step", type=int, default=1000)

parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--size", type=int, default=512)

parser.add_argument("--style_ckpt", type=str, default="UnlearnCanvas/ckpts/cls_model/style50-001.pth")
parser.add_argument("--class_ckpt", type=str, default="UnlearnCanvas/ckpts/cls_model/style50_cls.pth")
parser.add_argument("--tag", type=str, default="",
                    help="suffix for this run's tables in out_dir (e.g. 'sdxlcls' when re-scoring with other "
                         "classifiers), so they don't overwrite an earlier run's")
parser.add_argument("--vqa_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
parser.add_argument("--object_text", type=str, default="a photo of a {}", help="VQAScore/CLIPScore text for objects")
parser.add_argument("--style_text", type=str, default="an image in {} style", help="VQAScore/CLIPScore text for styles")
parser.add_argument("--score_batch_size", type=int, default=16)
parser.add_argument("--fid_batch_size", type=int, default=64)

parser.add_argument("--base_prompt_file", type=str, default="prompt_dir/base_prompts.txt")
parser.add_argument("--base_subject_file", type=str, default="prompt_dir/base_subjects.txt")
parser.add_argument("--placeholder", type=str, default="<sks>")
parser.add_argument("--panel_rows", type=int, default=6, help="target prompts (rows) per concept panel")
parser.add_argument("--panel_size", type=int, default=160, help="side of each panel cell in pixels")
parser.add_argument("--strength_list", nargs="*", type=float, default=[10.0])
parser.add_argument("--inject_value", type=str, default="checkpoint_mean",
                    choices=["checkpoint_mean", "pos_mean", "concept_mean"],
                    help="activation placed on each injected latent before * strength: the SAE checkpoint's "
                         "mean.pt, its mean over the concept's positive (mask) discovery patches, or its mean over "
                         "every patch of the concept's discovery images (SAeUron's avg_acts)")
parser.add_argument("--inject_panel_bases", type=int, default=6, help="base images (rows) per injection panel")
parser.add_argument("--inject_top_k", nargs="*", type=int, default=[],
                    help="with --auto_k, also inject the first k latents of each auto-k ranking for every k "
                         "listed (e.g. 1 3 5), to compare a fixed k with the searched one; injection only")

for flag in ["discover_generate", "sparsify", "masks", "attribution", "probe", "answers_generate", "answer_masks",
             "uc", "vqa", "clip", "psnr", "fid", "summary", "panels",
             "inject", "base", "inject_generate", "inject_masks", "inject_summary", "inject_panels"]:
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
        sl, cl = self.style(x).float(), self.cls(x).float()
        sp, cp = F.softmax(sl, dim=-1).cpu().numpy(), F.softmax(cl, dim=-1).cpu().numpy()
        return [{"score": THEMES[int(s.argmax())], "style_pred": THEMES[int(s.argmax())],
                 "class_pred": CLASSES[int(c.argmax())],
                 "style_probs": np.round(s, 5).tolist(), "class_probs": np.round(c, 5).tolist(),
                 "style_logits": np.round(slg, 4).tolist(), "class_logits": np.round(clg, 4).tolist()}
                for s, c, slg, clg in zip(sp, cp, sl.cpu().numpy(), cl.cpu().numpy())]


def uc_model_id(args) -> str:
    return f"{args.style_ckpt}|{args.class_ckpt}"


def uc_score_id(args) -> str:
    # the classifier score cache's id: "+logits" so scores cached before the logits were stored (EPR needs
    # them) are recomputed, without touching attribution_key (also built on uc_model_id)
    return f"{uc_model_id(args)}+logits"


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


# ---------------------------------------------------------------- attribution

def attribution_path(args, concept: str) -> str:
    return os.path.join(args.out_dir, "attribution", f"{safe(concept)}.json")


def attribution_key(args) -> str:
    '''Attribution results are cached per these settings.'''
    return "|".join([uc_model_id(args), args.mode, str(args.num_inference_steps), f"{args.guidance_scale:g}",
                     str(args.size), str(args.attribution_images), ",".join(map(str, args.discover_seeds)),
                     str(args.object_discover_prompt_file)])


def load_attribution(args, concept: str) -> dict:
    '''{block: {"order", "scores", "n_images"}} for the current settings (empty if not computed).'''
    return load_json(attribution_path(args, concept), {}).get(attribution_key(args), {})


def uc_head(ckpt: str, n: int, device):
    '''One UnlearnCanvas ViT-L/16 classifier (as UCClassifiers), frozen, for backprop.'''
    import timm
    m = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=False)
    m.head = torch.nn.Linear(1024, n)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    return m.to(device).eval().requires_grad_(False)


def run_attribution(args, models: UCModels, entries: list, targets: list, block_list: list):
    '''
    Gradient x activation of every SAE latent on the UnlearnCanvas classifier's
    log-probability of the concept. For each concept's discovery prompts (same
    prompt + seed as discovery), the block's SAE code a is spliced in as a leaf:
    output + decoder(a) - decoder(a).detach(), so the forward pass is unchanged
    but d/da flows through the rest of the UNet, the VAE decoder and the
    classifier (224px, as UCClassifiers). a * da summed over patches and
    averaged over images is a first-order estimate of how much log p(concept)
    drops if that latent is zeroed; latents are ranked by it (positive first)
    -> {out_dir}/attribution/{concept}.json. SDXL, the SAE and the classifier
    share the GPU for this step only (gradient checkpointing on the UNet + VAE).
    '''
    todo = [(t, c) for t, c in targets
            if not all(b in load_attribution(args, c) for b in block_list)]
    print(f"attribution: {len(todo)} of {len(targets)} concepts to compute")
    if not todo:
        return
    if not (os.path.exists(args.style_ckpt) and os.path.exists(args.class_ckpt)):
        raise FileNotFoundError(f"--rules attribution needs the UnlearnCanvas classifiers: {args.class_ckpt}")
    device = models.device
    pipe = models.get_pipe()
    sd = pipe.pipe
    call = getattr(type(sd).__call__, "__wrapped__", type(sd).__call__)  # the pipeline call without its no_grad
    unet, vae = sd.unet, sd.vae
    unet.requires_grad_(False); vae.requires_grad_(False)
    unet.enable_gradient_checkpointing(); vae.enable_gradient_checkpointing()
    unet.train(); vae.train()  # some diffusers versions only checkpoint in train mode (no dropout in either)
    upcast = vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
    if upcast:
        vae.to(torch.float32)
    heads = {}
    try:
        for ctype, concept in todo:
            if ctype not in heads:
                heads[ctype] = (uc_head(args.class_ckpt, len(CLASSES), device) if ctype == "object"
                                else uc_head(args.style_ckpt, len(THEMES), device))
            head = heads[ctype]
            label = (CLASSES if ctype == "object" else THEMES).index(concept)
            own = concept_entries(args, entries, ctype, concept)
            if args.attribution_images > 0:
                own = own[:args.attribution_images]
            result = load_json(attribution_path(args, concept), {})
            mine = result.setdefault(attribution_key(args), {})
            for block in block_list:
                if block in mine:
                    continue
                sae = models.get_sae(block).requires_grad_(False)
                store = {}

                def hook(module, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    x = o - inp[0] if args.mode == "diff" else o
                    a = sae.encode(x.permute(0, 2, 3, 1).float()).detach().requires_grad_(True)
                    store["a"] = a
                    d = sae.decoder(a)
                    o = o + (d - d.detach()).permute(0, 3, 1, 2).to(o.dtype)  # same value, gradient to a
                    return (o, *out[1:]) if isinstance(out, tuple) else o

                handle = unet.get_submodule(block).register_forward_hook(hook)
                total, logps = None, []
                try:
                    for n in own:
                        e = entries[n]
                        with torch.enable_grad():
                            lat = call(sd, prompt=e["prompt"], num_inference_steps=args.num_inference_steps,
                                       guidance_scale=args.guidance_scale, height=args.size, width=args.size,
                                       generator=torch.Generator().manual_seed(e["seed"]), output_type="latent").images
                            lat = lat.to(vae.dtype) / vae.config.scaling_factor
                            img = (vae.decode(lat, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)
                            img = F.interpolate(img.float(), size=(224, 224), mode="bilinear", antialias=True)
                            logp = F.log_softmax(head((img - 0.5) / 0.5).float(), dim=-1)[0, label]
                            logp.backward()
                        a = store.pop("a")
                        attr = (a.detach() * a.grad).sum(dim=tuple(range(a.dim() - 1))).double().cpu()
                        total = attr if total is None else total + attr
                        logps.append(float(logp.detach()))
                        del a, lat, img, logp
                finally:
                    handle.remove()
                scores = (total / len(own)).numpy()
                order = [int(j) for j in np.argsort(-scores) if scores[j] > 0][:args.attribution_keep]
                mine[block] = {"order": order, "scores": [float(scores[j]) for j in order],
                               "n_images": len(own), "mean_logp": float(np.mean(logps))}
                print(f"  attribution {ctype} '{concept}' @ {block}: log p = {np.mean(logps):.3f}, "
                      f"top latents {order[:8]} ({[round(float(scores[j]), 4) for j in order[:8]]})")
                os.makedirs(os.path.dirname(attribution_path(args, concept)), exist_ok=True)
                save_json(attribution_path(args, concept), result)
    finally:
        for head in heads.values():
            head.to("cpu")
        heads.clear()
        unet.disable_gradient_checkpointing(); vae.disable_gradient_checkpointing()
        unet.eval(); vae.eval()
        if upcast:
            vae.to(torch.float16)
        models.free()


# ---------------------------------------------------------------- stage 3

def run_tag(args) -> str:
    '''Mask methods of this run, so per-method jobs sharing an out_dir write separate tables.'''
    tag = "_".join(sorted(set(args.object_mask_methods) | set(args.style_mask_methods)))
    if "lasso" in args.rules:
        tag += "_lasso"
    if "attribution" in args.rules:
        tag += "_attr"
    if "saeuron" in args.rules:
        tag += "_saeuron"
    if args.saeuron_mask:
        tag += "_mask"
    if args.joint_blocks:
        tag += "_joint"
    mode = "g" if args.remove_mode == "saeuron" else "x"  # g = gamma x mean activation, x = direct
    if args.auto_gamma:
        tag += f"_{mode}auto{args.auto_gamma_target:g}"
    elif args.remove_scale_preset != "none":
        tag += f"_{mode}{args.remove_scale_preset}"
    elif args.remove_scale != [0.0]:
        tag += f"_{mode}" + "_".join(f"{x:g}" for x in args.remove_scale)
    if args.auto_k:
        tag += f"_auto{args.auto_k_frac:g}"
    if args.tag:
        tag += f"_{args.tag}"
    return tag


def k_metric(args) -> str:
    '''
    The metric half of the image folder name / k_metric column: --auto_k_metric with --auto_k; without it,
    "paper" for the SAeUron baseline (its own tau per concept, Table 5) and "single" for one latent per rule.
    '''
    if args.auto_k:
        return args.auto_k_metric
    return "paper" if "saeuron" in args.rules else "single"


# SAeUron (Cywiński & Deja, arXiv:2501.18052) Table 5, Appendix G (p. 18): per-object multiplier
# gamma_c (their number of selected features tau_c is in the comments). Styles: tau_c = 1, gamma_c = -1.
# Applied like SAeUron: latent * (gamma_c * the latent's mean activation on the concept) - see
# attach_latent_means - but at every active patch (SAeUron masks to patches above the all-concept mean).
SAEURON_MULTIPLIERS = {
    "Architectures": -20.0,  # tau 20
    "Bears": -30.0,          # tau 10
    "Birds": -10.0,          # tau 20
    "Butterfly": -15.0,      # tau 3
    "Cats": -15.0,           # tau 1
    "Dogs": -20.0,           # tau 2
    "Fishes": -30.0,         # tau 2
    "Flame": -25.0,          # tau 3
    "Flowers": -20.0,        # tau 20
    "Frogs": -5.0,           # tau 5
    "Horses": -25.0,         # tau 25
    "Human": -20.0,          # tau 25
    "Jellyfish": -15.0,      # tau 25
    "Rabbits": -30.0,        # tau 4
    "Sandwiches": -15.0,     # tau 20
    "Sea": -30.0,            # tau 15
    "Statues": -30.0,        # tau 20
    "Towers": -20.0,         # tau 25
    "Trees": -25.0,          # tau 30
    "Waterfalls": -30.0,     # tau 30
}
SAEURON_STYLE_MULTIPLIER = -1.0
SAEURON_TAU = {  # Table 5: number of selected features tau_c per object; styles use 1
    "Architectures": 20, "Bears": 10, "Birds": 20, "Butterfly": 3, "Cats": 1, "Dogs": 2, "Fishes": 2,
    "Flame": 3, "Flowers": 20, "Frogs": 5, "Horses": 25, "Human": 25, "Jellyfish": 25, "Rabbits": 4,
    "Sandwiches": 20, "Sea": 15, "Statues": 20, "Towers": 25, "Trees": 30, "Waterfalls": 30,
}


def saeuron_tau(args, ctype: str, concept: str) -> int:
    if args.saeuron_tau > 0:
        return args.saeuron_tau
    return SAEURON_TAU.get(concept, 1) if ctype == "object" else 1


JOINT = "joint"  # the pseudo-block of --joint_blocks


def joint_offsets(entries: list, block_list: list) -> list:
    """[(block, offset, n_dirs)]: joint latent id = offset + the block's own latent id."""
    out, offset = [], 0
    with np.load(entries[0]["sparse"]) as d:
        for block in block_list:
            n = int(d[f"{block}__n_dirs"])
            out.append((block, offset, n))
            offset += n
    return out


def load_joint_codes(entries: list, block_list: list):
    """
    load_block_codes for every block side by side: row i is the same patch of the same image in every
    block (all blocks must share the patch grid), its top-k codes concatenated with each block's latent
    ids shifted by its offset -> one (n_patches, k x n_blocks) code over sum(n_dirs) latents.
    """
    idxs, vals, owner, grid = [], [], None, None
    offsets = joint_offsets(entries, block_list)
    for block, offset, n in offsets:
        idx, val, own, g, _ = load_block_codes(entries, block)
        if grid is not None and tuple(g) != tuple(grid):
            raise ValueError(f"--joint_blocks needs one patch grid for every block: {block} is {g}, not {grid}")
        grid, owner = g, own
        idxs.append(idx.astype(np.int64) + offset)
        vals.append(val)
    return np.concatenate(idxs, axis=1), np.concatenate(vals, axis=1), owner, grid, sum(n for _, _, n in offsets)


def variant_parts(args, v: dict) -> dict:
    """{real block: [(position in v["latents"], the block's own latent id)]} - one block, or several for joint."""
    if v["block"] != JOINT:
        return {v["block"]: list(enumerate(v["latents"]))}
    out = {}
    for pos, j in enumerate(v["latents"]):
        for block, offset, n in args.joint_offsets:
            if offset <= j < offset + n:
                out.setdefault(block, []).append((pos, int(j - offset)))
                break
    return out


def split_by_block(args, v: dict, per_latent=None) -> dict:
    """{block: {"latents": [...], "values": ...}}; per_latent is aligned with v["latents"] (or one scalar / None)."""
    out = {}
    for block, items in variant_parts(args, v).items():
        vals = [per_latent[pos] for pos, _ in items] if isinstance(per_latent, (list, tuple)) else per_latent
        out[block] = {"latents": [j for _, j in items], "values": vals}
    return out


def same_kind_concepts(args, entries: list, ctype: str, concept: str) -> dict:
    '''{other concept: its discovery image indices}, from the same discovery source as `concept`.'''
    source = entries[concept_entries(args, entries, ctype, concept)[0]]["source"]
    out = {}
    for n, e in enumerate(entries):
        if e["source"] == source and e[ctype] is not None:
            out.setdefault(e[ctype], []).append(n)
    return out


def saeuron_scores(args, entries: list, means: np.ndarray, ctype: str, concept: str, eps: float = 1e-8):
    '''
    SAeUron's compute_feature_importance on our discovery images: each latent's share of the total mean
    activation on the concept's images minus its share on every other concept's images. Returns the
    latents with a positive score, best first, and their scores.
    '''
    groups = same_kind_concepts(args, entries, ctype, concept)
    mean_x = means[groups[concept]].mean(axis=0)
    others = [n for c, ns in groups.items() if c != concept for n in ns]
    mean_o = means[others].mean(axis=0) if others else np.zeros_like(mean_x)
    scores = mean_x / (mean_x.sum() + eps) - mean_o / (mean_o.sum() + eps)
    order = [int(j) for j in np.argsort(-scores) if scores[j] > 0]
    return order, scores


def remove_scales(args, v: dict) -> list:
    '''The removal scale(s) an edit of this concept is generated with.'''
    if args.remove_scale_preset == "saeuron":
        if v["concept_type"] == "style":
            return [SAEURON_STYLE_MULTIPLIER]
        return [SAEURON_MULTIPLIERS[v["subject"]]]
    return list(args.remove_scale)


def auto_key(args) -> str:
    '''The auto-k results in a features json are keyed by the search settings.'''
    return f"{args.auto_k_metric}_{args.auto_k_frac:g}_max{args.auto_k_max}"


def auto_rules(args) -> list:
    return [r for r in ["bce", "f1", "lasso"] if r in args.rules]


def top_n(args) -> int:
    '''How many leading latents of each auto-k ranking are kept with their stats (for --inject_top_k).'''
    return max(args.inject_top_k, default=0)


def has_top_k(args, a: dict) -> bool:
    '''Whether a stored auto-k search kept enough of its ranking for --inject_top_k.'''
    return "order" in a and len(a.get("order_top", [])) >= min(top_n(args), len(a["order"]))


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

    for block in ([JOINT] if args.joint_blocks else block_list):
        todo = [(t, c, m) for t, c in targets for m in concept_methods(args, t)
                if block not in features[c]["methods"].get(m, {})
                or (block == JOINT and features[c]["methods"][m][block].get("joint_blocks") != list(block_list))
                or (not args.auto_k and "lasso" in args.rules
                    and "1" not in features[c]["methods"][m][block].get("lasso_by_k", {}))
                or (args.auto_k and not all(
                    r in features[c]["methods"][m][block].get("auto_by_key", {}).get(auto_key(args), {})
                    for r in auto_rules(args) + (["attribution"] if "attribution" in args.rules else [])))
                or (not args.auto_k and "attribution" in args.rules
                    and features[c]["methods"][m][block].get("attribution", {}).get("key") != attribution_key(args))
                or (args.auto_k and "saeuron" in args.rules and "saeuron" not in
                    features[c]["methods"][m][block].get("auto_by_key", {}).get(auto_key(args), {}))
                or (not args.auto_k and "saeuron" in args.rules
                    and features[c]["methods"][m][block].get("saeuron", {}).get("tau") != saeuron_tau(args, t, c))
                or (args.auto_k and args.inject_top_k and not all(
                    has_top_k(args, a) for a in features[c]["methods"][m][block].get("auto_by_key", {})
                    .get(auto_key(args), {}).values()))]
        if not todo:
            continue
        if block == JOINT:
            idx_all, val_all, owner, (gh, gw), n_dirs = load_joint_codes(entries, block_list)
        else:
            idx_all, val_all, owner, (gh, gw), n_dirs = load_block_codes(entries, block)
        means = image_mean_codes(idx_all, val_all, owner, len(entries), n_dirs) if "saeuron" in args.rules else None
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
            auto = None
            if args.auto_k:
                auto = {"rules": auto_rules(args), "groups": owner[rows], "frac": args.auto_k_frac,
                        "metric": args.auto_k_metric, "max_k": args.auto_k_max, "seed": args.seed,
                        "top_n": top_n(args)}
            result = select_bce_and_f1(idx_all[rows], val_all[rows], labels, n_dirs,
                                       args.bce_ridge, args.bce_newton_steps,
                                       lasso="lasso" in args.rules and not args.auto_k, auto=auto)
            result["n_images"] = len(own)
            if block == JOINT:
                result["joint_blocks"] = list(block_list)
            # keep lasso picks and auto-k searches with other settings from earlier runs
            old = features[concept]["methods"].get(method, {}).get(block, {})
            result["lasso_by_k"] = {**old.get("lasso_by_k", {}), **result.get("lasso_by_k", {})}
            result["auto_by_key"] = old.get("auto_by_key", {})
            if auto:
                result["auto_by_key"][auto_key(args)] = {
                    **result["auto_by_key"].get(auto_key(args), {}), **result.pop("auto")}
            if "attribution" in args.rules:
                if block == JOINT:  # one ranking over every block: same objective, so the scores compare
                    per = load_attribution(args, concept)
                    merged = sorted(((sc, off + j) for b, off, _ in args.joint_offsets if b in per
                                     for j, sc in zip(per[b]["order"], per[b]["scores"])), reverse=True)
                    attr = ({"order": [j for _, j in merged], "scores": [sc for sc, _ in merged]}
                            if all(b in per for b in block_list) else None)
                else:
                    attr = load_attribution(args, concept).get(block)
                if attr is None:
                    print(f"  ! no attribution for '{concept}' @ {block} - run without --disable_attribution")
                else:
                    def describe(j, score):
                        return {"idx": int(j), "attribution": score,
                                **latent_activation_stats(idx_all[rows], val_all[rows], labels, j)}
                    if args.auto_k:
                        # fewest top-attribution latents whose mask classifier reaches --auto_k_frac of all latents
                        order = attr["order"][:args.auto_k_max]
                        res = smallest_k(idx_all[rows], val_all[rows], labels, owner[rows], n_dirs, order,
                                         frac=args.auto_k_frac, metric=args.auto_k_metric, seed=args.seed)
                        res["top"] = [describe(j, attr["scores"][i]) for i, j in enumerate(res["latents"])]
                        res["order_top"] = [describe(j, attr["scores"][i])
                                            for i, j in enumerate(res["order"][:top_n(args)])]
                        result["auto_by_key"].setdefault(auto_key(args), {})["attribution"] = res
                    elif attr["order"]:
                        result["attribution"] = {"key": attribution_key(args),
                                                 "top": [describe(attr["order"][0], attr["scores"][0])]}
            if "saeuron" in args.rules:
                order, scores = saeuron_scores(args, entries, means, ctype, concept)

                def describe_s(j):
                    return {"idx": int(j), "saeuron_score": float(scores[j]),
                            **latent_activation_stats(idx_all[rows], val_all[rows], labels, j)}
                if args.auto_k:
                    res = smallest_k(idx_all[rows], val_all[rows], labels, owner[rows], n_dirs,
                                     order[:args.auto_k_max], frac=args.auto_k_frac, metric=args.auto_k_metric,
                                     seed=args.seed)
                    res["top"] = [describe_s(j) for j in res["latents"]]
                    res["order_top"] = [describe_s(j) for j in res["order"][:top_n(args)]]
                    result["auto_by_key"].setdefault(auto_key(args), {})["saeuron"] = res
                elif order:
                    tau = saeuron_tau(args, ctype, concept)
                    result["saeuron"] = {"tau": tau, "top": [describe_s(j) for j in order[:tau]]}
                    print(f"    saeuron: tau {tau}, latents {order[:tau]}")
            features[concept]["methods"].setdefault(method, {})[block] = result
            print(f"{ctype} '{concept}' ({method}) @ {block}: bce latent {result['bce']['idx']} "
                  f"(bce={result['bce']['bce']:.4f}, explained={result['bce']['loss_explained']:.3f}) | "
                  f"f1 latent {result['f1']['idx']} (f1={result['f1']['f1']:.3f})"
                  + (f" | lasso latent {[d['idx'] for d in result['lasso_by_k']['1']['top']]}"
                     if "lasso" in args.rules and not args.auto_k else ""))
            if args.auto_k:
                for rule, a in result["auto_by_key"][auto_key(args)].items():
                    print(f"    auto-k {rule}: k={a['k']} ({a['metric']} {a['score']:.3f} vs all latents "
                          f"{a['full']:.3f}, target {a['target']:.3f}) curve {a['curve']}")
        for ctype, concept, method in todo:
            save_json(features_path(args, concept, method), features[concept]["methods"].get(method, {}))
    return features


# ---------------------------------------------------------------- stage 4

def answer_entries(args) -> list:
    base = os.path.join(args.cache_dir, "answers", "base")
    return [{"object": o, "style": s, "seed": seed, "prompt": fill(args, o, s),
             "file": f"{s}_{o}_seed{seed}.jpg", "image": os.path.join(base, f"{s}_{o}_seed{seed}.jpg")}
            for s in args.eval_styles for o in args.eval_objects for seed in args.eval_seeds]


def edit_dir(args, v: dict, root: str = "answers") -> str:
    '''
    {out_dir}/{root}/{method}/{rule}_{metric}/{concept}/{block}: no latent ids in the path. A gamma sweep
    (several --remove_scale values) adds a g{gamma} level so the sweep's images don't share a folder.
    '''
    parts = [args.out_dir, root, v["method"], f"{v['kind']}_{v.get('k_metric', k_metric(args))}", safe(v["subject"]),
             safe(v["block"].replace(".", "_"))]
    if len(args.remove_scale) > 1 and not args.auto_gamma and args.remove_scale_preset == "none":
        parts.append(f"g{v['gamma']:g}")
    return os.path.join(*parts)


def attach_latent_means(args, entries: list, var_list: list, block_list: list):
    '''
    v["latent_means"]: each latent's mean activation over every patch of the
    concept's own discovery images (SAeUron's avg_acts), for the edits with
    gamma != 0 (and v["thresholds"] with --saeuron_mask). Uses the cached
    top-k codes, one block at a time; a joint edit is split by block.
    '''
    need_inject = args.inject_value == "concept_mean" and not args.disable_inject
    if args.remove_mode != "saeuron" and not need_inject and not args.saeuron_mask:
        return  # direct: the scale doesn't depend on the latent's activations
    todo = [v for v in var_list if v["kind"] != "base" and (need_inject or args.saeuron_mask
                                                             or v["gamma"] is None or v["gamma"] != 0.0)]
    for v in todo:
        v["latent_means"] = [0.0] * len(v["latents"])
        if args.saeuron_mask:
            v["thresholds"] = [0.0] * len(v["latents"])
    for block in block_list:
        here = [(v, items) for v in todo for b, items in variant_parts(args, v).items() if b == block]
        if not here:
            continue
        idx_all, val_all, owner, _, n_dirs = load_block_codes(entries, block)
        means = image_mean_codes(idx_all, val_all, owner, len(entries), n_dirs) if args.saeuron_mask else None
        for v, items in here:
            rows = np.isin(owner, concept_entries(args, entries, v["concept_type"], v["subject"]))
            idx, val = idx_all[rows], val_all[rows]
            for pos, j in items:
                v["latent_means"][pos] = float(np.where(idx == j, val, 0.0).sum(axis=1).mean())
            if args.saeuron_mask:
                # SAeUron's all_concept_avg_acts: the mean over concepts of each concept's mean image code
                groups = same_kind_concepts(args, entries, v["concept_type"], v["subject"])
                thr = np.mean([means[ns].mean(axis=0) for ns in groups.values()], axis=0)
                for pos, j in items:
                    v["thresholds"][pos] = float(thr[j])
    for v in todo:
        means = ", ".join(f"{j}:{m:.3g}" for j, m in zip(v["latents"], v["latent_means"]))
        print(f"  {v['subject']} {v['method']} {v['kind']} @ {v['block']} gamma "
              f"{'auto' if v['gamma'] is None else format(v['gamma'], 'g')}: mean activations {means}")


def edit_scale(args, v: dict, gamma: float):
    '''Per-latent scale the hook applies: gamma x mean concept activation (saeuron) or gamma (direct).'''
    if gamma != 0.0 and args.remove_mode == "saeuron":
        return [gamma * m for m in v["latent_means"]]
    return gamma


def auto_gamma_path(args, concept: str, method: str) -> str:
    return os.path.join(args.out_dir, "auto_gamma", f"{safe(concept)}__{method}.json")


def auto_gamma_key(args, v: dict) -> str:
    return "|".join([v["block"], v["kind"], ",".join(str(i) for i in v["latents"]), args.remove_mode,
                     f"{args.auto_gamma_target:g}", f"{args.auto_gamma_min:g}", str(args.auto_gamma_steps),
                     args.mode, str(args.seed)])


@torch.no_grad()
def run_auto_gamma(args, models: UCModels, entries: list, var_list: list, block_list: list):
    '''
    For every learned edit: at each block it touches, fit a dense probe
    (StandardScaler + L2 logistic regression) on the block's un-encoded
    activations (out - in with --mode diff) of ~80% of the concept's discovery
    images, labels = its mask. On the held-out images, apply the edit offline
    exactly as make_zero_hook does (x - decoder(onehot of the chosen latents x
    (1 - scale)), each block with its own latents) and binary-search gamma in
    [--auto_gamma_min, 0] for the weakest one after which at most
    --auto_gamma_target of the held-out mask patches the probes called positive
    before are still positive (pooled over the blocks a joint edit spans).
    Sets v["gamma"] and v["gamma_search"]; cached in
    {out_dir}/auto_gamma/{concept}__{method}.json. Random controls get the mean
    gamma found for their concept at their block.
    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    learned = [v for v in var_list if v["kind"] != "base" and v["method"] != "random"]
    caches = {}
    for v in learned:
        path = auto_gamma_path(args, v["subject"], v["method"])
        caches.setdefault(path, load_json(path, {}))
        hit = caches[path].get(auto_gamma_key(args, v))
        if hit is not None:
            v["gamma"], v["gamma_search"] = hit["gamma"], hit
    todo = [v for v in learned if v["gamma"] is None]
    print(f"auto-gamma: {len(todo)} of {len(learned)} edits to search")
    if todo:
        models.free(keep="sae")

    groups = {}
    for v in todo:
        groups.setdefault((v["concept_type"], v["subject"], v["method"]), []).append(v)
    for (ctype, concept, method), vs in groups.items():
        own = concept_entries(args, entries, ctype, concept)
        rng = np.random.default_rng(args.seed)
        test_ids = set(rng.choice(len(own), max(1, int(round(0.2 * len(own)))), replace=False).tolist())
        data = {}  # block -> probe + held-out activations / codes, built on first use

        def block_data(block):
            if block in data:
                return data[block]
            images = []
            for n in own:
                e = entries[n]
                with np.load(e["embedding"]) as d:
                    out = d[f"saved_output.{block}"][0]
                    x = out - d[f"saved_input.{block}"][0] if args.mode == "diff" else out
                x = x.transpose(1, 2, 0).astype(np.float32)
                labels = patch_labels(args, e, ctype, concept, method, x.shape[0], x.shape[1]).reshape(-1)
                images.append((x, labels.astype(bool)))
            train = [im for i, im in enumerate(images) if i not in test_ids] or images
            test = [im for i, im in enumerate(images) if i in test_ids]
            Xtr = np.concatenate([x.reshape(-1, x.shape[-1]) for x, _ in train])
            ytr = np.concatenate([y for _, y in train])
            if ytr.all() or not ytr.any():
                data[block] = None
                return None
            scaler = StandardScaler().fit(Xtr)
            probe = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(Xtr), ytr)
            sae = models.get_sae(block)
            xt = [torch.tensor(x, device=models.device) for x, _ in test]
            yte = np.concatenate([y for _, y in test])
            flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x, _ in test])
            before = probe.predict(scaler.transform(flat)).astype(bool)
            data[block] = {"sae": sae, "probe": probe, "scaler": scaler, "xt": xt,
                           "codes": [sae.encode(x) for x in xt], "yte": yte, "before": before,
                           "was_pos": yte & before}
            return data[block]

        for v in vs:
            parts = {b: items for b, items in variant_parts(args, v).items() if block_data(b) is not None}
            if not parts:
                print(f"  auto-gamma: '{concept}' ({method}) @ {v['block']}: no mask contrast - gamma 0")
                v["gamma"], v["gamma_search"] = 0.0, {"gamma": 0.0, "reached": False}
                continue

            def measure(gamma):
                scale = edit_scale(args, v, gamma)
                pooled = {"was_pos": 0, "still": 0, "pos": 0, "rec": 0, "neg": 0, "fp": 0}
                ch_on, ch_off = [], []
                for block, items in parts.items():
                    d = data[block]
                    pos_list = [pos for pos, _ in items]
                    latents = [j for _, j in items]
                    sc = scale if not isinstance(scale, list) else [scale[p] for p in pos_list]
                    keep = 1.0 - torch.as_tensor(sc, dtype=torch.float32, device=models.device)
                    edited, change = [], []
                    for x, a in zip(d["xt"], d["codes"]):
                        onehot = torch.zeros_like(a)
                        sel = a[..., latents]
                        if v.get("thresholds") is not None:  # SAeUron's mask, as the hook applies it
                            thr = torch.tensor([v["thresholds"][p] for p in pos_list], dtype=sel.dtype,
                                               device=sel.device)
                            onehot[..., latents] = torch.where(sel > thr, sel * keep, torch.zeros_like(sel))
                        else:
                            onehot[..., latents] = sel * keep
                        delta = d["sae"].decoder(onehot)
                        edited.append((x - delta).reshape(-1, x.shape[-1]).cpu().numpy())
                        change.append((delta.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).reshape(-1).cpu().numpy())
                    after = d["probe"].predict(d["scaler"].transform(np.concatenate(edited))).astype(bool)
                    change = np.concatenate(change)
                    yte, was_pos = d["yte"], d["was_pos"]
                    pooled["was_pos"] += int(was_pos.sum())
                    pooled["still"] += int(after[was_pos].sum())
                    pooled["pos"] += int(yte.sum())
                    pooled["rec"] += int(after[yte].sum())
                    pooled["neg"] += int((~yte).sum())
                    pooled["fp"] += int(after[~yte].sum())
                    ch_on.append(change[yte])
                    ch_off.append(change[~yte])
                ch_on, ch_off = np.concatenate(ch_on), np.concatenate(ch_off)
                return {"still_positive": pooled["still"] / pooled["was_pos"] if pooled["was_pos"] else 0.0,
                        "recall_after": pooled["rec"] / pooled["pos"] if pooled["pos"] else float("nan"),
                        "fp_after": pooled["fp"] / pooled["neg"] if pooled["neg"] else float("nan"),
                        "change_on": float(ch_on.mean()) if len(ch_on) else float("nan"),
                        "change_off": float(ch_off.mean()) if len(ch_off) else float("nan")}

            curve = {}

            def at(g):
                g = float(f"{g:.4g}")  # 4 significant digits: keeps the numbers readable
                if g not in curve:
                    curve[g] = measure(g)
                return g, curve[g]

            target = args.auto_gamma_target
            g0, m0 = at(0.0)
            gmin, mmin = at(args.auto_gamma_min)
            if m0["still_positive"] <= target:
                best, reached = g0, True
            elif mmin["still_positive"] > target:
                best, reached = gmin, False
            else:
                lo, hi = gmin, 0.0  # lo meets the target, hi doesn't
                for _ in range(args.auto_gamma_steps):
                    g, m = at((lo + hi) / 2)
                    if m["still_positive"] <= target:
                        lo = g
                    else:
                        hi = g
                best, reached = lo, True
            yte_all = np.concatenate([data[b]["yte"] for b in parts])
            before_all = np.concatenate([data[b]["before"] for b in parts])
            res = {"gamma": best, "reached": reached,
                   "dense_recall_before": float(before_all[yte_all].mean()) if yte_all.any() else float("nan"),
                   "dense_fp_before": float(before_all[~yte_all].mean()) if (~yte_all).any() else float("nan"),
                   **curve[best],
                   "curve": {f"{g:g}": round(m["still_positive"], 4) for g, m in sorted(curve.items())}}
            v["gamma"], v["gamma_search"] = best, res
            caches[auto_gamma_path(args, concept, method)][auto_gamma_key(args, v)] = res
            print(f"  auto-gamma '{concept}' ({method}) {v['kind']} @ {v['block']} ({', '.join(parts)}) "
                  f"{len(v['latents'])} latents: gamma {best:g}{'' if reached else ' (target NOT reached)'} - still "
                  f"positive {res['still_positive']:.3f} (target {target:g}), recall {res['dense_recall_before']:.3f}"
                  f" -> {res['recall_after']:.3f}, fp {res['dense_fp_before']:.3f} -> {res['fp_after']:.3f}, "
                  f"rel. change on/off mask {res['change_on']:.3f}/{res['change_off']:.3f} | curve {res['curve']}")
        path = auto_gamma_path(args, concept, method)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_json(path, caches[path])
        del data

    # random controls: the mean gamma found for the same concept at the same block
    for v in var_list:
        if v["method"] == "random" and v["gamma"] is None:
            found = [u["gamma"] for u in learned
                     if u["subject"] == v["subject"] and u["block"] == v["block"] and u["gamma"] is not None]
            v["gamma"] = float(f"{np.mean(found):.4g}") if found else 0.0


def resolve_random_latents(args, models: UCModels, block_list: list) -> dict:
    path = os.path.join(args.cache_dir, "random_latents.json")
    chosen = load_json(path, {})
    before = dict(chosen)
    rng = np.random.default_rng(args.seed)
    for block in block_list:
        for r in range(args.n_random_controls):
            key = f"{block}__random{r}"
            if key not in chosen:
                chosen[key] = int(rng.integers(models.get_sae(block).n_dirs))
    if chosen != before:  # only write when something was added - every parallel job reads this file
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
            for block in ([JOINT] if args.joint_blocks else block_list):
                info = per_block.get(block)
                if info is None:
                    continue
                if args.auto_k:
                    found = info.get("auto_by_key", {}).get(auto_key(args), {})
                    found = {r: a for r, a in found.items() if r in args.rules and a["latents"]}
                    same = "bce" in found and "f1" in found and sorted(found["bce"]["latents"]) == sorted(found["f1"]["latents"])
                    rules = (["bce+f1"] if same else []) + [r for r in found if not (same and r in ("bce", "f1"))]
                    for rule in rules:
                        a = found["bce" if rule == "bce+f1" else rule]
                        out.append({**common, "method": method, "block": block, "kind": rule,
                                    "feature_idx": a["latents"][0], "latents": a["latents"], "probe": a["top"][0],
                                    "latent_stats": a["top"],
                                    "auto": {"full": a["full"], "score": a["score"], "k": a["k"]}})
                    continue
                picks = {r: [info[r]["idx"]] for r in ["bce", "f1"]}
                same = picks["bce"] == picks["f1"]
                ranked = [r for r in ["bce", "f1"] if r in args.rules]
                rules = (["bce+f1"] if same else ranked) if ranked else []
                for rule in rules:
                    key = "f1" if rule == "f1" else "bce"
                    chosen = info[key]  # top-1: its probe stats go in the tables
                    out.append({**common, "method": method, "block": block, "kind": rule,
                                "feature_idx": chosen["idx"], "latents": picks[key], "probe": chosen})
                lasso = info.get("lasso_by_k", {}).get("1", {}).get("top", [])
                if "lasso" in args.rules and lasso:
                    out.append({**common, "method": method, "block": block, "kind": "lasso",
                                "feature_idx": lasso[0]["idx"], "latents": [d["idx"] for d in lasso],
                                "probe": lasso[0]})
                sae_top = info.get("saeuron", {}).get("top", [])
                if "saeuron" in args.rules and sae_top:
                    out.append({**common, "method": method, "block": block, "kind": "saeuron",
                                "feature_idx": sae_top[0]["idx"], "latents": [d["idx"] for d in sae_top],
                                "probe": sae_top[0], "latent_stats": sae_top})
                attr = info.get("attribution", {}).get("top", [])
                if "attribution" in args.rules and attr:
                    out.append({**common, "method": method, "block": block, "kind": "attribution",
                                "feature_idx": attr[0]["idx"], "latents": [attr[0]["idx"]], "probe": attr[0]})
        for block in block_list if not args.auto_k else []:  # auto-k: k differs per variant, no random match
            for r in range(args.n_random_controls):
                latents = [random_latents[f"{block}__random{r}"]]
                out.append({**common, "method": "random", "block": block, "kind": f"random{r}",
                            "feature_idx": latents[0], "latents": latents, "probe": {}})
    if args.auto_k and args.n_random_controls:
        print("--auto_k: no random controls (each variant keeps a different number of latents)")
    for v in out:
        v["k_metric"] = k_metric(args)
        v["remove_mode"] = args.remove_mode
    # one copy of every edit per --remove_scale; the unedited model is scale 1. remove_scale is the
    # setting (NaN = --auto_gamma), gamma the value actually used (filled in by run_auto_gamma)
    scaled = []
    for v in out:
        v["auto_gamma_target"] = args.auto_gamma_target if args.auto_gamma else 0.0
        if v["kind"] == "base":
            scaled.append({**v, "remove_scale": 1.0, "gamma": None})
        elif args.auto_gamma:
            scaled.append({**v, "remove_scale": float("nan"), "gamma": None})
        else:
            scaled += [{**v, "remove_scale": scale, "gamma": scale} for scale in remove_scales(args, v)]
    return scaled


def top_k_variants(args, features: dict, targets: list, block_list: list) -> list:
    '''
    --inject_top_k: for every auto-k variant, the first k latents of the same ranking the search ran over
    (per-latent BCE / F1, lasso weight, attribution or SAeUron score), for each k listed. kind is the rule
    (bce+f1 when both rankings give the same set), k_metric "top{k}", so images go to
    {rule}_top{k}/ next to the searched {rule}_{metric}/. Injection only - no removal images.
    '''
    if not (args.auto_k and args.inject_top_k):
        return []
    out = []
    for ctype, concept in targets:
        common = {"subject": concept, "concept_type": ctype}
        for method, per_block in features.get(concept, {}).get("methods", {}).items():
            if method not in concept_methods(args, ctype):
                continue
            for block in block_list:
                found = (per_block.get(block) or {}).get("auto_by_key", {}).get(auto_key(args), {})
                found = {r: a for r, a in found.items() if r in args.rules and a.get("order_top")}
                for k in args.inject_top_k:
                    picks = {r: a["order_top"][:k] for r, a in found.items() if len(a["order_top"]) >= k}
                    same = "bce" in picks and "f1" in picks and \
                        [d["idx"] for d in picks["bce"]] == [d["idx"] for d in picks["f1"]]
                    rules = (["bce+f1"] if same else []) + [r for r in picks if not (same and r in ("bce", "f1"))]
                    for rule in rules:
                        top = picks["bce" if rule == "bce+f1" else rule]
                        out.append({**common, "method": method, "block": block, "kind": rule,
                                    "feature_idx": top[0]["idx"], "latents": [d["idx"] for d in top],
                                    "probe": top[0], "latent_stats": top, "k_metric": f"top{k}",
                                    "remove_mode": args.remove_mode,
                                    "auto_gamma_target": args.auto_gamma_target if args.auto_gamma else 0.0})
    for v in out:  # same scale fields as the first --remove_scale copy of a variant, which inject_jobs keeps
        scale = float("nan") if args.auto_gamma else remove_scales(args, v)[0]
        v.update({"remove_scale": scale, "gamma": None if args.auto_gamma else scale})
    print(f"--inject_top_k {args.inject_top_k}: {len(out)} fixed-k injection variants")
    return out


def variant_image(args, v: dict, a: dict) -> str:
    if v["kind"] == "base":
        return a["image"]
    return os.path.join(edit_dir(args, v), a["file"])


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
            # make_zero_hook scales a list of latents as well as one
            # saeuron: each latent -> activation x (gamma x its mean activation on the concept);
            # direct: activation x gamma
            scale = edit_scale(args, v, v["gamma"])
            thr = split_by_block(args, v, v["thresholds"]) if v.get("thresholds") is not None else None
            parts = {b: {"latents": p["latents"], "scale": p["values"],
                         "thresholds": thr[b]["values"] if thr else None}
                     for b, p in split_by_block(args, v, scale).items()}
            jobs[path] = {"block": v["block"], "parts": parts, "prompt": a["prompt"], "seed": a["seed"],
                          "image": path}
    # every edit folder records the latents + scale its images were made with; a folder made with
    # different ones (the probe or gamma changed) is emptied so its images are regenerated
    specs = {}
    for j in jobs.values():
        specs[os.path.dirname(j["image"])] = {"block": j["block"], "parts": j["parts"]}
    for folder, spec in specs.items():
        manifest = os.path.join(folder, "latents.json")
        if os.path.exists(manifest) and load_json(manifest, None) != json.loads(json.dumps(spec)):
            print(f"  {folder}: latents/scale changed - regenerating its images")
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))
        if not os.path.exists(manifest):
            os.makedirs(folder, exist_ok=True)
            save_json(manifest, spec)
    print(f"answers: {len(jobs)} edited images in {len(specs)} folders (k {k_metric(args)}, gamma="
          f"{'auto' if args.auto_gamma else args.remove_scale if args.remove_scale_preset == 'none' else args.remove_scale_preset})")
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
                               model=uc_score_id(args))
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
    if not args.disable_fid:
        base = sorted({a["image"] for a in answers if os.path.exists(a["image"])})
        fid_features(models, sorted(set(images) | set(base)), args.fid_batch_size)


def add_fid(args, df: pd.DataFrame, keys: list) -> pd.DataFrame:
    '''
    Per edit (one keys group): pytorch-fid FID of its images against the unedited images of the same
    prompts - fid (all), fid_target (images with the concept), fid_retain (the others, --eval_scope all
    only). Written on every row of the group, so means / summaries carry it. FID from a few dozen images
    is biased upward - compare edits with the same image counts, not to published values.
    '''
    def feats(paths):
        paths = [p for p in paths if os.path.exists(fid_cache_path(p))]
        return np.stack([np.load(fid_cache_path(p)) for p in paths]) if paths else np.zeros((0, 2048))

    df = df.copy()
    for col in ["fid", "fid_target", "fid_retain"]:
        df[col] = np.nan
    for _, g in df[df["kind"] != "base"].groupby(keys, dropna=False):
        for col, part in [("fid", g), ("fid_target", g[g["is_target"] == 1]), ("fid_retain", g[g["is_target"] == 0])]:
            if len(part) >= 2:
                df.loc[part.index if col != "fid" else g.index, col] = frechet_distance(
                    feats(part["image"]), feats(part["base_image"]))
    return df


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
                "kind": v["kind"], "feature_idx": v["feature_idx"], "k_metric": v["k_metric"],
                "remove_mode": v["remove_mode"], "remove_scale": v["remove_scale"],
                "auto_gamma_target": v["auto_gamma_target"], "gamma": v["gamma"],
                **{f"dense_{k}": v.get("gamma_search", {}).get(k) for k in
                   ["recall_before", "recall_after", "fp_before", "fp_after", "still_positive",
                    "change_on", "change_off"]},
                "latents": ";".join(str(i) for i in v["latents"]),
                "object": a["object"], "style": a["style"], "seed": a["seed"], "image": path, "base_image": a["image"],
                "is_target": float(target),
                "probe_bce": v["probe"].get("bce"), "probe_loss_explained": v["probe"].get("loss_explained"),
                "probe_f1": v["probe"].get("f1"),
                "n_latents": len(v["latents"]),
                "probe_score_all_latents": v.get("auto", {}).get("full"),
                "probe_score_k_latents": v.get("auto", {}).get("score"),
            }
            uc = load_uc(path) if not args.disable_uc else None
            if uc is not None and uc.get("model") == uc_score_id(args):
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
            if target and v["kind"] != "base":
                d_t, d_n = uc_logit_changes(args, path, a["image"], ctype, concept)
                if d_t is not None:
                    row.update({"epr_target_change": d_t, "epr_nontarget_change": d_n})
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
    fid_keys = ["subject", "concept_type", "method", "block", "kind", "k_metric", "remove_mode", "remove_scale",
                "auto_gamma_target"]
    if not args.disable_fid:
        df = add_fid(args, df, fid_keys)
    df.to_csv(os.path.join(args.out_dir, f"uc_results_{run_tag(args)}.csv.gz"), index=False)

    keys = ["subject", "concept_type", "method", "block", "kind", "k_metric", "remove_mode", "remove_scale",
            "auto_gamma_target"]
    metrics = ["UA", "IRA", "CRA", "CRA_target", "p_target", "style_acc", "object_acc", "sam_removed", "sam_score", "sam_area",
               "vqa", "clip", "psnr_target", "psnr_retain", "epr_target_change", "epr_nontarget_change",
               "probe_bce", "probe_loss_explained", "probe_f1",
               "n_latents", "probe_score_all_latents", "probe_score_k_latents", "gamma", "fid", "fid_target", "fid_retain",
               "dense_recall_before", "dense_recall_after", "dense_fp_before", "dense_fp_after",
               "dense_still_positive", "dense_change_on", "dense_change_off"]
    metrics = [m for m in metrics if m in df]
    for m in metrics:
        df[m] = pd.to_numeric(df[m].replace([np.inf], np.nan), errors="coerce")
    df["block"] = df["block"].fillna("none")
    grouped = df.groupby(keys, dropna=False)  # remove_scale is NaN for --auto_gamma
    summary = grouped[metrics].mean()
    summary.insert(0, "n_images", grouped.size())
    summary.insert(0, "latents", grouped["latents"].first())
    summary.insert(0, "feature_idx", grouped["feature_idx"].first())
    summary = add_epr(summary.reset_index())

    # the unedited model's numbers next to every row of the same concept
    base_cols = [m for m in ["UA", "IRA", "CRA", "CRA_target", "p_target", "sam_removed", "vqa", "clip"] if m in summary]
    base = summary[summary["kind"] == "base"].set_index("subject")[base_cols].add_suffix("_base")
    summary = summary.join(base, on="subject")
    summary.to_csv(os.path.join(args.out_dir, f"uc_summary_{run_tag(args)}.csv"), index=False)

    shown = [m for m in ["UA", "IRA", "CRA", "CRA_target", "sam_removed", "vqa", "psnr_target", "psnr_retain", "EPR"]
             if m in summary and summary[m].notna().any()]
    print(summary.groupby(["concept_type", "method", "kind"])[shown].mean().to_string())

    write_outputs_results(args, df.drop(columns=["seed", "latents", "base_image"]), filename="uc_results.csv", keys=keys,
                          replace_on=["subject", "method", "k_metric", "remove_mode", "remove_scale",
                                      "auto_gamma_target"],
                          defaults={"k_metric": "single", "remove_mode": "direct", "auto_gamma_target": 0.0,
                                    # rows from before --remove_scale: edits zeroed (0), the unedited model is 1
                                    "remove_scale": lambda old: np.where(old["kind"] == "base", 1.0, 0.0)})
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
        vs = sorted(vs, key=lambda v: (order.get(v["method"], 1), v["method"], v["block"], v["kind"],
                                       v["gamma"] if v["gamma"] is not None else 1.0))
        rows = [a for a in answers if a[ctype] == concept][:args.panel_rows]
        if not rows:
            continue
        grid = Image.new("RGB", (size * len(vs), head + (size + cap) * len(rows)), "white")
        draw = ImageDraw.Draw(grid)
        for c, v in enumerate(vs):
            label = "unedited" if v["kind"] == "base" else \
                f"{v['method']} {v['kind']}\n{v['block'].replace('_blocks', '').replace('.attentions', '')} " \
                f"#{','.join(str(i) for i in v['latents'])[:18]} " \
                f"{'g' if args.remove_mode == 'saeuron' else 'x'}{v['gamma']:g}"
            draw.text((c * size + 3, 2), label, fill="black")
            for r, a in enumerate(rows):
                x, y = c * size, head + r * (size + cap)
                path = variant_image(args, v, a)
                if not os.path.exists(path):
                    draw.rectangle([x, y, x + size - 1, y + size - 1], fill="lightgray")
                    continue
                grid.paste(Image.open(path).convert("RGB").resize((size, size)), (x, y))
                uc = load_uc(path) if not args.disable_uc else None
                if uc is not None and uc.get("model") == uc_score_id(args):
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
    One job per (variant, strength, masked base image), in
    {out_dir}/inject/{method}/{rule}_{metric}/{concept}/{block}/s{strength}/.
    '''
    usable = [e for e in base_entries if e.get("mask_area", 0) > 0]
    jobs = []
    for v in var_list:
        if v["kind"] == "base" or (not args.auto_gamma and v["remove_scale"] != remove_scales(args, v)[0]):
            continue  # injection doesn't depend on the removal scale: one copy per latent
        pos_mean = v["probe"].get("pos_mean")
        latents = list(v["latents"])
        # each latent's value (x strength): None = the checkpoint mean (filled in by run_ablate_generate)
        if args.inject_value == "pos_mean":
            stats = {d["idx"]: d for d in v.get("latent_stats", [v["probe"]])}
            values = [stats.get(j, {}).get("pos_mean") for j in latents]
        elif args.inject_value == "concept_mean":
            values = v.get("latent_means")
        else:
            values = None
        for strength in args.strength_list:
            for e in usable:
                jobs.append({
                    **{k: v[k] for k in ["subject", "concept_type", "method", "block", "kind", "feature_idx",
                                         "k_metric"]},
                    "latents": latents, "values": values, "n_latents": len(latents),
                    "parts": split_by_block(args, v, values),
                    "probe": v["probe"], "pos_mean": pos_mean, "strength": strength, "base": e["name"],
                    "image": os.path.join(edit_dir(args, v, root="inject"), f"s{strength:g}", f"{e['name']}.jpg"),
                })
    return jobs


def save_inject_panels(args, jobs: list, base_by_name: dict):
    '''
    {out_dir}/panels_inject/{concept}_{tag}.jpg: rows = the first --inject_panel_bases base images,
    columns = the unedited base, then every block x rule x strength with the concept's latent set added
    inside the base subject's mask. Only uses images already generated.
    '''
    d = os.path.join(args.out_dir, "panels_inject")
    os.makedirs(d, exist_ok=True)
    size, head = args.panel_size, 34
    by_concept = {}
    for j in jobs:
        by_concept.setdefault(j["subject"], []).append(j)
    for concept, js in by_concept.items():
        bases = sorted({j["base"] for j in js})[:args.inject_panel_bases]
        cols = sorted({(j["block"], j["kind"], j["k_metric"], j["strength"], j["n_latents"]) for j in js})
        image_of = {(j["block"], j["kind"], j["k_metric"], j["strength"], j["base"]): j["image"] for j in js}
        grid = Image.new("RGB", (size * (len(cols) + 1), head + size * len(bases)), "white")
        draw = ImageDraw.Draw(grid)
        draw.text((3, 2), "unedited", fill="black")
        for c, (block, kind, metric, strength, k) in enumerate(cols, start=1):
            short = block.replace("_blocks", "").replace(".attentions", "")
            draw.text((c * size + 3, 2), f"{kind} {short} {metric}\nk={k} x{strength:g}", fill="black")
        for r, b in enumerate(bases):
            y = head + r * size
            paths = [base_by_name[b]["image"]] + [image_of.get((bl, kd, km, st, b)) for bl, kd, km, st, _ in cols]
            for c, path in enumerate(paths):
                if path and os.path.exists(path):
                    grid.paste(Image.open(path).convert("RGB").resize((size, size)), (c * size, y))
                else:
                    draw.rectangle([c * size, y, c * size + size - 1, y + size - 1], fill="lightgray")
        save_image(grid, os.path.join(d, f"{safe(concept)}_{run_tag(args)}.jpg"))
    print(f"injection panels -> {d}")


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
                           model=uc_score_id(args))

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
    if uc is None or uc.get("model") != uc_score_id(args):
        return None, None
    pred = uc["style_pred"] if ctype == "style" else uc["class_pred"]
    probs = uc["style_probs"] if ctype == "style" else uc["class_probs"]
    labels = THEMES if ctype == "style" else CLASSES
    return float(pred == concept), float(probs[labels.index(concept)])


def uc_logit_changes(args, path: str, original: str, ctype: str, concept: str):
    '''
    CASL's EPR terms (arXiv:2601.15441, Eqs. 12-13) for one edited / original image pair, on the
    UnlearnCanvas classifiers' logits: |change| of the concept's own logit, and the mean |change| of every
    other logit of both classifiers (the other styles incl. Seed_Images and the other objects).
    (None, None) if either image has no logits cached.
    '''
    if args.disable_uc:
        return None, None
    a, b = load_uc(path), load_uc(original)
    if any(u is None or u.get("model") != uc_score_id(args) for u in (a, b)):
        return None, None
    own, other = ("style_logits", "class_logits") if ctype == "style" else ("class_logits", "style_logits")
    i = (THEMES if ctype == "style" else CLASSES).index(concept)
    d_own = np.abs(np.asarray(a[own]) - np.asarray(b[own]))
    d_other = np.abs(np.asarray(a[other]) - np.asarray(b[other]))
    return float(d_own[i]), float(np.concatenate([np.delete(d_own, i), d_other]).mean())


def add_epr(summary: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    '''EPR (CASL Eq. 14) per summary row: mean target-logit change / (mean non-target-logit change + eps).'''
    if "epr_target_change" in summary and "epr_nontarget_change" in summary:
        summary["EPR"] = summary["epr_target_change"] / (summary["epr_nontarget_change"] + eps)
    return summary


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
            **{k: j[k] for k in ["subject", "concept_type", "method", "block", "kind", "k_metric", "feature_idx",
                                 "strength", "n_latents"]},
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
        d_t, d_n = uc_logit_changes(args, j["image"], base["image"], ctype, concept)
        if d_t is not None:
            row.update({"epr_target_change": d_t, "epr_nontarget_change": d_n})
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

    keys = ["subject", "concept_type", "method", "block", "kind", "k_metric", "strength"]
    metrics = [c for c in df.columns if c not in keys and c != "feature_idx"
               and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
               and df[c].notna().any() and c not in ["base", "base_subject", "base_prompt", "image"]]
    for m in metrics:
        df[m] = pd.to_numeric(df[m].replace([np.inf], np.nan), errors="coerce")
    grouped = df.groupby(keys)
    summary = grouped[metrics].mean()
    summary.insert(0, "n_images", grouped.size())
    summary.insert(0, "feature_idx", grouped["feature_idx"].first())
    add_epr(summary.reset_index()).to_csv(os.path.join(args.out_dir, f"inject_summary_{run_tag(args)}.csv"), index=False)

    shown = [m for m in ["mask_iou", "base_subject_remaining", "vqa_subject_gain", "uc_classified_as",
                         "uc_p_target_gain", "background_psnr"] if m in df]
    print(df.groupby(["concept_type", "method", "kind", "k_metric", "strength"])[shown].mean().to_string())
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
    args.joint_offsets = None
    if not args.disable_discover_generate:
        run_discover_generate(args, models, entries, block_list)
    if not args.disable_sparsify:
        run_dream_sparsify(args, models, entries, block_list)
    if not args.disable_masks:
        run_masks(args, models, entries, targets, device)
    if args.joint_blocks and not args.prepare_only:
        args.joint_offsets = joint_offsets(entries, block_list)
    if "attribution" in args.rules and not args.prepare_only and not args.disable_attribution:
        run_attribution(args, models, entries, targets, block_list)
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
    attach_latent_means(args, entries, var_list, block_list)
    if args.auto_gamma:  # (prepare-only: no learned edits, just gives the random controls gamma 0)
        run_auto_gamma(args, models, entries, var_list, block_list)
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
    if not args.disable_inject:
        # run_base writes {out_dir}/base - point it at the shared cache
        cache_args = argparse.Namespace(**{**vars(args), "out_dir": args.cache_dir})
        if args.disable_base:
            base_entries = load_json(os.path.join(args.cache_dir, "base", "manifest.json"), [])
        else:
            base_entries = run_base(cache_args, models, device)
        base_by_name = {e["name"]: e for e in base_entries}
        top_k = top_k_variants(args, features, targets, block_list)
        if args.inject_value == "concept_mean":
            attach_latent_means(args, entries, top_k, block_list)
        i_jobs = inject_jobs(args, base_entries, var_list + top_k)
        if not args.disable_inject_generate:
            unique = list({j["image"]: j for j in i_jobs}.values())
            run_ablate_generate(args, models, unique, base_by_name, device)
        if not args.disable_inject_panels:
            save_inject_panels(args, i_jobs, base_by_name)
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
