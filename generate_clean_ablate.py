# given a single base prompt and a mask_target, generates one base image and
# a SAM3 mask for mask_target on it, then for every {query, block} feature
# saved in --npz_dict (by generate_clean_inference.py), regenerates the same
# image (same seed/prompt) injecting that feature ONLY inside the masked
# region - background stays anchored to the base image, same technique as
# generate_clean_swap.py's patch_mask-restricted editing.
#
# Each feature gets up to 3 variants: injected at its mean activation, mean
# plus one std, and mean minus one std. The plus/minus variants only exist
# for features from --feature_selection bce/f1 (generate_clean_inference.py
# only tracks a std for those - see its "chosen_pos_mean"/"chosen_pos_std"
# npz keys - since only bce/f1 concentrate the whole vector at one latent;
# "auroc" averages over a top_k shortlist, so there's no single latent's std
# to add/subtract and only the "mean" variant is generated for those entries).

import os
import time

import numpy as np
import torch

from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from experiment_helpers.init_helpers import default_parser, repo_api_init

from attribution import DEFAULT_BLOCK_LIST
from sdxl_pipe import HookedStableDiffusionXLWithUNetPipeline
from experiment_helpers.image_helpers import concat_images_horizontally

from sam3_repo.sam3.model_builder import build_sam3_image_model

from generate_clean_swap import (
    load_sae, load_npz_dict, highlight_pixel_mask, compute_query_pixel_mask, make_position_hook_dict,
)

parser = default_parser(
    {
        "repo_id": "jlbaker361/nsfw"
    }
)

parser.add_argument("--base_prompt", type=str, required=True)  # single prompt for the one base image
parser.add_argument("--mask_target", type=str, required=True)  # SAM3 query for the region to inject features into
parser.add_argument("--npz_dict", type=str, default="platonic.npz")

parser.add_argument("--beta", type=float, default=1.0)  # injection strength multiplier on top of each variant's value

parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--end_step", type=int, default=1000)
parser.add_argument("--mode", type=str, default=None)  # "diff" or "out"; defaults to whatever generate_clean_inference.py recorded in npz_dict

parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--image_dest_dir", type=str, default="ablated_images")


def iter_npz_features(npz_data: dict, block_list: list):
    '''
    Every {query, block} feature vector generate_clean_inference.py saved,
    keyed "{query}__{block}" (mean_vec) with companion "__topk_idx"/
    "__topk_auc" bookkeeping and, for --feature_selection bce/f1 only,
    "__chosen_pos_mean"/"__chosen_pos_std" scalars for the single latent that
    actually matters there. "auroc"-built entries don't have those scalars.

    Yields (query, block, mean_vec, best_idx, pos_mean, pos_std) - the last
    three are None when this entry has no tracked std (an "auroc" entry).
    '''
    reserved_suffixes = ("__topk_idx", "__topk_auc", "__chosen_pos_mean", "__chosen_pos_std")
    for key in npz_data:
        if key == "__meta_mode__" or key.endswith(reserved_suffixes):
            continue
        block = next((b for b in block_list if key.endswith(f"__{b}")), None)
        if block is None:
            continue
        query = key[: -(len(block) + 2)]  # strip the trailing "__{block}"

        mean_vec = npz_data[key]
        mean_key, std_key, idx_key = f"{key}__chosen_pos_mean", f"{key}__chosen_pos_std", f"{key}__topk_idx"

        if mean_key in npz_data and std_key in npz_data:
            best_idx = int(npz_data[idx_key][0])
            pos_mean = float(npz_data[mean_key])
            pos_std = float(npz_data[std_key])
        else:
            best_idx = pos_mean = pos_std = None

        yield query, block, mean_vec, best_idx, pos_mean, pos_std


def build_variants(mean_vec: np.ndarray, best_idx, pos_mean, pos_std):
    '''
    ("mean", vec) always; ("plus_std", vec)/("minus_std", vec) only when this
    feature has a tracked single-latent std - those two just swap that one
    latent's value in an otherwise-identical copy of mean_vec. Negative values
    aren't clipped: sae_forward_swap's own torch.relu(vals) after top-k
    selection already floors a latent at 0 if it's selected with a negative
    pre-activation, so "minus_std" going negative just naturally reads as
    "suppress this latent" without any special-casing here.
    '''
    variants = [("mean", mean_vec)]
    if best_idx is not None:
        plus_vec = mean_vec.copy()
        plus_vec[best_idx] = pos_mean + pos_std
        variants.append(("plus_std", plus_vec))

        minus_vec = mean_vec.copy()
        minus_vec[best_idx] = pos_mean - pos_std
        variants.append(("minus_std", minus_vec))
    return variants


def main(args):
    api, accelerator, device = repo_api_init(args)
    os.makedirs(args.image_dest_dir, exist_ok=True)

    npz_data = load_npz_dict(args.npz_dict)
    mode = args.mode or str(npz_data.get("__meta_mode__", np.array("diff")))
    print(f"using mode='{mode}'")

    block_list = list(DEFAULT_BLOCK_LIST)
    on_cuda = device == "cuda" or (hasattr(device, "type") and device.type == "cuda")
    dtype = torch.float16 if (torch.cuda.is_available() and args.mixed_precision == "fp16") else torch.float32

    pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
        'stabilityai/sdxl-turbo',
        torch_dtype=dtype,
        variant=("fp16" if dtype == torch.float16 else None),
    )
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    if on_cuda:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # loaded on CPU and only moved to `device` transiently inside
    # compute_query_pixel_mask, same convention as generate_clean_swap.py
    sam3_model = build_sam3_image_model(device="cpu")

    base_gen = torch.Generator()
    base_gen.manual_seed(args.seed)
    base_image = pipe(args.base_prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                       num_inference_steps=args.num_inference_steps, generator=base_gen).images[0]

    pixel_mask = compute_query_pixel_mask(base_image, args.mask_target, sam3_model, device)
    safe_target = args.mask_target.replace(" ", "_")
    base_image.save(os.path.join(args.image_dest_dir, f"_base_{safe_target}.jpg"))
    highlight_pixel_mask(base_image, pixel_mask).save(
        os.path.join(args.image_dest_dir, f"_base_{safe_target}_highlighted.jpg")
    )

    sae_cache = {}

    def get_sae(block):
        if block not in sae_cache:
            sae_cache[block] = load_sae(block).to(device)
        return sae_cache[block]

    entries = list(iter_npz_features(npz_data, block_list))
    print(f"{len(entries)} features found in {args.npz_dict}")

    for query, block, mean_vec, best_idx, pos_mean, pos_std in entries:
        safe_query = query.replace(" ", "_")
        safe_block = block.replace(".", "_")
        variants = build_variants(mean_vec, best_idx, pos_mean, pos_std)
        if best_idx is None:
            print(f"'{query}' @ {block}: no tracked std (an 'auroc' entry) - only generating the mean variant")
        print(f"ablating '{query}' @ {block} into the '{args.mask_target}' mask region "
              f"({len(variants)} variant(s))...")

        sae = get_sae(block)
        panels = [base_image]
        for variant_name, vec in variants:
            to_vec = torch.tensor(vec, device=device, dtype=torch.float32)
            hook_dict = make_position_hook_dict(
                {block: sae}, {block: (None, to_vec)}, mode, args.start_step, args.end_step,
                1.0, args.beta, device, pixel_mask
            )
            gen = torch.Generator()
            gen.manual_seed(args.seed)  # same seed as the base image - only the masked region should differ
            out_image = pipe.run_with_hooks(
                args.base_prompt, position_hook_dict=hook_dict,
                height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps, generator=gen,
            ).images[0]
            panels.append(out_image)

        out_path = os.path.join(args.image_dest_dir, f"{safe_query}_{safe_block}.jpg")
        concat_images_horizontally(panels).save(out_path)

        if on_cuda:
            torch.cuda.empty_cache()


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
