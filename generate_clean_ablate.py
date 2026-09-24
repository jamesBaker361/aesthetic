# given a single base prompt and a mask_target, generates one base image and
# a SAM3 mask for mask_target on it, then for every {query, block} feature
# saved in --npz_dict (by generate_clean_inference.py), regenerates the same
# image (same seed/prompt) injecting that feature ONLY inside the masked
# region - background stays anchored to the base image, same injection
# sdxl_unbox/utils/hooks.py's add_feature_on_area_turbo uses for its own demo
# app: build an all-zero latent vector except at the injected latent(s),
# decode it through just the SAE's linear decoder (no pre_bias - this is a
# delta, not a full reconstruction), and add that delta straight onto the
# block's real output. Unlike generate_clean_swap.py's make_swap_hook (which
# adds the injected vector to the block's *pre*-top-k activation and then
# re-runs top-k, so the injection can knock one of the patch's own genuinely
# active latents out of the reconstruction), the patch's own activation is
# never encoded or touched at all here - see make_add_hook/sae_decode_add
# below.
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
from sdxl_unbox.SAE import SparseAutoencoder

from sam3_repo.sam3.model_builder import build_sam3_image_model

from generate_clean_swap import load_sae, load_npz_dict, highlight_pixel_mask, compute_query_pixel_mask
from generate_clean_inference import resize_mask_to_grid

parser = default_parser(
    {
        "repo_id": "jlbaker361/nsfw"
    }
)

parser.add_argument("--base_prompt", type=str, required=True)  # single prompt for the one base image
parser.add_argument("--mask_target", type=str, required=True)  # SAM3 query for the region to inject features into
parser.add_argument("--npz_dict", type=str, default="platonic.npz")

parser.add_argument("--sae_source", type=str, default="local", choices=["local", "saeuron"],
                     help="'local': this repo's own trained checkpoints (default). 'saeuron': checkpoints converted "
                          "by convert_saeuron_checkpoint.py from github.com/cywinski/SAeUron - pair with "
                          "--block_list since they don't overlap DEFAULT_BLOCK_LIST.")
parser.add_argument("--block_list", nargs="*", default=None,
                     help="overrides attribution.DEFAULT_BLOCK_LIST - required when --sae_source=saeuron")

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

    Yields (query, block, mean_vec, top_idx, best_idx, pos_mean, pos_std) -
    top_idx is the saved "__topk_idx" array (one entry for bce/f1, up to
    --top_k for auroc); best_idx/pos_mean/pos_std are None when this entry
    has no tracked std (an "auroc" entry).
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
        top_idx = npz_data[idx_key]

        if mean_key in npz_data and std_key in npz_data:
            best_idx = int(top_idx[0])
            pos_mean = float(npz_data[mean_key])
            pos_std = float(npz_data[std_key])
        else:
            best_idx = pos_mean = pos_std = None

        yield query, block, mean_vec, top_idx, best_idx, pos_mean, pos_std


def build_variants(mean_vec: np.ndarray, top_idx: np.ndarray, best_idx, pos_mean, pos_std):
    '''
    The injected vector is all-zero except at the selected latent(s).

    - bce/f1 (best_idx is not None): a single chosen latent - its value comes
      straight from the tracked pos_mean/pos_std scalars, never from the
      saved mean_vec array (mean_vec happens to already equal pos_mean there,
      but pos_mean is the actual source of truth, not a byproduct of it).
    - auroc (best_idx is None): a top_idx shortlist with no per-latent scalar
      tracked for them, so mean_vec[top_idx] - the mean over positive patches
      of already-top-k-sparse-per-patch activations, restricted to just the
      shortlisted latents - is the only per-latent magnitude available.

    Either way only the selected latent(s) end up nonzero; because
    sae_forward_swap adds beta * to_vec into the pre-activation, the zeroed
    entries contribute nothing and only the targeted latents' pre-activations
    are actually added to.

    ("mean", vec) always; ("plus_std", vec)/("minus_std", vec) only when this
    feature has a tracked single-latent std - those two just swap that one
    latent's value in an otherwise-identical copy of the sparse vector.
    Negative values aren't clipped: sae_forward_swap's own torch.relu(vals)
    after top-k selection already floors a latent at 0 if it's selected with
    a negative pre-activation, so "minus_std" going negative just naturally
    reads as "suppress this latent" without any special-casing here.
    '''
    sparse_vec = np.zeros_like(mean_vec)
    if best_idx is not None:
        sparse_vec[best_idx] = pos_mean
    else:
        sparse_vec = mean_vec

    variants = [("mean", sparse_vec)]
    if best_idx is not None:
        plus_vec = sparse_vec.copy()
        plus_vec[best_idx] = pos_mean + pos_std
        variants.append(("plus_std", plus_vec))

        minus_vec = sparse_vec.copy()
        minus_vec[best_idx] = pos_mean - pos_std
        variants.append(("minus_std", minus_vec))
    return variants


def sae_decode_add(sae: SparseAutoencoder, x: torch.Tensor, to_vec: torch.Tensor, beta: float,
                    patch_mask: torch.Tensor = None) -> torch.Tensor:
    '''
    sdxl_unbox/utils/hooks.py's add_feature_on_area_turbo, generalized from a
    single feature_idx/value to the full sparse to_vec from build_variants:
    build an all-zero latent vector except at the injected latent(s), decode
    it through just the SAE's linear decoder - no pre_bias, since this is a
    delta being added on top of a real activation, not a full reconstruction
    - and add beta * that delta straight onto x. x's own activation is never
    encoded or otherwise touched; only the injected direction is added, and
    only inside patch_mask.
    '''
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1])

    edit_mask = 1.0
    if patch_mask is not None:
        b = orig_shape[0]
        edit_mask = patch_mask.reshape(-1).to(flat.dtype).repeat(b).unsqueeze(-1)

    mask = beta * to_vec.to(flat.dtype) * edit_mask  # (N, n_dirs)
    to_add = sae.decoder(mask)  # linear decode, bias=False -> (N, C)
    return (flat + to_add).reshape(orig_shape)


def make_add_hook(sae: SparseAutoencoder, to_vec: torch.Tensor, start_step: int, end_step: int,
                   beta: float, device, pixel_mask: np.ndarray = None):
    # same step-counter/patch_mask-cache structure as generate_clean_swap.py's
    # make_swap_hook, minus the "mode"/from_vec/encode/top-k machinery - see
    # sae_decode_add for what actually changed
    step_counter = {"step": 0}
    patch_mask_cache = {}

    def hook_fn(module, input, output):
        step = step_counter["step"]
        if start_step <= step <= end_step:
            out = output[0] if isinstance(output, tuple) else output
            orig_dtype = out.dtype

            patch_mask = None
            if pixel_mask is not None:
                grid_h, grid_w = out.shape[2], out.shape[3]  # out is channel-first (B,C,H,W) here
                if (grid_h, grid_w) not in patch_mask_cache:
                    resized = resize_mask_to_grid(pixel_mask, grid_h, grid_w)
                    patch_mask_cache[(grid_h, grid_w)] = torch.from_numpy(resized).to(device)
                patch_mask = patch_mask_cache[(grid_h, grid_w)]

            out = sae_decode_add(sae, out.permute(0, 2, 3, 1).float(), to_vec, beta, patch_mask)
            out = out.permute(0, 3, 1, 2).to(device=device, dtype=orig_dtype)
            output = (out, *output[1:]) if isinstance(output, tuple) else out
        step_counter["step"] = step + 1
        return output

    return hook_fn


def make_add_position_hook_dict(sae_dict: dict, vec_dict: dict, start_step: int, end_step: int,
                                 beta: float, device, pixel_mask: np.ndarray = None) -> dict:
    position_hook_dict = {}
    for block, sae in sae_dict.items():
        to_vec = vec_dict[block]
        position_hook_dict[f"unet.{block}"] = make_add_hook(
            sae, to_vec, start_step, end_step, beta, device, pixel_mask
        )
    return position_hook_dict


def main(args):
    api, accelerator, device = repo_api_init(args)
    os.makedirs(args.image_dest_dir, exist_ok=True)
    
    base_prompt=args.base_prompt.replace("_"," ")

    npz_data = load_npz_dict(args.npz_dict)
    mode = args.mode or str(npz_data.get("__meta_mode__", np.array("diff")))
    print(f"using mode='{mode}'")

    block_list = args.block_list if args.block_list else list(DEFAULT_BLOCK_LIST)
    on_cuda = device == "cuda" or (hasattr(device, "type") and device.type == "cuda")
    dtype = torch.float16 if (torch.cuda.is_available() and args.mixed_precision == "fp16") else torch.float32

    pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
        'stabilityai/sdxl-turbo',
        torch_dtype=dtype,
        variant=("fp16" if dtype == torch.float16 else None),
    )
    # HookedStableDiffusionXLWithUNetPipeline doesn't define these itself -
    # go through .pipe (the wrapped diffusers pipeline) explicitly rather
    # than relying on HookedDiffusionAbstractPipeline's __getattr__ proxy
    pipe.pipe.vae.enable_slicing()  # enable_vae_slicing() was removed from the pipeline itself in this diffusers version
    pipe.pipe.enable_attention_slicing()
    if on_cuda:
        # keeps the UNet/VAE/text-encoders on GPU only while each is actually
        # running instead of the whole pipe sitting resident for the entire
        # script - without this, SAM3's own GPU use (right after the base
        # image generation, before any hooked generation) OOMs on smaller
        # GPUs since the full fp16 SDXL pipe never gets a chance to shrink
        pipe.pipe.enable_model_cpu_offload()
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
    base_image = pipe(base_prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
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
            sae_cache[block] = load_sae(block, args.sae_source).to(device)
        return sae_cache[block]

    entries = list(iter_npz_features(npz_data, block_list))
    print(f"{len(entries)} features found in {args.npz_dict}")

    # if different queries' bce/f1 selection all converged on the same
    # best_idx for a block, every one of their ablations will look alike
    # regardless of which query "caused" them - surface that directly rather
    # than leaving it to be inferred from the output images
    idx_by_block = {}
    for query, block, _mean_vec, _top_idx, best_idx, _pos_mean, _pos_std in entries:
        if best_idx is not None:
            idx_by_block.setdefault(block, []).append((query, best_idx))
    for block, pairs in idx_by_block.items():
        unique_idxs = sorted(set(idx for _, idx in pairs))
        print(f"{block}: {len(pairs)} queries -> {len(unique_idxs)} distinct best_idx {unique_idxs}")
        if len(unique_idxs) < len(pairs):
            from collections import Counter
            dupes = {idx: [q for q, i in pairs if i == idx] for idx, c in Counter(i for _, i in pairs).items() if c > 1}
            print(f"  ! shared best_idx across queries: {dupes}")

    for query, block, mean_vec, top_idx, best_idx, pos_mean, pos_std in entries:
        safe_query = query.replace(" ", "_")
        safe_block = block.replace(".", "_")
        variants = build_variants(mean_vec, top_idx, best_idx, pos_mean, pos_std)
        if best_idx is None:
            print(f"'{query}' @ {block}: no tracked std (an 'auroc' entry) - only generating the mean variant")
        print(f"ablating '{query}' @ {block} (best_idx={best_idx}) into the '{args.mask_target}' mask region "
              f"({len(variants)} variant(s))...")

        sae = get_sae(block)
        panels = [base_image]
        for variant_name, vec in variants:
            to_vec = torch.tensor(vec, device=device, dtype=torch.float32)
            hook_dict = make_add_position_hook_dict(
                {block: sae}, {block: to_vec}, args.start_step, args.end_step, args.beta, device, pixel_mask
            )
            gen = torch.Generator()
            gen.manual_seed(args.seed)  # same seed as the base image - only the masked region should differ
            out_image = pipe.run_with_hooks(
                base_prompt, position_hook_dict=hook_dict,
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
