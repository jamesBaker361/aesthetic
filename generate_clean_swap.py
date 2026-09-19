# given the npz_dict of "{query}__{block}" -> SAE embedding produced by
# generate_clean_inference.py, generate images from a prompt file while
# either removing a query concept (subtract its embedding from the SAE
# latents before top-k, SAEURON style) or replacing it with another query's
# embedding (subtract the source query's vector and inject the target
# query's vector in its place) at every hooked UNet block.

import os
import time
import argparse

import numpy as np
import torch

from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from experiment_helpers.init_helpers import default_parser, repo_api_init

from sdxl_unbox.SAE import SparseAutoencoder
from sdxl_pipe import HookedStableDiffusionXLWithUNetPipeline
from attribution import DEFAULT_BLOCK_LIST
from generate_clean_inference import read_prompts
from experiment_helpers.image_helpers import concat_images_horizontally

parser = default_parser(
    {
        "repo_id": "jlbaker361/nsfw"
    }
)

parser.add_argument("--prompt_file", type=str, default="prompts.txt")
parser.add_argument("--npz_dict", type=str, default="platonic.npz")

parser.add_argument("--query", type=str, required=True)  # concept to remove (or replace)
parser.add_argument("--replace_query", type=str, default=None)  # if set, inject this concept's embedding in query's place instead of just removing it

parser.add_argument("--alpha", type=float, default=1.0)  # subtraction strength for --query
parser.add_argument("--beta", type=float, default=1.0)  # injection strength for --replace_query

parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--end_step", type=int, default=1000)
parser.add_argument("--mode", type=str, default=None)  # "diff" or "out"; defaults to whatever generate_clean_inference.py recorded in npz_dict

parser.add_argument("--num_inference_steps", type=int, default=8)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--image_dest_dir", type=str, default="swapped_images")
parser.add_argument("--save_baseline", action="store_true")  # concat the no-hook baseline (same seed+prompt) alongside the swapped image

SAE_CHECKPOINTS = "./sdxl_unbox/checkpoints/"


def load_sae(block: str) -> SparseAutoencoder:
    return SparseAutoencoder.load_from_disk(
        os.path.join(SAE_CHECKPOINTS, f"unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final"),
    )


def load_npz_dict(npz_dict_path: str) -> dict:
    with np.load(npz_dict_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def get_query_vec(npz_data: dict, query: str, block: str, device):
    key = f"{query}__{block}"
    if key not in npz_data:
        return None
    return torch.tensor(npz_data[key], device=device, dtype=torch.float32)


def sae_forward_swap(sae: SparseAutoencoder, x: torch.Tensor, from_vec, alpha: float, to_vec, beta: float):
    # x arrives as (B,H,W,d_model); decode_sparse (unlike sae.encode) hard-codes
    # rows, cols = inds.shape[0], self.n_dirs and only accepts 2D (N,k) inds -
    # flatten the spatial dims before topk/decode and restore them after
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    x = x - sae.pre_bias
    latents_pre_act = sae.encoder(x) + sae.latent_bias
    if from_vec is not None:
        latents_pre_act = latents_pre_act - alpha * from_vec
    if to_vec is not None:
        latents_pre_act = latents_pre_act + beta * to_vec
    vals, inds = torch.topk(latents_pre_act, k=sae.k, dim=-1)
    recons = sae.decode_sparse(inds, torch.relu(vals))
    return recons.reshape(*orig_shape[:-1], recons.shape[-1])


def make_swap_hook(sae: SparseAutoencoder, from_vec, to_vec, mode: str, start_step: int, end_step: int,
                    alpha: float, beta: float, device):
    # own step counter per hook closure - a fresh one is built for every
    # pipe.run_with_hooks() call, so it doesn't need resetting between prompts
    step_counter = {"step": 0}

    def hook_fn(module, input, output):
        step = step_counter["step"]
        if start_step <= step <= end_step:
            out = output[0] if isinstance(output, tuple) else output
            inp = input[0] if isinstance(input, tuple) else input
            orig_dtype = out.dtype  # the SAE runs in float32 regardless of the pipe's dtype (fp16)
            if mode == "diff":
                out = out - inp
            # hook output is channel-first (B,C,H,W); the SAE (as everywhere
            # else it's used, e.g. sparsify.py) expects channel-last (...,d_model)
            out = sae_forward_swap(sae, out.permute(0, 2, 3, 1), from_vec, alpha, to_vec, beta)
            out = out.permute(0, 3, 1, 2).to(device=device, dtype=orig_dtype)
            output = (out, *output[1:]) if isinstance(output, tuple) else out
        step_counter["step"] = step + 1
        return output

    return hook_fn


def make_position_hook_dict(sae_dict: dict, vec_dict: dict, mode: str, start_step: int, end_step: int,
                             alpha: float, beta: float, device) -> dict:
    position_hook_dict = {}
    for block, sae in sae_dict.items():
        from_vec, to_vec = vec_dict[block]
        position_hook_dict[f"unet.{block}"] = make_swap_hook(
            sae, from_vec, to_vec, mode, start_step, end_step, alpha, beta, device
        )
    return position_hook_dict


def main(args):
    api, accelerator, device = repo_api_init(args)

    query: str = args.query
    replace_query = args.replace_query
    os.makedirs(args.image_dest_dir, exist_ok=True)

    npz_data = load_npz_dict(args.npz_dict)
    mode = args.mode or str(npz_data.get("__meta_mode__", np.array("diff")))
    print(f"using mode='{mode}'")

    block_list = list(DEFAULT_BLOCK_LIST)

    sae_dict, vec_dict = {}, {}
    for block in block_list:
        from_vec = get_query_vec(npz_data, query, block, device)
        to_vec = get_query_vec(npz_data, replace_query, block, device) if replace_query else None
        if from_vec is None and to_vec is None:
            print(f"skipping {block}: no embedding for '{query}'" + (f" or '{replace_query}'" if replace_query else ""))
            continue
        sae_dict[block] = load_sae(block).to(device)
        vec_dict[block] = (from_vec, to_vec)

    if not sae_dict:
        raise ValueError(f"no blocks had a saved embedding for '{query}' in {args.npz_dict}")

    dtype = torch.float16 if (torch.cuda.is_available() and args.mixed_precision == "fp16") else torch.float32
    pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
        'stabilityai/sdxl-turbo',
        torch_dtype=dtype,
        variant=("fp16" if dtype == torch.float16 else None),
    ).to(device)

    prompts = read_prompts(args.prompt_file)
    suffix = f"remove_{query}" if not replace_query else f"{query}_to_{replace_query}"
    safe_suffix = suffix.replace(" ", "_")

    # vec_dict with every to_vec dropped: subtracts the removed query's
    # embedding same as a full swap would, but never injects replace_query's
    removal_only_vec_dict = {block: (from_vec, None) for block, (from_vec, _) in vec_dict.items()}

    for i, prompt in enumerate(prompts):
        panels = []

        if args.save_baseline:
            # same seed + prompt as the hooked runs below, just without any
            # hooks registered, so it's the "before" half of the comparison
            baseline_gen = torch.Generator()
            baseline_gen.manual_seed(i)
            baseline_image = pipe(prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                                   num_inference_steps=args.num_inference_steps, generator=baseline_gen).images[0]
            panels.append(baseline_image)

        if replace_query:
            # same seed again, hooked but only ever subtracting from_vec -
            # shows removal alone, without replace_query's injection - always
            # included when doing a replace, not just when --save_baseline
            removal_gen = torch.Generator()
            removal_gen.manual_seed(i)
            removal_hook_dict = make_position_hook_dict(
                sae_dict, removal_only_vec_dict, mode, args.start_step, args.end_step, args.alpha, args.beta, device
            )
            removal_only_image = pipe.run_with_hooks(
                prompt, position_hook_dict=removal_hook_dict,
                height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps, generator=removal_gen,
            ).images[0]
            panels.append(removal_only_image)

        rand_gen = torch.Generator()
        rand_gen.manual_seed(i)
        # a fresh position_hook_dict per prompt - run_with_hooks registers
        # these and removes them again once pipe() returns, so each hook's
        # own step_counter naturally starts at 0 for every generation
        position_hook_dict = make_position_hook_dict(
            sae_dict, vec_dict, mode, args.start_step, args.end_step, args.alpha, args.beta, device
        )
        swapped_image = pipe.run_with_hooks(
            prompt, position_hook_dict=position_hook_dict,
            height=args.size, width=args.size, guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps, generator=rand_gen,
        ).images[0]
        panels.append(swapped_image)

        out_image = concat_images_horizontally(panels) if len(panels) > 1 else swapped_image
        out_image.save(os.path.join(args.image_dest_dir, f"{safe_suffix}_{i}.jpg"))


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
