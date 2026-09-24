# sanity check for generate_clean_ablate.py's injection mechanism
# (sae_decode_add/make_add_hook) using sdxl_unbox's own documented "known"
# features (see its app.py "Paint!" tab) instead of anything this repo
# discovered itself via SAM3/AUROC/BCE/f1. If injecting down.2.1 #4998
# visibly cartoonifies the output, up.0.1 #4977 visibly adds tiger stripes,
# etc., that confirms the SAE loading + decode + hook wiring actually work,
# independent of whether the feature-discovery side of the pipeline is any
# good - so a failure here means "the injection mechanism is broken", and a
# pass means any failure to find good ablation images elsewhere is a
# feature-selection/masking problem, not a plumbing problem.
#
# --mask_target is optional: with none given, the feature is injected
# everywhere (whole-image), which is the clearest test of the injection
# mechanism (sae_decode_add/make_add_hook) on its own. Passing --mask_target
# additionally sanity-checks the SAM3 masking machinery, by restricting the
# same known-good injection to just that masked region - e.g. --mask_target
# person should show the cartoon/fur/etc. effect appear only on the person
# and nowhere else, confirming compute_query_pixel_mask + resize_mask_to_grid
# + the patch_mask multiply in sae_decode_add all line up correctly.

import os
import time

import numpy as np
import torch

from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from experiment_helpers.init_helpers import default_parser, repo_api_init

from sdxl_pipe import HookedStableDiffusionXLWithUNetPipeline
from experiment_helpers.image_helpers import concat_images_horizontally

from sam3_repo.sam3.model_builder import build_sam3_image_model

from generate_clean_swap import load_sae, load_feature_mean, highlight_pixel_mask, compute_query_pixel_mask
from generate_clean_ablate import make_add_position_hook_dict

parser = default_parser(
    {
        "repo_id": "jlbaker361/nsfw"
    }
)

parser.add_argument("--base_prompt", type=str, default="a photo of a man standing outside")
parser.add_argument("--mask_target", type=str, default=None)  # SAM3 query for the region to inject into; None = whole image
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--image_dest_dir", type=str, default="known_feature_sanity")

# each known feature gets one panel: base image + one image per strength, so
# you can see the effect strengthen (or, if the mechanism is broken, fail to
# strengthen) across the sweep rather than judging a single fixed magnitude
STRENGTHS = [1.0, 2.0, 5.0, 10.0]

# (block, feature_idx, human label) - the block names match attribution.py's
# DEFAULT_BLOCK_LIST / load_sae's naming, not app.py's short "down.2.1" codes
KNOWN_FEATURES = [
    ("down_blocks.2.attentions.1", 4998, "cartoon"),
    ("down_blocks.2.attentions.1", 230, "furry"),
    ("down_blocks.2.attentions.1", 89, "muscleman"),
    ("down_blocks.2.attentions.1", 4074, "anime"),
    ("up_blocks.0.attentions.1", 4977, "tiger_stripes"),
    ("up_blocks.0.attentions.1", 90, "fur"),
    ("up_blocks.0.attentions.1", 2165, "twilight_blur"),
]


def main(args):
    api, accelerator, device = repo_api_init(args)
    os.makedirs(args.image_dest_dir, exist_ok=True)

    base_prompt = args.base_prompt.replace("_", " ")
    on_cuda = device == "cuda" or (hasattr(device, "type") and device.type == "cuda")
    dtype = torch.float16 if (torch.cuda.is_available() and args.mixed_precision == "fp16") else torch.float32

    pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
        'stabilityai/sdxl-turbo',
        torch_dtype=dtype,
        variant=("fp16" if dtype == torch.float16 else None),
    )
    pipe.pipe.vae.enable_slicing()
    pipe.pipe.enable_attention_slicing()
    if on_cuda:
        pipe.pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_gen = torch.Generator()
    base_gen.manual_seed(args.seed)
    base_image = pipe(base_prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                       num_inference_steps=args.num_inference_steps, generator=base_gen).images[0]
    base_image.save(os.path.join(args.image_dest_dir, "_base.jpg"))

    pixel_mask = None
    if args.mask_target:
        # loaded on CPU and only moved to `device` transiently inside
        # compute_query_pixel_mask, same convention as generate_clean_swap.py
        sam3_model = build_sam3_image_model(device="cpu")
        pixel_mask = compute_query_pixel_mask(base_image, args.mask_target, sam3_model, device)
        safe_target = args.mask_target.replace(" ", "_")
        highlight_pixel_mask(base_image, pixel_mask).save(
            os.path.join(args.image_dest_dir, f"_base_{safe_target}_highlighted.jpg")
        )

    sae_cache = {}

    def get_sae(block):
        if block not in sae_cache:
            sae_cache[block] = load_sae(block, "local").to(device)
        return sae_cache[block]

    for block, feature_idx, label in KNOWN_FEATURES:
        pos_mean = load_feature_mean(block, feature_idx, device)
        vec = np.zeros(5120, dtype=np.float32)
        vec[feature_idx] = pos_mean
        to_vec = torch.tensor(vec, device=device, dtype=torch.float32)

        sae = get_sae(block)
        panels = [base_image]
        for strength in STRENGTHS:
            # pixel_mask is None (whole image) unless --mask_target was given
            hook_dict = make_add_position_hook_dict(
                {block: sae}, {block: to_vec}, 0, 1000, strength, device, pixel_mask=pixel_mask
            )
            gen = torch.Generator()
            gen.manual_seed(args.seed)  # same seed as the base image
            out_image = pipe.run_with_hooks(
                base_prompt, position_hook_dict=hook_dict,
                height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps, generator=gen,
            ).images[0]
            panels.append(out_image)
            print(f"{block} #{feature_idx} ({label}): strength={strength} pos_mean={pos_mean:.4f} "
                  f"injected={strength * pos_mean:.4f}")

        safe_block = block.replace(".", "_")
        out_path = os.path.join(args.image_dest_dir, f"{safe_block}_{feature_idx}_{label}.jpg")
        concat_images_horizontally(panels).save(out_path)
        print(f"-> {out_path}")

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
    print(f"successful generating:) time elapsed: {seconds} seconds = {seconds / 3600} hours")
    print("all done!")
