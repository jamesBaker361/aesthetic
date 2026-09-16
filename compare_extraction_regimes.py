'''
Compares two ways of getting SAE-relevant UNet block activations for the same
image, to check how much the SAE's own training distribution ("generation
regime": activations produced while the model generates an image from pure
noise in a single turbo step) differs from the distribution used elsewhere in
this repo to extract features from existing photos ("encode regime": VAE-encode
a real image, add a small amount of noise, single UNet forward pass) - see
sdxl_extract.py / attribution.py / dino_pce.py for the encode-regime pattern
this mirrors.

Procedure, isolating the regime as the only variable (same image, same model
weights, same blocks):
  1. Generate an image with HookedStableDiffusionXLPipeline.run_with_cache -
     this is exactly how sdxl_unbox/scripts/collect_latents_dataset.py built
     the SAEs' own training data, so these are "in-distribution" activations.
  2. Feed that SAME generated image back in through the encode-regime pipeline
     (VAE encode + small noise + one UNet forward, matching sdxl_extract.py).
  3. For each block: cosine similarity + relative L2 distance between the two
     raw diff activations, and - the more relevant question for concept
     steering - how much the SAE's *sparse codes* agree (whether the two
     regimes even activate the same top-k feature indices).
'''
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL

from sdxl_unbox.SDLens.hooked_sd_pipeline import HookedStableDiffusionXLPipeline
from sdxl_unbox.SAE import SparseAutoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"
# float32 SDXL-turbo (~3.5B params across UNet + 2 text encoders + VAE) needs
# ~14GB just for weights, which OOMs on the smaller nodes in the generic gpu
# pool (10.58GB seen on g20-01). fp16 halves that to ~7GB. Both extraction
# regimes go through the same dtype, so fp16 rounding is common-mode and
# shouldn't bias the regime-to-regime comparison itself.
dtype = torch.float16 if device == "cuda" else torch.float32

BLOCKS = [
    "down_blocks.2.attentions.1",
    "mid_block.attentions.0",
    "up_blocks.0.attentions.0",
    "up_blocks.0.attentions.1",
]
PROMPT = "a photo of a banana"
PATH_TO_CHECKPOINTS = "./sdxl_unbox/checkpoints/"

pipe = HookedStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=dtype
)
pipe.to(device)
pipe.set_progress_bar_config(disable=True)
pipe.vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
).to(device)

saes = {}
for block in BLOCKS:
    saes[block] = SparseAutoencoder.load_from_disk(
        os.path.join(
            PATH_TO_CHECKPOINTS,
            f"unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001",
            "final",
        )
    ).to(device)  # load_from_disk loads on CPU; a_flat/b_flat are GPU tensors

# --- Step 1: generation regime - exactly matches collect_latents_dataset.py ---
positions_to_cache = [f"unet.{block}" for block in BLOCKS]
output, cache = pipe.run_with_cache(
    prompt=PROMPT,
    positions_to_cache=positions_to_cache,
    save_input=True,
    save_output=True,
    num_inference_steps=1,
    guidance_scale=0.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
    output_type="pil",
)
gen_image = output.images[0]
gen_image.save("compare_regimes_image.png")

gen_diff = {}
for block in BLOCKS:
    pos = f"unet.{block}"
    diff = cache["output"][pos] - cache["input"][pos]  # [B,T,C,H,W], T=1 (single inference step)
    gen_diff[block] = diff.permute(0, 1, 3, 4, 2).squeeze(0).squeeze(0).to(dtype)  # [H,W,C]

# --- Step 2: encode regime - mirrors sdxl_extract.py's real-photo pipeline ---
saved = {}
def make_hook(block):
    def hook(module, input, output):
        saved[block] = {"input": input, "output": output}
        return output
    return hook

handles = []
for name, module in pipe.unet.named_modules():
    if name in BLOCKS:
        handles.append(module.register_forward_hook(make_hook(name)))

image_processor = VaeImageProcessor()
with torch.no_grad():
    image_pt = image_processor.preprocess(gen_image).to(device=device, dtype=dtype)
    size = image_pt.shape[-1]

    latents = pipe.vae.config.scaling_factor * pipe.vae.encode(image_pt).latent_dist.sample()
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, 10, (latents.shape[0],), device=latents.device).long()  # "very low noise", matches sdxl_extract.py
    noisy_model_input = pipe.scheduler.add_noise(latents, noise, timesteps)

    (prompt_embeds, _, pooled_prompt_embeds, _) = pipe.encode_prompt(
        "image", "image", device, 1, False, " ", " "
    )
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
    add_time_ids = pipe._get_add_time_ids(
        (size, size), (0, 0), (size, size),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    ).to(device)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds.to(device), "time_ids": add_time_ids}

    pipe.unet.forward(
        noisy_model_input, timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )

for h in handles:
    h.remove()

enc_diff = {}
for block in BLOCKS:
    out = saved[block]["output"]
    inp = saved[block]["input"]
    out = out[0] if isinstance(out, tuple) else out
    inp = inp[0] if isinstance(inp, tuple) else inp
    diff = (out - inp).squeeze(0).permute(1, 2, 0).to(dtype)  # [H,W,C]
    enc_diff[block] = diff

# --- Step 3: compare ---
print(f"{'block':<28} {'cos_sim (mean)':>15} {'rel_l2':>10} {'sae_cos (mean)':>16} {'topk_jaccard':>13}")
fig, axes = plt.subplots(1, len(BLOCKS), figsize=(5 * len(BLOCKS), 4))
for ax, block in zip(axes, BLOCKS):
    a = gen_diff[block]
    b = enc_diff[block]
    if a.shape[:2] != b.shape[:2]:
        # align spatial resolution before comparing, nearest to avoid blending
        b = F.interpolate(
            b.permute(2, 0, 1)[None], size=a.shape[:2], mode="nearest"
        )[0].permute(1, 2, 0)

    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])

    cos_sim = F.cosine_similarity(a_flat, b_flat, dim=-1)
    rel_l2 = (torch.linalg.norm(a_flat - b_flat, dim=-1) /
              (torch.linalg.norm(a_flat, dim=-1) + torch.linalg.norm(b_flat, dim=-1) + 1e-8))

    sae = saes[block]
    with torch.no_grad():
        # SparseAutoencoder's own weights stay float32 regardless of pipe
        # dtype (load_state_dict preserves the destination param's dtype),
        # so cast here to avoid a dtype-mismatch crash under fp16
        a_codes = sae.encode(a_flat.float())
        b_codes = sae.encode(b_flat.float())
    sae_cos = F.cosine_similarity(a_codes, b_codes, dim=-1)

    a_active = a_codes > 0
    b_active = b_codes > 0
    intersection = (a_active & b_active).sum(dim=-1).float()
    union = (a_active | b_active).sum(dim=-1).float()
    jaccard = (intersection / union.clamp(min=1)).mean()

    print(f"{block:<28} {cos_sim.mean().item():>15.3f} {rel_l2.mean().item():>10.3f} "
          f"{sae_cos.mean().item():>16.3f} {jaccard.item():>13.3f}")

    ax.hist(cos_sim.detach().cpu().numpy(), bins=40, color="steelblue")
    ax.set_title(block, fontsize=9)
    ax.set_xlabel("per-pixel raw-activation cosine sim")

plt.tight_layout()
plt.savefig("compare_regimes_cosine_hist.png")
plt.close()
print("\nSaved generated image to compare_regimes_image.png and per-block cosine-similarity histograms to compare_regimes_cosine_hist.png")
