# Downloads a pretrained SAE checkpoint from SAeUron (github.com/cywinski/SAeUron,
# HF hub "bcywinski/SAeUron") and converts it into this repo's SparseAutoencoder
# checkpoint format (config.json + state_dict.pth), so it loads through the
# exact same SparseAutoencoder.load_from_disk path as every other checkpoint
# under sdxl_unbox/checkpoints/ - no changes to sae.py itself.
#
# Both are TopK SAEs computing the same thing - relu(topk(W_enc(x - b_pre) +
# b_enc)) -> decode -> + b_pre - just with the parameters split up
# differently. Verified against SAeUron's own SAE/sae.py and SAE/config.py:
#
#   SAeUron "encoder.weight" (n_dirs, d_model) -> "encoder.weight"  direct copy (same shape/orientation)
#   SAeUron "encoder.bias"   (n_dirs,)         -> "latent_bias"     direct copy (this repo's encoder Linear
#                                                                    has bias=False; the bias instead lives
#                                                                    in its own latent_bias parameter)
#   SAeUron "W_dec"          (n_dirs, d_model) -> "decoder.weight"  TRANSPOSED (this repo's decoder Linear
#                                                                    stores weight as (d_model, n_dirs))
#   SAeUron "b_dec"          (d_model,)        -> "pre_bias"        direct copy
#
# One behavioral nuance, not corrected here: SAeUron applies ReLU *before*
# top-k selection (pre_acts() relu's the whole vector, then select_topk);
# this repo's SparseAutoencoder does top-k *then* ReLU on just the selected
# values. These only disagree when fewer than k latents have a positive
# pre-activation for a given input - negligible at typical n_dirs >> k.
#
# auxk/dead_steps_threshold are training-only fields, never read by
# encode()/decode_sparse() (only by the training-time forward() path) - set
# to inert placeholders (auxk=None disables that branch entirely).
#
# SAeUron's public checkpoints only cover two hookpoints, neither of which
# is in attribution.DEFAULT_BLOCK_LIST: "up_blocks.1.attentions.1" (object
# unlearning) and "up_blocks.1.attentions.2" (style unlearning).

import os
import json
import argparse

import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from sdxl_unbox.SAE.sae import SparseAutoencoder

SAEURON_HF_REPO = "bcywinski/SAeUron"
SAEURON_HOOKPOINTS = ["up_blocks.1.attentions.1", "up_blocks.1.attentions.2"]


def download_saeuron_checkpoint(hookpoint: str, cache_dir: str = None) -> str:
    # SAeUron's own HF repo stores each hookpoint's checkpoint under a
    # "unet.{hookpoint}" folder (their naming convention, confirmed against
    # their own scripts/load_from_hub.py usage) - translated here, at this
    # one boundary, so `hookpoint` stays bare (no "unet." prefix) everywhere
    # else in this repo, consistent with how every other block name is used
    # (hook registration, sparsify, etc. all build "unet.{block}" themselves)
    hf_hookpoint = f"unet.{hookpoint}"
    repo_path = snapshot_download(SAEURON_HF_REPO, allow_patterns=f"{hf_hookpoint}/*", cache_dir=cache_dir)
    return os.path.join(repo_path, hf_hookpoint)


def convert_saeuron_checkpoint(saeuron_dir: str, out_dir: str) -> SparseAutoencoder:
    with open(os.path.join(saeuron_dir, "cfg.json")) as f:
        cfg = json.load(f)

    d_model = cfg["d_in"]
    n_dirs = cfg.get("num_latents") or d_model * cfg["expansion_factor"]  # same fallback Sae.__init__ uses
    k = cfg["k"]

    state = load_file(os.path.join(saeuron_dir, "sae.safetensors"))

    sae = SparseAutoencoder(
        n_dirs_local=n_dirs,
        d_model=d_model,
        k=k,
        auxk=None,
        dead_steps_threshold=1_000_000,
    )

    with torch.no_grad():
        sae.encoder.weight.copy_(state["encoder.weight"])
        sae.latent_bias.copy_(state["encoder.bias"])
        sae.decoder.weight.copy_(state["W_dec"].T)
        sae.pre_bias.copy_(state["b_dec"])

    sae.save_to_disk(out_dir)
    print(f"converted SAeUron checkpoint '{saeuron_dir}' -> '{out_dir}' (n_dirs={n_dirs}, d_model={d_model}, k={k})")
    return sae


def saeuron_checkpoint_dir(checkpoints_dir: str, hookpoint: str) -> str:
    # mirrors the local convention (unet.{block}_k{k}_hidden{n_dirs}_...) with
    # a "saeuron" marker so converted checkpoints never collide with locally
    # trained ones for the same block name
    return os.path.join(checkpoints_dir, f"saeuron.unet.{hookpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hookpoint", type=str, required=True, choices=SAEURON_HOOKPOINTS)
    parser.add_argument("--checkpoints_dir", type=str, default="./sdxl_unbox/checkpoints/",
                         help="matches SAE_CHECKPOINTS in generate_clean_swap.py / path_to_checkpoints in sparsify.py")
    args = parser.parse_args()

    saeuron_dir = download_saeuron_checkpoint(args.hookpoint)
    out_dir = os.path.join(saeuron_checkpoint_dir(args.checkpoints_dir, args.hookpoint), "final")
    convert_saeuron_checkpoint(saeuron_dir, out_dir)


if __name__ == "__main__":
    main()
