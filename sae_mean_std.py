# print the standard deviation of each SAE checkpoint's mean.pt (the
# per-feature mean vector load_feature_mean in generate_clean_swap.py reads),
# alongside a few other summary stats, plus the std over all blocks pooled.
# Only "local" checkpoints - SAeUron ones have no mean.pt.

import os
import argparse

import torch

SAE_CHECKPOINTS = "./sdxl_unbox/checkpoints/"

# same as attribution.DEFAULT_BLOCK_LIST, copied to avoid its heavy imports
DEFAULT_BLOCK_LIST = [
    "down_blocks.2.attentions.1",
    "mid_block.attentions.0",
    "up_blocks.0.attentions.0",
    "up_blocks.0.attentions.1",
]

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, default=SAE_CHECKPOINTS)
parser.add_argument("--block_list", type=str, nargs="+", default=DEFAULT_BLOCK_LIST)
parser.add_argument("--file", type=str, default="mean.pt", help="vector file inside each checkpoint's final/ dir")


def load_mean_vec(checkpoint_dir: str, block: str, file: str) -> torch.Tensor:
    path = os.path.join(checkpoint_dir, f"unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final", file)
    return torch.load(path, weights_only=True, map_location="cpu").float().flatten()


def main(args):
    all_vecs = []
    for block in args.block_list:
        vec = load_mean_vec(args.checkpoint_dir, block, args.file)
        if torch.isnan(vec).any():
            print(f"{block}: {int(torch.isnan(vec).sum())} nan entries, ignored below")
            vec = vec[~torch.isnan(vec)]
        all_vecs.append(vec)
        print(f"{block}: n={vec.numel()} std={vec.std().item():.6f} mean={vec.mean().item():.6f} "
              f"min={vec.min().item():.6f} max={vec.max().item():.6f} norm={vec.norm().item():.6f}")

    pooled = torch.cat(all_vecs)
    print(f"all blocks pooled: n={pooled.numel()} std={pooled.std().item():.6f} mean={pooled.mean().item():.6f}")


if __name__ == "__main__":
    main(parser.parse_args())
