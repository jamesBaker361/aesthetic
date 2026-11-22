# Collects casl_faces.py's results.csv under --root. Per config x snapshot, the means over its attributes (k = 1) of:
# EPR at each attribute's best alpha (the paper picks alpha per concept, App. 7.10) and over every alpha, the signed
# target change, dCLIP and |target change| at that alpha, dL_clip, and the "Recon." EPR_real. -> {root}/summary.csv,
# and the --top configs by best-alpha EPR (among those with every stage-1 attribute) -> {root}/best_configs.txt,
# "{run} {snapshot}" per line, for scripts/casl_faces_sweep.sh STAGE=2 (edit the file to pick others). EPR is a
# ratio, so an edit that barely changes anything can still score well: check d_target next to it.
# Run: python casl_faces_summary.py
import argparse
import glob
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="evaluation/casl_faces")
parser.add_argument("--top", type=int, default=4)
parser.add_argument("--min_attrs", type=int, default=2, help="configs scored on fewer attributes are not picked")

CONFIG = ["model", "start", "t_start", "n_steps", "t_edit", "block", "optimizer", "lr", "lambda_clip", "lambda_recon",
          "ref"]


def main(args):
    paths = sorted(glob.glob(os.path.join(args.root, "*", "results.csv")))
    assert paths, f"no results.csv under {args.root}"
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    keys = ["run", "step"]
    edits = df[(df["k"] == 1) & (df["alpha"] > 0)]
    best = edits.loc[edits.groupby(keys + ["attribute"])["EPR"].idxmax()]
    per = best.groupby(keys).agg(EPR_best=("EPR", "mean"), d_target=("d_target", "mean"),
                                 d_target_signed=("d_target_signed", "mean"), d_clip=("d_clip", "mean"),
                                 alpha_best=("alpha", "median"), dL_clip=("dL_clip", "mean"),
                                 n_attrs=("attribute", "nunique"))
    every = edits.groupby(keys).agg(EPR_mean=("EPR", "mean"), d_clip_mean=("d_clip", "mean"),
                                    d_clip_max=("d_clip", "max"), fire=("fire", "mean"))
    recon = df[df["k"] == 0].groupby(keys).agg(EPR_recon=("EPR_real", "mean"))
    config = df.drop_duplicates("run").set_index("run")[CONFIG]
    out = per.join(every).join(recon).reset_index().join(config, on="run")
    out = out.sort_values("EPR_best", ascending=False)
    out.to_csv(os.path.join(args.root, "summary.csv"), index=False)
    with pd.option_context("display.max_rows", 500, "display.width", 250, "display.float_format", "{:.3f}".format):
        print(out[keys + ["EPR_best", "EPR_mean", "d_target", "d_target_signed", "d_clip", "d_clip_max", "dL_clip",
                          "alpha_best", "fire", "EPR_recon", "n_attrs"]].to_string(index=False))
    top = out[out["n_attrs"] >= args.min_attrs].head(args.top)
    with open(os.path.join(args.root, "best_configs.txt"), "w") as f:
        f.writelines(f"{r.run} {int(r.step)}\n" for r in top.itertuples())
    print(f"-> {os.path.join(args.root, 'best_configs.txt')}:")
    print(top[keys + ["EPR_best", "d_clip"]].to_string(index=False))


if __name__ == "__main__":
    main(parser.parse_args())
