# One figure per sampled edit: the same concept + prompt + seed under every method x strength setting of the paper's
# tables (experiments.tex), at --block (CASL-Steer 5k at --casl_block). Rows = the strength settings, columns =
# ESSI, Adaptive, OSP, SAeUron, CASL, with the unedited image on the left. Each image is captioned with the
# strength it used (gamma for removal, alpha for insertion).
#   insertion  --n random (concept, base image) pairs, the base's own subject skipped: the base prompt
#              ("a good picture of {b}", its manifest seed) with the concept inserted; rows alpha_auto,
#              alpha = -gamma_auto, alpha = -gamma_search
#   removal    --n random (concept, style) pairs: "A {concept} image in {style} style." at --seed with the concept
#              removed; rows gamma_auto, gamma_search
# The images are found from the results tables' rows (out_dir, method, kind, k_metric, gamma / strength):
#   removal    {out_dir}/answers/{method}/{kind}_{k_metric}/{concept}/{block}/[a{gamma}/ | g{gamma}/]{style}_{concept}_seed{seed}.jpg
#   insertion  {out_dir}/{--inject_root}/{method}/{kind}_{k_metric}/{concept}/{block}/s{alpha}/{base}.jpg
# Writes {out}/insertion/NN_{concept}_{base}.png and {out}/removal/NN_{concept}_{style}.png.
# Run on the cluster from the repo root:
#   python edit_grids.py
#   python edit_grids.py --n 10 --sample_seed 1 --casl_block mid
import argparse
import glob
import json
import os
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10, help="figures per task")
parser.add_argument("--sample_seed", type=int, default=0, help="seed of the random choice of concept / prompt")
parser.add_argument("--seed", type=int, default=188, help="the evaluation prompts' generation seed")
parser.add_argument("--block", type=str, default="down_blocks.2.attentions.1")
parser.add_argument("--casl_block", choices=["down", "mid"], default="down", help="CASL (down) or CASL (mid)")
parser.add_argument("--target", type=float, default=0.95, help="scripts/uc_priority.sh's TARGET (auto alpha)")
parser.add_argument("--run", type=str, default="evaluation/uc_auto")
parser.add_argument("--removal_csvs", nargs="*", default=["evaluation/outputs_sdxlcls/uc_results.csv",
                                                          "evaluation/outputs_sdxlcls_ucbase/uc_results.csv"])
parser.add_argument("--inject_csv", type=str, default="evaluation/outputs_sdxlcls_ucbase/uc_inject_results.csv")
parser.add_argument("--cache_dir", type=str, default="evaluation/uc/cache")
parser.add_argument("--base_cache_dir", type=str, default="evaluation/uc/cache_ucbase")
parser.add_argument("--inject_root", type=str, default="inject_uc")
parser.add_argument("--size", type=float, default=2.0, help="inches per image")
parser.add_argument("--out", type=str, default="edit_grids")
args = parser.parse_args()

BLOCK_DIR = args.block.replace(".", "_")
CASL_BLOCK = {"down": "down_blocks.2.attentions.1", "mid": "mid_block.attentions.0"}[args.casl_block]
AUTOK = {"ESSI": "saeuron_masked", "Adaptive": "adaptive", "OSP": "osp"}  # name -> --rules (= the rows' kind)
COLUMNS = [*AUTOK, "SAeUron", f"CASL ({args.casl_block})"]
AUTO = f"_sauto{args.target:g}"


def casl_dir(suffix):
    """CASL-Steer 5k (Adam lr 1e-2): suffix _autoa (auto gamma) or _asearch (SAeUron's alpha search)."""
    name = f"{args.run}/casl_steer_all_paper_adam_lr1e-2_s5000"
    if CASL_BLOCK != "mid_block.attentions.0":
        name += "_" + CASL_BLOCK.replace(".", "_")
    return name + suffix


def saeuron_dir(auto=False):
    return f"{args.run}/saeuron_all_searched_toptau{'_autog' if auto else ''}_{BLOCK_DIR}"


def autok_dir(rule, searched=False):
    return f"{args.run}/{rule}_all_autok_accuracy_{'gsearchpre' if searched else 'autog'}"


def is_(d, kind, metric=None, suffix=None):
    """Filter of the rows of one setting: its out_dir and kind, and its k_metric (exact, or by suffix)."""
    def f(t):
        ok = (t["out_dir"] == d) & (t["kind"] == kind)
        if metric is not None:
            ok &= t["k_metric"] == metric
        if suffix is not None:
            ok &= t["k_metric"].astype(str).str.endswith(suffix)
        return ok
    return f


# (row label, column) -> filter, as uc_priority_comparison.ipynb's settings
REMOVAL = {}
for name, rule in AUTOK.items():
    REMOVAL[("γ_auto", name)] = is_(autok_dir(rule), rule, "accuracy")
    REMOVAL[("γ_search", name)] = is_(autok_dir(rule, searched=True), rule, "accuracy")
REMOVAL[("γ_auto", "SAeUron")] = is_(saeuron_dir(auto=True), "saeuron")
REMOVAL[("γ_search", "SAeUron")] = is_(saeuron_dir(), "saeuron")
REMOVAL[("γ_auto", COLUMNS[-1])] = is_(casl_dir("_autoa"), "casl_steer")
REMOVAL[("γ_search", COLUMNS[-1])] = is_(casl_dir("_asearch"), "casl_steer")

INSERTION = {}
for name, rule in AUTOK.items():
    INSERTION[("α_auto", name)] = is_(autok_dir(rule), rule, f"accuracy{AUTO}")
    INSERTION[("α = −γ_auto", name)] = is_(autok_dir(rule), rule, "accuracy_snegg")
    INSERTION[("α = −γ_search", name)] = is_(autok_dir(rule, searched=True), rule, "accuracy_snegg")
INSERTION[("α_auto", "SAeUron")] = is_(saeuron_dir(), "saeuron", suffix=AUTO)
INSERTION[("α = −γ_auto", "SAeUron")] = is_(saeuron_dir(auto=True), "saeuron", suffix="_snegg")
INSERTION[("α = −γ_search", "SAeUron")] = is_(saeuron_dir(), "saeuron", suffix="_snegg")
INSERTION[("α_auto", COLUMNS[-1])] = is_(casl_dir("_autoa"), "casl_steer", f"top1{AUTO}")
INSERTION[("α = −γ_auto", COLUMNS[-1])] = is_(casl_dir("_autoa"), "casl_steer", "top1")
INSERTION[("α = −γ_search", COLUMNS[-1])] = is_(casl_dir("_asearch"), "casl_steer")


def load(paths):
    tables = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    for p in paths:
        if not os.path.exists(p):
            print(f"! {p} missing")
    t = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return t[t["kind"] != "base"]


def setting_rows(table, settings):
    """(row label, column) -> {concept: that setting's row at its block}."""
    out = {}
    for (row, col), f in settings.items():
        block = CASL_BLOCK if col == COLUMNS[-1] else args.block
        t = table[(table["block"] == block)]
        t = t[f(t)].drop_duplicates("subject", keep="first")
        out[(row, col)] = {r["subject"]: r for _, r in t.iterrows()}
        if not out[(row, col)]:
            print(f"! no rows for {row} / {col}")
    return out


def find(folder, name, value):
    """{folder}/{name} or {folder}/{x}{number}/{name}; several (a sweep): the number closest to value."""
    found = glob.glob(os.path.join(glob.escape(folder), name)) + glob.glob(os.path.join(glob.escape(folder), "*", name))
    if len(found) <= 1 or value is None or pd.isna(value):
        return found[0] if found else None

    def number(p):
        m = re.fullmatch(r"[a-z]+(-?[\d.e+-]+)", os.path.basename(os.path.dirname(p)))
        return float(m.group(1)) if m else np.inf
    return min(found, key=lambda p: abs(number(p) - float(value)))


def edit_folder(r, root, block):
    return os.path.join(r["out_dir"], root, r["method"], f"{r['kind']}_{r['k_metric']}",
                        r["subject"].strip().replace(" ", "_").replace("/", "_"), block.replace(".", "_"))


def cell(ax, path, caption=None, title=None, ylabel=None):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if path and os.path.exists(path):
        ax.imshow(Image.open(path))
    else:
        ax.set_facecolor("#eeeeee")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, transform=ax.transAxes)
    if caption:
        ax.set_xlabel(caption, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def figure(unedited, grid, rows, title, path):
    """grid: (row label, column) -> (image path, caption); the unedited image top left."""
    fig, axes = plt.subplots(len(rows), len(COLUMNS) + 1, squeeze=False,
                             figsize=(args.size * (len(COLUMNS) + 1), args.size * 1.15 * len(rows)))
    cell(axes[0, 0], unedited, title="unedited")
    for ax in axes[1:, 0]:
        ax.axis("off")
    for i, row in enumerate(rows):
        for j, col in enumerate(COLUMNS):
            img, caption = grid[(row, col)]
            cell(axes[i, j + 1], img, caption, title=col if i == 0 else None, ylabel=row if j == 0 else None)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{path}: {sum(p is not None and os.path.exists(p) for p, _ in grid.values())}/{len(grid)} edits found")


rng = random.Random(args.sample_seed)

# ---------------- insertion
rows = ["α_auto", "α = −γ_auto", "α = −γ_search"]
inject = setting_rows(load([args.inject_csv]), INSERTION)
manifest = json.load(open(os.path.join(args.base_cache_dir, "base", "manifest.json")))
bases = [e for e in manifest if e.get("mask_area", 0) > 0]
concepts = sorted(set.intersection(*[set(v) for v in inject.values()]) or set().union(*[set(v) for v in inject.values()]))
pairs = [(c, e) for c in concepts for e in bases if e["subject"].strip().lower() != c.strip().lower()]
for n, (concept, e) in enumerate(rng.sample(pairs, min(args.n, len(pairs)))):
    grid = {}
    for row in rows:
        for col in COLUMNS:
            r = inject[(row, col)].get(concept)
            if r is None:
                grid[(row, col)] = (None, None)
                continue
            block = CASL_BLOCK if col == COLUMNS[-1] else args.block
            path = find(edit_folder(r, args.inject_root, block), f"{e['name']}.jpg", r["strength"])
            grid[(row, col)] = (path, f"α = {float(r['strength']):.3g}")
    figure(e["image"], grid, rows, f"Inserting {concept} into \"{e['prompt']}\" (seed {e['seed']})",
           os.path.join(args.out, "insertion", f"{n:02d}_{concept}_{e['name']}.png"))

# ---------------- removal
rows = ["γ_auto", "γ_search"]
removal = setting_rows(load(args.removal_csvs), REMOVAL)
concepts = sorted(set.intersection(*[set(v) for v in removal.values()]) or set().union(*[set(v) for v in removal.values()]))
base_dir = os.path.join(args.cache_dir, "answers", "base")
suffix = lambda c: f"_{c}_seed{args.seed}.jpg"
pairs = [(c, os.path.basename(p)[:-len(suffix(c))]) for c in concepts
         for p in sorted(glob.glob(os.path.join(base_dir, f"*{suffix(c)}")))]
for n, (concept, style) in enumerate(rng.sample(pairs, min(args.n, len(pairs)))):
    name = f"{style}{suffix(concept)}"
    grid = {}
    for row in rows:
        for col in COLUMNS:
            r = removal[(row, col)].get(concept)
            if r is None:
                grid[(row, col)] = (None, None)
                continue
            block = CASL_BLOCK if col == COLUMNS[-1] else args.block
            path = find(edit_folder(r, "answers", block), name, r["gamma"])
            grid[(row, col)] = (path, f"γ = {float(r['gamma']):.3g}" if pd.notna(r["gamma"]) else None)
    figure(os.path.join(base_dir, name), grid, rows,
           f"Removing {concept}: \"A {concept} image in {style.replace('_', ' ')} style.\" (seed {args.seed})",
           os.path.join(args.out, "removal", f"{n:02d}_{concept}_{style}.png"))
print(f"-> {args.out}/insertion, {args.out}/removal")
