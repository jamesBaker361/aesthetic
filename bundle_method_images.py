# Copies a few same-prompt images of several methods into one small folder (repo-relative paths kept, so the
# notebook reads it like the repo), instead of downloading evaluation/. For uc_method_images.ipynb.
#   removal    for every --concepts x --styles x --seed benchmark prompt ("A {object} image in {style} style."):
#              the unedited image ({cache_dir}/answers/base/) and each method's edit of it at every block the run
#              has ({run}/answers/{mask}/{rule}_{metric}/{concept}/{block}/{style}_{concept}_seed{seed}.jpg) +
#              the folder's latents.json - the notebook picks the block each method uses (as uc_method_comparison)
#   injection  for every --concepts x --bases base image ("a good picture of a cube" ...): the unedited base
#              ({cache_dir}/base/images/) + each method's injection at --strength in any of its inject folders
#              ({run}/inject*/{mask}/{rule}_{metric}/{concept}/{block}/s{strength}/{base}.jpg), and base/manifest.json;
#              the fixed top-k sets ({rule}_top{k}/, from --inject_top_k runs - scripts/uc_inject_topk_more.sh) too
#   CASL-Steer every alpha it was run with: removal one folder deeper ({block}/a-{alpha}/, a{alpha}/ for the older
#              reversed-pair maps), injection at s{alpha}/ instead of --strength
#   captions   every copied image's cached UnlearnCanvas classifier predictions ({image}.uc.image.json, and the
#              sdxl-trained pair's {image}.uc_{hash}.image.json - uc_classifier_comparison.ipynb), if scored
# Writes {out}/... and {out}_bundle.tar.gz holding only what this run copied (the folder may hold other files).
# Run on the cluster from the repo root:
#   python bundle_method_images.py                       # defaults: the four methods below
#   python bundle_method_images.py --concepts Dogs Trees --styles Monet Pop_Art
#   python bundle_method_images.py --add --methods "SAeUron · searched (top-feature block)"   # just one more method
#   python bundle_method_images.py --methods "SAeUron · searched" "SAeUron score (masked) · per-block · bal-acc" CASL-Steer
#   python bundle_method_images.py --add --methods saeuron_all_searched_toptau_down_blocks_2_attentions_1 \
#       saeuron_masked_all_autok_accuracy_autog casl_steer_all_paper_sgd_lr1e-1_s1000   # run folders: through slurm
# then download and untar images_bundle.tar.gz in the repo root and open uc_method_images.ipynb.
import argparse
import glob
import json
import os
import re
import shutil
import tarfile

# label -> run folder; the same labels as uc_method_comparison.ipynb. Per-block runs: every block is copied and
# uc_method_images.ipynb shows the one with the fewest auto-k latents (ties: auto-k probe score), as the comparison
METHODS = {
    "SAeUron score (masked) · per-block · bal-acc": "evaluation/uc_auto/saeuron_masked_all_autok_balanced_accuracy_autog",
    "bce · per-block · acc": "evaluation/uc_auto/bce_all_autok_accuracy_autog",          # block: min k (notebook)
    "bce · per-block · bal-acc": "evaluation/uc_auto/bce_all_autok_balanced_accuracy_autog",
    "bce · joint · acc": "evaluation/uc_auto/bce_alljoint_autok_accuracy_autog",
    "bce · joint · bal-acc": "evaluation/uc_auto/bce_alljoint_autok_balanced_accuracy_autog",
    "SAeUron · searched": "evaluation/uc_auto/saeuron_all_searched_toptau",                 # scripts/uc_saeuron_searched.sh
    "SAeUron · searched (top-feature block)": "evaluation/uc_auto/saeuron_all_searched_toptau_topfeature",
    # fixed-block rows of uc_method_comparison ("{label} @ {block}"): the notebook shows the block in the label
    "SAeUron · searched @ down.2.1": "evaluation/uc_auto/saeuron_all_searched_toptau_down_blocks_2_attentions_1",
    "SAeUron score (masked) @ down.2.1 · acc": "evaluation/uc_auto/saeuron_masked_all_autok_accuracy_autog",
    # CASL-Steer (scripts/uc_casl_steer_all.sh): every alpha - removal in {block}/a-{alpha}/, injection in s{alpha}/
    "CASL-Steer": "evaluation/uc_auto/casl_steer_all_paper_sgd_lr1e-1_s1000",
}

parser = argparse.ArgumentParser()
parser.add_argument("--methods", nargs="*", default=list(METHODS), help="labels of METHODS to copy")
parser.add_argument("--extra_runs", nargs="*", default=[], help="more run folders to copy (label = folder name)")
parser.add_argument("--concepts", nargs="*", default=["Cats", "Bears", "Flowers", "Towers"])
parser.add_argument("--styles", nargs="*", default=["Van_Gogh", "Cartoon", "Pencil_Drawing"])
parser.add_argument("--seed", type=int, default=188)
parser.add_argument("--bases", nargs="*", default=None,
                    help="injection base image names (default: --n_bases spread over base/manifest.json)")
parser.add_argument("--n_bases", type=int, default=4)
parser.add_argument("--strength", type=float, default=10.0)
parser.add_argument("--cache_dir", type=str, default="evaluation/uc/cache")
parser.add_argument("--out", type=str, default="images")
parser.add_argument("--add", action="store_true",
                    help="add --methods to the bundle already in {out}: same concepts / styles / bases / strength as it "
                         "(read from its bundle_info.json), methods merged into it, and the tarball holds only the new "
                         "files - untar it over the images you already have")
args = parser.parse_args()
previous = None
if args.add:
    path = os.path.join(args.out, "bundle_info.json")
    assert os.path.exists(path), f"--add needs an existing bundle in {args.out}/ (no {path})"
    previous = json.load(open(path))
    args.concepts, args.styles, args.seed = previous["concepts"], previous["styles"], previous["seed"]
    args.bases, args.strength, args.cache_dir = previous["bases"], previous["strength"], previous["cache_dir"]

# a method is its label or its run folder's name (no spaces: the slurm runner drops the quotes around a label)
BY_FOLDER = {os.path.basename(run): label for label, run in METHODS.items()}
unknown = [m for m in args.methods if m not in METHODS and m not in BY_FOLDER]
assert not unknown, f"unknown --methods {unknown}; use a label or a run folder name: {sorted(BY_FOLDER)}"
runs = {BY_FOLDER.get(m, m): METHODS[BY_FOLDER.get(m, m)] for m in args.methods}
runs.update({os.path.basename(r.rstrip("/")): r for r in args.extra_runs})
copied = []


def copy(path):
    '''Copy one file to {out}/{path} (and its cached classifier predictions - every pair's, uc / uc_{hash} - if any).'''
    for p in [path] + glob.glob(f"{glob.escape(path)}.uc*.image.json"):
        if os.path.exists(p):
            dest = os.path.join(args.out, p)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(p, dest)
            copied.append(dest)


# --- removal: same benchmark prompts for every method
files = [f"{s}_{c}_seed{args.seed}.jpg" for c in args.concepts for s in args.styles]
for name in files if not args.add else []:  # --add: the unedited images are already in the bundle
    path = os.path.join(args.cache_dir, "answers", "base", name)
    copy(path) if os.path.exists(path) else print(f"! no unedited image {path}")
for label, run in runs.items():
    n = 0
    for concept in args.concepts:
        for style in args.styles:
            name = f"{style}_{concept}_seed{args.seed}.jpg"
            # {block}/{name}, or {block}/a{alpha}/{name} for CASL-Steer
            for path in glob.glob(os.path.join(run, "answers", "*", "*", concept, "*", name)) + \
                    glob.glob(os.path.join(run, "answers", "*", "casl_steer_*", concept, "*", "a*", name)):
                copy(path)
                copy(os.path.join(os.path.dirname(path), "latents.json"))
                n += 1
    print(f"removal   {label}: {n} images" + ("" if n else f"  ! none under {run}/answers"))

# --- injection: same base images for every method
manifest_path = os.path.join(args.cache_dir, "base", "manifest.json")
manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
if not args.add:
    copy(manifest_path)
usable = [e for e in manifest if e.get("mask_area", 1) > 0]
bases = args.bases or [e["name"] for e in usable[::max(1, len(usable) // args.n_bases)][:args.n_bases]]
for e in manifest:
    if e["name"] in bases and not args.add:
        copy(e["image"])
copied_sources = set()
for label, run in runs.items():
    n, top = 0, {}
    for concept in args.concepts:
        for base in bases:
            for path in glob.glob(os.path.join(run, "inject*", "*", "*", concept, "*", f"s{args.strength:g}", f"{base}.jpg")) + \
                    glob.glob(os.path.join(run, "inject*", "*", "casl_steer_*", concept, "*", "s*", f"{base}.jpg")):
                if path in copied_sources:
                    continue  # CASL-Steer at alpha = --strength matches both globs
                copied_sources.add(path)
                copy(path)
                if path.split(os.sep)[-5].startswith("casl_steer_"):  # its fixed top-k is the method, not a top-k set
                    n += 1
                    continue
                k = re.search(r"_top(\d+)$", path.split(os.sep)[-5])  # kind folder: {rule}_{metric} or {rule}_top{k}
                if k:
                    top[int(k.group(1))] = top.get(int(k.group(1)), 0) + 1
                else:
                    n += 1
    print(f"injection {label}: {n} auto-k images, "
          + (", ".join(f"top{k}: {c}" for k, c in sorted(top.items())) if top else "no top-k images")
          + ("" if n else f"  ! none under {run}/inject*"))

# what the notebook needs to know about this bundle
info = {"methods": {**(previous["methods"] if previous else {}), **runs}, "concepts": args.concepts, "styles": args.styles, "seed": args.seed, "bases": bases,
        "strength": args.strength, "cache_dir": args.cache_dir}
info_path = os.path.join(args.out, "bundle_info.json")
os.makedirs(args.out, exist_ok=True)
json.dump(info, open(info_path, "w"), indent=1)
copied.append(info_path)

with tarfile.open(f"{args.out}_bundle.tar.gz", "w:gz") as tar:
    for dest in sorted(set(copied)):
        tar.add(dest)
print(f"{len(set(copied))} files -> {args.out}/ and {args.out}_bundle.tar.gz (only these files)")
