# Sanity check for the UnlearnCanvas classifiers: run the style and object
# classifiers (same model, weights and preprocessing as UCClassifiers in
# evaluate_unlearncanvas.py / UnlearnCanvas's accuracy.py) on real UnlearnCanvas
# dataset images, and on our sdxl-turbo unedited answer images for the same
# styles. If the style classifier is accurate on the dataset but near chance on
# sdxl-turbo, preprocessing is fine and the low CRA is sdxl-turbo not drawing
# UnlearnCanvas's styles.
#
# The dataset (huggingface.co/datasets/OPTML-Group/UnlearnCanvas) is 153
# parquet shards (~0.5 GB each, columns image + text); only --shards are
# downloaded. Style and object are read off each image's text.
#
#   python scripts/uc_classifier_on_dataset.py --shards 0 40 80 120 --per_style 10
#
# Writes {out}/per_image.csv and {out}/per_style.csv and prints the summary.
import argparse
import importlib.util
import io
import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default="OPTML-Group/UnlearnCanvas")
parser.add_argument("--shards", nargs="*", type=int, default=[0, 40, 80, 120],
                    help="which of the 153 parquet shards to download (each ~0.5 GB)")
parser.add_argument("--per_style", type=int, default=10, help="dataset images per style (cap)")
parser.add_argument("--style_ckpt", type=str, default="UnlearnCanvas/ckpts/cls_model/style50-001.pth")
parser.add_argument("--class_ckpt", type=str, default="UnlearnCanvas/ckpts/cls_model/style50_cls.pth")
parser.add_argument("--const", type=str, default="UnlearnCanvas/machine_unlearning/evaluation/constants/const.py")
parser.add_argument("--sdxl_dir", type=str, default="evaluation/uc/cache/answers/base",
                    help="our unedited sdxl-turbo answer images ({style}_{object}_seed{seed}.jpg); '' to skip")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--out", type=str, default="evaluation/uc_classifier_check")
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

spec = importlib.util.spec_from_file_location("uc_const", args.const)
const = importlib.util.module_from_spec(spec)
spec.loader.exec_module(const)
THEMES, CLASSES = list(const.theme_available), list(const.class_available)


# ---------------------------------------------------------------- classifiers (as UCClassifiers)

def load(ckpt, n):
    import timm
    m = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=False)
    m.head = torch.nn.Linear(1024, n)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    return m.to(device).eval()


from torchvision import transforms  # noqa: E402

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
style_model, class_model = load(args.style_ckpt, len(THEMES)), load(args.class_ckpt, len(CLASSES))


@torch.no_grad()
def classify(images):
    x = torch.stack([transform(im.convert("RGB")) for im in images]).to(device)
    return style_model(x).argmax(-1).tolist(), class_model(x).argmax(-1).tolist()


# ---------------------------------------------------------------- labels from the text

def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def variants(name):
    """A name, its spaced form and (for objects) singular forms, normalized."""
    base = norm(name)
    out = {base}
    for suffix in ["es", "s"]:
        if base.endswith(suffix) and len(base) > len(suffix) + 2:
            out.add(base[:-len(suffix)])
    return out


def find(text, names):
    """The longest label whose (normalized) name occurs in the text, else None."""
    t = norm(text)
    hits = [n for n in names if any(v in t for v in variants(n))]
    return max(hits, key=lambda n: len(norm(n))) if hits else None


# ---------------------------------------------------------------- dataset images

def dataset_rows():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(args.repo_id, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".parquet"))
    print(f"{len(files)} shards in {args.repo_id}; using {args.shards}")
    for i in args.shards:
        path = hf_hub_download(args.repo_id, files[i], repo_type="dataset")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=16, columns=["image", "text"]):
            for img, text in zip(batch.column("image").to_pylist(), batch.column("text").to_pylist()):
                yield files[i], img, text


os.makedirs(args.out, exist_ok=True)
rows, counts, shown, pending = [], {}, 0, []


def flush():
    if not pending:
        return
    sp, cp = classify([p["image"] for p in pending])
    for p, s, c in zip(pending, sp, cp):
        rows.append({"source": "dataset", "shard": p["shard"], "text": p["text"], "style": p["style"],
                     "object": p["object"], "style_pred": THEMES[s], "object_pred": CLASSES[c]})
    pending.clear()


for shard, img, text in dataset_rows():
    if shown < 5:
        print("  text example:", repr(text)[:160])
        shown += 1
    style, obj = find(text or "", THEMES), find(text or "", CLASSES)
    if style is None or counts.get(style, 0) >= args.per_style:
        continue
    counts[style] = counts.get(style, 0) + 1
    image = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else img
    pending.append({"shard": shard, "text": text, "style": style, "object": obj, "image": image})
    if len(pending) >= args.batch_size:
        flush()
flush()
print(f"dataset: {len(rows)} images over {len(counts)} styles")

# ---------------------------------------------------------------- our sdxl-turbo images, same styles

if args.sdxl_dir and os.path.isdir(args.sdxl_dir):
    pat = re.compile(r"^(?P<style>.+)_(?P<object>[A-Za-z]+)_seed(?P<seed>\d+)\.jpg$")
    picked = {}
    for f in sorted(os.listdir(args.sdxl_dir)):
        m = pat.match(f)
        if m and m["style"] in counts and len(picked.setdefault(m["style"], [])) < args.per_style:
            picked[m["style"]].append((f, m["object"]))
    for style, items in picked.items():
        for start in range(0, len(items), args.batch_size):
            part = items[start:start + args.batch_size]
            sp, cp = classify([Image.open(os.path.join(args.sdxl_dir, f)) for f, _ in part])
            for (f, obj), s, c in zip(part, sp, cp):
                rows.append({"source": "sdxl_turbo", "shard": "", "text": f, "style": style, "object": obj,
                             "style_pred": THEMES[s], "object_pred": CLASSES[c]})
    print(f"sdxl-turbo: {sum(len(v) for v in picked.values())} images over {len(picked)} styles")

# ---------------------------------------------------------------- results

df = pd.DataFrame(rows)
df["style_ok"] = (df["style_pred"] == df["style"]).astype(float)
df["object_ok"] = np.where(df["object"].notna(), (df["object_pred"] == df["object"]).astype(float), np.nan)
df.to_csv(os.path.join(args.out, "per_image.csv"), index=False)
per_style = df.pivot_table(index="style", columns="source", values=["style_ok", "object_ok"], aggfunc="mean")
per_style.to_csv(os.path.join(args.out, "per_style.csv"))
print("\nper style (accuracy):")
print(per_style.round(2).to_string())
print("\noverall:")
print(df.groupby("source")[["style_ok", "object_ok"]].mean().round(3).to_string())
print(f"(chance: style {1 / len(THEMES):.3f}, object {1 / len(CLASSES):.3f})")
unparsed = df["object"].isna().sum()
if unparsed:
    print(f"! {unparsed} dataset images had no object name in their text (object_ok left blank)")
print(f"-> {args.out}/per_image.csv, {args.out}/per_style.csv")
