# Head-to-head check of the two UnlearnCanvas classifier pairs on the same
# fresh sdxl-turbo images: the released ViT-L/16 style (51-way) and object
# (20-way) heads, and the ones retrained on sdxl-turbo images
# (train_uc_classifier_sdxl.py). Same model, weights and preprocessing as
# UCClassifiers in evaluate_unlearncanvas.py.
#
# Stage 1 draws --n_images distinct style x object pairs at random (the 50 real
# styles - Seed_Images, the unstylized class, is left out - x the 20 objects),
# and generates one image per pair from UnlearnCanvas's prompt
# ("A {object} image in {style} style.") on seed --seed_offset + j - apart from
# the sdxl classifier's training seeds (100000 + i) and the answer set's
# (188, 288, ...), so neither pair has seen these images. Existing images are
# reused. Stage 2 classifies every image with both pairs (one on the GPU at a
# time) and compares them, per head: each pair's accuracy, how often their
# argmaxes agree, and the 2x2 of right/wrong (both right, only one right, both
# wrong), plus per-style / per-object accuracy.
#
#   python compare_uc_classifiers.py                       # 100 images
#   python compare_uc_classifiers.py --n_images 300 --seed 1
#
# Writes {out}/per_image.csv, {out}/summary.csv, {out}/per_label.csv and prints them.
import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--n_images", type=int, default=100, help="distinct style x object pairs, one image each")
parser.add_argument("--seed", type=int, default=0, help="picks the pairs")
parser.add_argument("--seed_offset", type=int, default=300000, help="image j uses seed seed_offset + j")
parser.add_argument("--template", type=str, default="A {object} image in {style} style.")
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--pretrained_style_ckpt", type=str, default="UnlearnCanvas/ckpts/cls_model/style50-001.pth")
parser.add_argument("--pretrained_class_ckpt", type=str, default="UnlearnCanvas/ckpts/cls_model/style50_cls.pth")
parser.add_argument("--sdxl_style_ckpt", type=str, default="UnlearnCanvas/ckpts/cls_model_sdxl/style_sdxl.pth")
parser.add_argument("--sdxl_class_ckpt", type=str, default="UnlearnCanvas/ckpts/cls_model_sdxl/class_sdxl.pth")
parser.add_argument("--const", type=str, default="UnlearnCanvas/machine_unlearning/evaluation/constants/const.py")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--out", type=str, default="evaluation/uc_classifier_compare")
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

spec = importlib.util.spec_from_file_location("uc_const", args.const)
const = importlib.util.module_from_spec(spec)
spec.loader.exec_module(const)
THEMES, CLASSES = list(const.theme_available), list(const.class_available)  # label order of the heads
STYLES = [t for t in THEMES if t != "Seed_Images"]
PAIRS = {"pretrained": (args.pretrained_style_ckpt, args.pretrained_class_ckpt),
         "sdxl": (args.sdxl_style_ckpt, args.sdxl_class_ckpt)}
HEADS = {"style": "style", "object": "class"}  # head -> prediction column prefix

image_dir = os.path.join(args.out, "images")
os.makedirs(image_dir, exist_ok=True)


def words(name):
    return name.replace("_", " ")


# ---------------------------------------------------------------- stage 1: images

rng = np.random.default_rng(args.seed)
grid = [(s, o) for s in STYLES for o in CLASSES]
if args.n_images > len(grid):
    raise ValueError(f"--n_images {args.n_images} > {len(grid)} style x object pairs")
picked = [grid[i] for i in rng.choice(len(grid), args.n_images, replace=False)]
items = [{"style": s, "object": o, "seed": args.seed_offset + j,
          "image": os.path.join(image_dir, f"{s}_{o}_seed{args.seed_offset + j}.jpg")}
         for j, (s, o) in enumerate(picked)]

todo = [it for it in items if not os.path.exists(it["image"])]
print(f"generate: {len(todo)} of {len(items)} images to make "
      f"({len({it['style'] for it in items})} styles, {len({it['object'] for it in items})} objects)")
if todo:
    from diffusers import AutoPipelineForText2Image
    fp16 = device == "cuda"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16 if fp16 else torch.float32,
        variant="fp16" if fp16 else None).to(device)
    pipe.set_progress_bar_config(disable=True)
    start = time.time()
    with torch.no_grad():
        for n, it in enumerate(todo, 1):
            im = pipe(prompt=args.template.format(object=words(it["object"]), style=words(it["style"])),
                      height=args.size, width=args.size, num_inference_steps=args.num_inference_steps,
                      guidance_scale=args.guidance_scale,
                      generator=torch.Generator().manual_seed(it["seed"])).images[0]
            im.save(it["image"])
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({time.time() - start:.0f}s)")
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

# ---------------------------------------------------------------- stage 2: classify with both pairs

from torchvision import transforms  # noqa: E402

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


def load(ckpt, n):
    import timm
    # pretrained=False: every weight, head included, comes from the checkpoint
    m = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=False)
    m.head = torch.nn.Linear(1024, n)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    return m.to(device).eval()


rows = [{k: it[k] for k in ["image", "style", "object", "seed"]} for it in items]
for name, (style_ckpt, class_ckpt) in PAIRS.items():
    for head, field, ckpt, labels, label_key in [("style", "style", style_ckpt, THEMES, "style"),
                                                 ("object", "class", class_ckpt, CLASSES, "object")]:
        model = load(ckpt, len(labels))
        for b in range(0, len(rows), args.batch_size):
            part = rows[b:b + args.batch_size]
            x = torch.stack([transform(Image.open(r["image"]).convert("RGB")) for r in part]).to(device)
            with torch.no_grad():
                probs = F.softmax(model(x).float(), dim=-1).cpu().numpy()
            for r, p in zip(part, probs):
                r[f"{head}_pred_{name}"] = labels[int(p.argmax())]
                r[f"{head}_conf_{name}"] = float(p.max())
                r[f"{head}_p_label_{name}"] = float(p[labels.index(r[label_key])])
        model.to("cpu")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"classified {len(rows)} images: {name} {head} head ({ckpt})")

# ---------------------------------------------------------------- results

df = pd.DataFrame(rows)
for head in HEADS:
    for name in PAIRS:
        df[f"{head}_ok_{name}"] = df[f"{head}_pred_{name}"] == df[head]
    df[f"{head}_agree"] = df[f"{head}_pred_pretrained"] == df[f"{head}_pred_sdxl"]
df.to_csv(os.path.join(args.out, "per_image.csv"), index=False)

summary = []
for head in HEADS:
    a, b = df[f"{head}_ok_pretrained"], df[f"{head}_ok_sdxl"]
    agree = df[f"{head}_agree"]
    summary.append({"head": head, "n": len(df),
                    "acc pretrained": a.mean(), "acc sdxl": b.mean(),
                    "mean p(label) pretrained": df[f"{head}_p_label_pretrained"].mean(),
                    "mean p(label) sdxl": df[f"{head}_p_label_sdxl"].mean(),
                    "agree": agree.mean(),
                    "both right": (a & b).mean(), "only pretrained": (a & ~b).mean(),
                    "only sdxl": (~a & b).mean(), "both wrong": (~a & ~b).mean(),
                    "both wrong, same label": (~a & ~b & agree).mean()})
summary = pd.DataFrame(summary).set_index("head")
summary.to_csv(os.path.join(args.out, "summary.csv"))

per_label = pd.concat([
    df.groupby(head).agg(n=(head, "size"),
                         **{f"acc {name}": (f"{head}_ok_{name}", "mean") for name in PAIRS},
                         agree=(f"{head}_agree", "mean"))
    .assign(head=head).rename_axis("label").reset_index()
    for head in HEADS]).set_index(["head", "label"])
per_label["Δ acc"] = per_label["acc sdxl"] - per_label["acc pretrained"]
per_label = per_label.sort_values(["head", "Δ acc"], ascending=[True, False])
per_label.to_csv(os.path.join(args.out, "per_label.csv"))

pd.set_option("display.width", 250)
print("\nsummary (shares of the images):")
print(summary.round(3).T.to_string())
print(f"(chance: style {1 / len(THEMES):.3f}, object {1 / len(CLASSES):.3f})")
print("\nper label (sorted by Δ acc = sdxl - pretrained):")
print(per_label.round(2).to_string())
for head in HEADS:
    wrong = df[~df[f"{head}_ok_pretrained"]][f"{head}_pred_pretrained"].value_counts().head(5)
    print(f"\npretrained {head} head, most common wrong predictions: "
          + ", ".join(f"{k} ({v})" for k, v in wrong.items()))
print(f"-> {args.out}/per_image.csv, {args.out}/summary.csv, {args.out}/per_label.csv")
