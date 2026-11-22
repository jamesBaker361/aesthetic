# CelebA-HQ with its 40 CelebA attributes, for casl_faces.py and train_celeba_classifier.py: CelebAMask-HQ's
# CelebA-HQ-img/{idx}.jpg (1024 px), CelebAMask-HQ-attribute-anno.txt (the 40 attributes, +1 / -1) and
# CelebA-HQ-to-CelebA-mapping.txt (each image's CelebA index -> the official CelebA split, as the CASL paper's
# "official train and test partitions", App. 7.7: CelebA index < 162,770 train, < 182,637 val, else test).
# Source: --src_dir (an unpacked CelebAMask-HQ), else kagglehub.dataset_download(--kaggle_handle).
# -> {out_dir}/{size}/{idx}.jpg (resized, JPEG q95) and {out_dir}/attrs.csv: idx, file (relative to out_dir),
# split, and one 0 / 1 column per attribute in the annotation file's order.
# Run: python prepare_celebahq.py   (CPU is enough: sbatch runpycpu_chip.sh prepare_celebahq.py)
import argparse
import os

import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str, default=None, help="an unpacked CelebAMask-HQ (else --kaggle_handle)")
parser.add_argument("--kaggle_handle", type=str, default="ipythonx/celebamaskhq")
parser.add_argument("--out_dir", type=str, default="data/celebahq")
parser.add_argument("--size", type=int, default=512)

CELEBA_TRAIN_END, CELEBA_VAL_END = 162770, 182637  # list_eval_partition.txt: 0-based CelebA indices


def find(root: str, name: str, is_dir: bool = False) -> str:
    for d, dirs, files in os.walk(root):
        if name in (dirs if is_dir else files):
            return os.path.join(d, name)
    raise FileNotFoundError(f"no {name} under {root}")


def main(args):
    src = args.src_dir
    if src is None:
        import kagglehub
        src = kagglehub.dataset_download(args.kaggle_handle)
    print(f"source: {src}")
    img_dir = find(src, "CelebA-HQ-img", is_dir=True)
    with open(find(src, "CelebAMask-HQ-attribute-anno.txt")) as f:
        lines = f.read().splitlines()
    names = lines[1].split()
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        rows.append({"idx": int(os.path.splitext(parts[0])[0]),
                     **{n: int(v == "1") for n, v in zip(names, parts[1:])}})
    df = pd.DataFrame(rows)
    assert len(names) == 40 and len(df) == int(lines[0]), f"{len(names)} attributes, {len(df)} of {lines[0]} rows"
    mapping = pd.read_csv(find(src, "CelebA-HQ-to-CelebA-mapping.txt"), sep=r"\s+")
    orig = dict(zip(mapping["idx"], mapping["orig_idx"]))
    df["split"] = ["train" if orig[i] < CELEBA_TRAIN_END else "val" if orig[i] < CELEBA_VAL_END else "test"
                   for i in df["idx"]]
    df["file"] = [os.path.join(str(args.size), f"{i}.jpg") for i in df["idx"]]
    os.makedirs(os.path.join(args.out_dir, str(args.size)), exist_ok=True)
    for n, i in enumerate(df["idx"]):
        dst = os.path.join(args.out_dir, str(args.size), f"{i}.jpg")
        if not os.path.exists(dst):
            img = Image.open(os.path.join(img_dir, f"{i}.jpg")).convert("RGB")
            img.resize((args.size, args.size), Image.BICUBIC).save(dst, quality=95)
        if n % 2000 == 0:
            print(f"  {n}/{len(df)}")
    df = df[["idx", "file", "split"] + names]
    df.to_csv(os.path.join(args.out_dir, "attrs.csv"), index=False)
    print(df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main(parser.parse_args())
