# The 40-attribute CelebA classifier whose logits casl_faces.py's EPR reads (the CASL paper's EPR uses attribute
# classifier logits over every annotated attribute, Sec. 4.4 / App. 7.8, but does not name the classifier): an
# ImageNet ResNet-50 with its fc replaced by 40 logits, BCE on prepare_celebahq.py's train split, random flips;
# the epoch with the best val mean accuracy is kept. -> --out ({"state_dict", "attrs", "size"}) and --out's .json
# (per-attribute test accuracy). classifier_input is the preprocessing both scripts use.
# Run: sbatch runpygpu_chip.sh train_celeba_classifier.py
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=str, default="data/celebahq/attrs.csv")
parser.add_argument("--out", type=str, default="data/celebahq/attr_r50.pth")
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--workers", type=int, default=1)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def classifier_input(img: torch.Tensor, size: int) -> torch.Tensor:
    '''(B, 3, H, W) in [0, 1] -> the classifier's input: resized to size x size, ImageNet-normalized.'''
    x = F.interpolate(img.float(), size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def attr_net(n: int) -> nn.Module:
    net = torchvision.models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, n)
    return net


class Faces(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, root: str, attrs: list):
        self.paths = [os.path.join(root, f) for f in df["file"]]
        self.y = torch.from_numpy(df[attrs].to_numpy(np.float32))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, n):
        img = torchvision.transforms.functional.to_tensor(Image.open(self.paths[n]).convert("RGB"))
        return img, self.y[n]


@torch.no_grad()
def accuracy(net, loader, size, device) -> np.ndarray:
    net.eval()
    right, total = 0, 0
    for img, y in loader:
        logits = net(classifier_input(img.to(device), size))
        right = right + ((logits > 0).float().cpu() == y).sum(0).numpy()
        total += len(y)
    return right / total


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.index)
    attrs = [c for c in df.columns if c not in ("idx", "file", "split")]
    root = os.path.dirname(args.index)
    loader = {s: torch.utils.data.DataLoader(Faces(df[df["split"] == s], root, attrs), batch_size=args.batch_size,
                                             shuffle=(s == "train"), num_workers=args.workers)
              for s in ("train", "val", "test")}
    net = attr_net(len(attrs))
    net.load_state_dict({k: v for k, v in torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).state_dict().items() if not k.startswith("fc.")},
        strict=False)
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader["train"]))
    best = -1.0
    for epoch in range(args.epochs):
        net.train()
        for n, (img, y) in enumerate(loader["train"]):
            img = img.to(device)
            flip = torch.rand(len(img), device=device) < 0.5
            img = torch.where(flip.view(-1, 1, 1, 1), img.flip(-1), img)
            loss = F.binary_cross_entropy_with_logits(net(classifier_input(img, args.size)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if n % 100 == 0:
                print(f"  epoch {epoch} batch {n}/{len(loader['train'])}: loss {loss.item():.4f}")
        val = accuracy(net, loader["val"], args.size, device).mean()
        print(f"epoch {epoch}: val mean accuracy {val:.4f}")
        if val > best:
            best = val
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "attrs": attrs, "size": args.size}, args.out)
    net.load_state_dict(torch.load(args.out, map_location=device)["state_dict"])
    test = accuracy(net, loader["test"], args.size, device)
    print(f"test mean accuracy {test.mean():.4f}")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"val_mean": float(best), "test_mean": float(test.mean()),
                   "test": {a: float(v) for a, v in zip(attrs, test)}}, f, indent=2)


if __name__ == "__main__":
    start = time.time()
    main(parser.parse_args())
    print(f"all done! {(time.time() - start) / 3600:.2f} hours")
