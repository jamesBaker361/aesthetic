# Retrains UnlearnCanvas's style and object classifiers the way the released
# ones were trained, but on sdxl-turbo images instead of the UnlearnCanvas
# dataset, so CRA / style accuracy measure what sdxl-turbo actually draws.
# The released checkpoints (UnlearnCanvas/ckpts/cls_model/*.pth) record their
# own training state, which this follows:
#   model      timm vit_large_patch16_224.augreg_in21k (pretrained), head -> Linear(1024, n)
#   optimizer  Adam, lr 1e-5, betas (0.9, 0.999), eps 1e-8, weight_decay 5e-4
#   schedule   CosineAnnealingLR(T_max=10, eta_min=0), stepped once per epoch; best test epoch kept
#              (released: style best at epoch 4 of 10, object at epoch 3)
#   data       400 images per style (20 objects x 20), 80/20 split; batch 32 (style) / 16 (object)
#              - not stored, inferred from the step counts (510 and 1275 steps per epoch)
#   transform  Resize((224, 224)) -> ToTensor -> Normalize([0.5], [0.5]) (UnlearnCanvas's eval transform)
# Labels are UnlearnCanvas's own lists (51 styles incl. Seed_Images, 20 objects),
# and checkpoints are saved in the same format, so evaluate_unlearncanvas.py
# uses them unchanged: --style_ckpt / --class_ckpt {out_dir}/style_sdxl.pth / class_sdxl.pth
# (cached classifier scores are keyed by checkpoint path, so they are recomputed).
#
# Stage 1 generates --n_per_pair images per style x object with sdxl-turbo from
# UnlearnCanvas's prompt ("A {object} image in {style} style."; Seed_Images =
# --seed_image_template), on seeds --seed_offset + i - kept apart from the answer
# set's seeds (188, 288, ...) so the classifier never trains on evaluated images.
# Stage 2 trains each classifier. One model on the GPU at a time.
#
#   python scripts/train_uc_classifier_sdxl.py                 # both classifiers, 51 x 20 x 20 images
#   python scripts/train_uc_classifier_sdxl.py --task style --n_per_pair 10
import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="both", choices=["style", "class", "both"])
parser.add_argument("--const", type=str, default="UnlearnCanvas/machine_unlearning/evaluation/constants/const.py")
parser.add_argument("--styles", nargs="*", default=None, help="subset of styles (default: all 51 incl. Seed_Images)")
parser.add_argument("--objects", nargs="*", default=None, help="subset of objects (default: all 20)")
parser.add_argument("--n_per_pair", type=int, default=20, help="images per style x object (UnlearnCanvas: 20)")
parser.add_argument("--seed_offset", type=int, default=100000, help="image i of a pair uses seed_offset + i")
parser.add_argument("--template", type=str, default="A {object} image in {style} style.")
parser.add_argument("--seed_image_template", type=str, default="A photo of {object}.",
                    help="prompt for the Seed_Images class (UnlearnCanvas's unstylized photos)")
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--gen_batch_size", type=int, default=8)
parser.add_argument("--data_dir", type=str, default="evaluation/uc_sdxl_classifier/images")
parser.add_argument("--out_dir", type=str, default="UnlearnCanvas/ckpts/cls_model_sdxl")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--style_batch_size", type=int, default=32)
parser.add_argument("--class_batch_size", type=int, default=16)
parser.add_argument("--test_frac", type=float, default=0.2)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--amp", type=str, default="fp16", choices=["fp16", "bf16", "none"],
                    help="mixed precision (bf16 needs an Ampere+ GPU; the RTX 6000 / 2080 Ti nodes are Turing)")
parser.add_argument("--grad_accum", type=int, default=1,
                    help="split each batch into this many micro-batches (same effective batch, less memory - "
                         "e.g. 2-4 on an 11 GB card)")
parser.add_argument("--no_optimizer_state", action="store_true",
                    help="don't store optimizer/scheduler state (the released files do; it triples the size)")
parser.add_argument("--disable_generate", action="store_true")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

spec = importlib.util.spec_from_file_location("uc_const", args.const)
const = importlib.util.module_from_spec(spec)
spec.loader.exec_module(const)
THEMES, CLASSES = list(const.theme_available), list(const.class_available)  # label order of the released heads
styles = args.styles or THEMES
objects = args.objects or CLASSES
for s in styles:
    assert s in THEMES, f"unknown style {s}"
for o in objects:
    assert o in CLASSES, f"unknown object {o}"


def words(name):
    return name.replace("_", " ")


def prompt(style, obj):
    if style == "Seed_Images":
        return args.seed_image_template.format(object=words(obj))
    return args.template.format(object=words(obj), style=words(style))


def image_path(style, obj, i):
    return os.path.join(args.data_dir, f"{style}_{obj}_seed{args.seed_offset + i}.jpg")


# ---------------------------------------------------------------- stage 1: sdxl-turbo images

@torch.no_grad()
def generate():
    jobs = [(s, o, i) for s in styles for o in objects for i in range(args.n_per_pair)
            if not os.path.exists(image_path(s, o, i))]
    total = len(styles) * len(objects) * args.n_per_pair
    print(f"generate: {len(jobs)} of {total} images to make")
    if not jobs:
        return
    from diffusers import AutoPipelineForText2Image
    fp16 = device == "cuda"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16 if fp16 else torch.float32,
        variant="fp16" if fp16 else None).to(device)
    pipe.set_progress_bar_config(disable=True)
    os.makedirs(args.data_dir, exist_ok=True)
    start = time.time()
    for b in range(0, len(jobs), args.gen_batch_size):
        part = jobs[b:b + args.gen_batch_size]
        images=[]
        for s, o, i in part:
            
            image = pipe(prompt=prompt(s, o), height=args.size, width=args.size,
                      num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                      generator=torch.Generator().manual_seed(args.seed_offset + i)).images[0]
            images.append(image)
        for (s, o, i), im in zip(part, images):
            path = image_path(s, o, i)
            tmp = f"{path[:-4]}.tmp{os.getpid()}.jpg"
            im.save(tmp)
            os.replace(tmp, path)  # write-then-rename: a parallel job never reads half an image
        done = b + len(part)
        if done % (50 * args.gen_batch_size) < args.gen_batch_size or done == len(jobs):
            print(f"  {done}/{len(jobs)} ({(time.time() - start) / done:.2f}s/image)")
    pipe.to("cpu")
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- stage 2: classifiers

from torchvision import transforms  # noqa: E402

TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


class Images(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items  # (path, label)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return TRANSFORM(Image.open(path).convert("RGB")), label


def split():
    """Per style x object pair, the last test_frac of its seeds are test - every pair is in both splits."""
    n_test = max(1, int(round(args.test_frac * args.n_per_pair)))
    train, test = [], []
    for s in styles:
        for o in objects:
            for i in range(args.n_per_pair):
                path = image_path(s, o, i)
                if os.path.exists(path):
                    (test if i >= args.n_per_pair - n_test else train).append((path, s, o))
    return train, test


def train(task):
    import timm
    labels = THEMES if task == "style" else CLASSES
    train_rows, test_rows = split()

    def to_items(rows):  # rows are (path, style, object)
        return [(path, labels.index(style if task == "style" else obj)) for path, style, obj in rows]

    train_items, test_items = to_items(train_rows), to_items(test_rows)
    batch = args.style_batch_size if task == "style" else args.class_batch_size
    print(f"\n{task} classifier: {len(labels)} classes, {len(train_items)} train / {len(test_items)} test images, "
          f"batch {batch}, {args.epochs} epochs")
    g = torch.Generator().manual_seed(args.seed)
    assert batch % args.grad_accum == 0, "--grad_accum must divide the batch size"
    train_dl = torch.utils.data.DataLoader(Images(train_items), batch_size=batch, shuffle=True, generator=g,
                                           num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_dl = torch.utils.data.DataLoader(Images(test_items), batch_size=64, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=True)

    torch.manual_seed(args.seed)
    model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True)
    model.head = torch.nn.Linear(1024, len(labels))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp) if device == "cuda" else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)

    def evaluate():
        model.eval()
        correct, n, per_class = 0, 0, np.zeros((len(labels), 2))
        with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            for x, y in test_dl:
                pred = model(x.to(device)).argmax(-1).cpu()
                correct += int((pred == y).sum())
                n += len(y)
                for yy, pp in zip(y.tolist(), pred.tolist()):
                    per_class[yy] += [pp == yy, 1]
        return 100.0 * correct / max(n, 1), per_class

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{task}_sdxl.pth")
    best_acc, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        loss_sum, correct, n, start = 0.0, 0, 0, time.time()
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            for xm, ym in zip(x.chunk(args.grad_accum), y.chunk(args.grad_accum)):  # one optimizer step per batch
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    logits = model(xm)
                    loss = F.cross_entropy(logits.float(), ym)
                scaler.scale(loss * len(ym) / len(y)).backward()
                loss_sum += loss.item() * len(ym)
                correct += int((logits.argmax(-1) == ym).sum())
                n += len(ym)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        train_loss, train_acc = loss_sum / n, 100.0 * correct / n
        test_acc, per_class = evaluate()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "test_accuracy": test_acc})
        print(f"  epoch {epoch}: train loss {train_loss:.4f}, train acc {train_acc:.2f}%, test acc {test_acc:.2f}% "
              f"({time.time() - start:.0f}s)")
        if test_acc > best_acc:  # keep the best test epoch, as the released checkpoints did
            best_acc = test_acc
            ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                    "train loss": train_loss, "train_accuracy": train_acc, "test_accuracy": test_acc,
                    "best_acc": best_acc, "labels": labels,
                    "per_class_test_accuracy": {labels[c]: float(a / t) for c, (a, t) in enumerate(per_class) if t},
                    "data": {"styles": styles, "objects": objects, "n_per_pair": args.n_per_pair,
                             "seed_offset": args.seed_offset, "template": args.template,
                             "seed_image_template": args.seed_image_template, "model": "stabilityai/sdxl-turbo",
                             "num_inference_steps": args.num_inference_steps, "size": args.size}}
            if not args.no_optimizer_state:
                ckpt["optimizer_state_dict"] = optimizer.state_dict()
                ckpt["scheduler_state_dict"] = scheduler.state_dict()
            tmp = f"{out_path}.tmp{os.getpid()}"
            torch.save(ckpt, tmp)
            os.replace(tmp, out_path)
            print(f"    saved best -> {out_path}")
    with open(os.path.join(args.out_dir, f"{task}_sdxl_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    worst = sorted(((a / t, labels[c]) for c, (a, t) in enumerate(per_class) if t))[:5]
    print(f"  {task}: best test acc {best_acc:.2f}% | hardest classes (last epoch): "
          + ", ".join(f"{name} {acc:.2f}" for acc, name in worst))
    model.to("cpu")
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    start = time.time()
    if not args.disable_generate:
        generate()
    for task in (["style", "class"] if args.task == "both" else [args.task]):
        train(task)
    print(f"all done! {(time.time() - start) / 3600:.2f} hours")
