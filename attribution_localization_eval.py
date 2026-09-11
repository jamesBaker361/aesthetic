'''
Quantifies "does the attribution map actually point at the genitalia, or just
the face" instead of eyeballing heatmaps. For each image, NudeDetector gives
ground-truth boxes for genitalia and face; for each attribution method in
attribution.py, we compute the nsfw importance map and measure what fraction
of its mass falls inside each box, normalized by how much of the image that
box covers (so a bigger box isn't unfairly favored just for being bigger).

concentration = mass_fraction / area_fraction
  - 1.0 means "no better than a uniformly random map would do by chance"
  - >1.0 means the method really is concentrating attribution there
  - a method whose genitalia concentration is close to (or below) 1, while
    its face concentration is high, is the quantitative version of "it just
    highlights the face"

Needs: pip install nudenet
'''

import os
import csv

import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from nudenet import NudeDetector

from rewards import get_nsfw_model, get_aesthetic_model
from attribution import get_importance, get_importance_smoothgrad, get_importance_integrated_gradients

image_src_dir = "artificial_nsfw"
extension = "jpeg"
limit = 20  # smoothgrad/integrated-gradients do many forward/backward passes per image - keep this small
start_layer = 5
stop_layer = 15
out_dir = "attribution_localization_eval"

GENITALIA_CLASSES = {"FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"}
FACE_CLASSES = {"FACE_FEMALE", "FACE_MALE"}

METHODS = ["grad_cam", "smooth", "integrated"]


def best_box(detections, classes):
    '''Highest-confidence detection whose class is in `classes`, or None.'''
    candidates = [d for d in detections if d["class"] in classes]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d["score"])["box"]  # [x, y, w, h]


def importance_map(method, pil_img, nsfw_model, aesthetic_model, device, processor, clip_model):
    '''Returns the nsfw importance map (H, W) as a numpy array for the given method.'''
    if method == "grad_cam":
        _, importance_nsfw, _, _ = get_importance(pil_img, nsfw_model, aesthetic_model, device, processor, clip_model)
        importance_nsfw = importance_nsfw[start_layer:stop_layer]
    elif method == "smooth":
        _, importance_nsfw, _, _ = get_importance_smoothgrad(pil_img, nsfw_model, aesthetic_model, device, processor, clip_model)
        importance_nsfw = importance_nsfw[start_layer:stop_layer]
    elif method == "integrated":
        _, importance_nsfw, _, _ = get_importance_integrated_gradients(pil_img, nsfw_model, aesthetic_model, device, processor, clip_model)
        # single-element list - no per-layer slicing to do
    else:
        raise ValueError(f"unknown method {method}")

    return torch.stack(importance_nsfw).mean(dim=0).detach().cpu().numpy()


def box_stats(map_np, box):
    '''(mass_fraction, area_fraction, concentration) of `map_np` inside `box` ([x,y,w,h]).'''
    h_img, w_img = map_np.shape
    x, y, w, h = box
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w_img, int(x + w)), min(h_img, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, 0.0

    total_mass = map_np.sum()
    box_mass = map_np[y0:y1, x0:x1].sum()
    mass_fraction = float(box_mass / total_mass) if total_mass > 0 else 0.0
    area_fraction = ((x1 - x0) * (y1 - y0)) / (h_img * w_img)
    concentration = mass_fraction / area_fraction if area_fraction > 0 else 0.0
    return mass_fraction, area_fraction, concentration


def main():
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nsfw_model = get_nsfw_model()
    aesthetic_model = get_aesthetic_model()
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    detector = NudeDetector()

    files = [f for f in os.listdir(image_src_dir) if f.endswith(extension)]
    files = files[:limit] if limit >= 0 else files

    rows = []
    for file in files:
        path = os.path.join(image_src_dir, file)
        detections = detector.detect(path)

        genitalia_box = best_box(detections, GENITALIA_CLASSES)
        if genitalia_box is None:
            continue  # this check only means something on images with genitalia actually present
        face_box = best_box(detections, FACE_CLASSES)

        pil_img = Image.open(path).convert("RGB")

        for method in METHODS:
            map_np = importance_map(method, pil_img, nsfw_model, aesthetic_model, device, processor, clip_model)

            g_mass, g_area, g_conc = box_stats(map_np, genitalia_box)
            row = {
                "file": file,
                "method": method,
                "genitalia_mass_fraction": g_mass,
                "genitalia_area_fraction": g_area,
                "genitalia_concentration": g_conc,
            }
            if face_box is not None:
                f_mass, f_area, f_conc = box_stats(map_np, face_box)
                row.update({
                    "face_mass_fraction": f_mass,
                    "face_area_fraction": f_area,
                    "face_concentration": f_conc,
                })
            rows.append(row)
            print(f"{file} [{method}]: genitalia_concentration={g_conc:.2f}" +
                  (f" face_concentration={row.get('face_concentration'):.2f}" if face_box is not None else " (no face detected)"))

    if not rows:
        print("no images with a detected genitalia box - can't run this eval")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} (image, method) rows across {len(set(r['file'] for r in rows))} images with a genitalia box")
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        g_conc = np.mean([r["genitalia_concentration"] for r in method_rows])
        face_rows = [r for r in method_rows if "face_concentration" in r]
        f_conc = np.mean([r["face_concentration"] for r in face_rows]) if face_rows else float("nan")
        print(f"{method}: mean genitalia_concentration={g_conc:.2f}  mean face_concentration={f_conc:.2f}  (n={len(method_rows)}, n_with_face={len(face_rows)})")


if __name__ == "__main__":
    main()
