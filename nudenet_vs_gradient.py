'''
For n images: builds an "importance map" from NudeNet's detected boxes
(each box filled with its detection score, so overlapping boxes take the max)
and compares it side by side with the grad*activation attribution map from
attribution.get_importance. Saves [original | nudenet boxes | gradient] concatenated
horizontally per image, so the two can be eyeballed directly against each other.

Also saves a second image per input: at each threshold in THRESHOLDS, every
NudeNet box (excluding face boxes) that has ANY pixel in the gradient map's
top-quantile patches (quantile >= threshold, same rank-based quantile
clip_attribution computes) is drawn in FULL - not just the intersecting
pixels - and boxes with no overlap at all are dropped entirely. So this
shows "which whole regions the two methods agree matter," not a pixel-level
intersection mask.
'''

import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from nudenet import NudeDetector
from experiment_helpers.image_helpers import concat_images_horizontally

from rewards import get_nsfw_model, get_aesthetic_model
from attribution import get_importance

image_src_dir = "artificial_nsfw"
extension = "jpeg"
n = 10
start_layer = 5
stop_layer = 15
out_dir = "nudenet_vs_gradient"

FACE_CLASSES = {"FACE_FEMALE", "FACE_MALE"}
THRESHOLDS = [0.95, 0.9, 0.75, 0.5]


def nudenet_importance_map(detections, h_img, w_img):
    map_np = np.zeros((h_img, w_img), dtype=np.float32)
    for d in detections:
        x, y, w, h = d["box"]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w_img, int(x + w)), min(h_img, int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        map_np[y0:y1, x0:x1] = np.maximum(map_np[y0:y1, x0:x1], d["score"])
    return map_np


def full_box_overlap_mask(detections, quantile, threshold, exclude_classes):
    '''
    Binary (H, W) mask: for each box whose class isn't in exclude_classes, if
    ANY pixel inside it has quantile >= threshold, the WHOLE box is set to 1;
    otherwise the box contributes nothing (not even its overlapping pixels).
    '''
    h_img, w_img = quantile.shape
    mask = np.zeros((h_img, w_img), dtype=np.float32)
    for d in detections:
        if d["class"] in exclude_classes:
            continue
        x, y, w, h = d["box"]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w_img, int(x + w)), min(h_img, int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        if (quantile[y0:y1, x0:x1] >= threshold).any():
            mask[y0:y1, x0:x1] = 1.0
    return mask


def quantile_map(map_np):
    '''Per-pixel rank, normalized to [0,1] - same construction clip_attribution uses.'''
    flat = map_np.flatten()
    ranks = flat.argsort().argsort().astype(np.float64)
    quantile = ranks / max(flat.size - 1, 1)
    return quantile.reshape(map_np.shape)


def gradient_importance_map(pil_img, nsfw_model, aesthetic_model, device, processor, clip_model):
    _, importance_nsfw, _, _ = get_importance(pil_img, nsfw_model, aesthetic_model, device, processor, clip_model)
    importance_nsfw = importance_nsfw[start_layer:stop_layer]
    return torch.stack(importance_nsfw).mean(dim=0).detach().cpu().numpy()


def overlay_heatmap(pil_img, map_np):
    '''Same COLORMAP_BONE/sqrt-sharpen/invert style used elsewhere in this repo (see get_maps in attribution.py).'''
    img_np = np.array(pil_img.convert("RGB"))
    h_img, w_img = img_np.shape[:2]

    heatmap = cv2.resize(map_np.astype(np.float32), (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.clip(heatmap, 0, 1) ** 0.5

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_BONE)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    return Image.fromarray(np.uint8(255 - overlay))


def main():
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nsfw_model = get_nsfw_model()
    aesthetic_model = get_aesthetic_model()
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    detector = NudeDetector()

    files = [f for f in os.listdir(image_src_dir) if f.endswith(extension)][:n]

    for i, file in enumerate(files):
        path = os.path.join(image_src_dir, file)
        pil_img = Image.open(path).convert("RGB")
        w_img, h_img = pil_img.size

        detections = detector.detect(path)
        nudenet_map = nudenet_importance_map(detections, h_img, w_img)
        gradient_map = gradient_importance_map(pil_img, nsfw_model, aesthetic_model, device, processor, clip_model)

        nudenet_overlay = overlay_heatmap(pil_img, nudenet_map)
        gradient_overlay = overlay_heatmap(pil_img, gradient_map)

        concat = concat_images_horizontally([pil_img, nudenet_overlay, gradient_overlay])
        concat.save(os.path.join(out_dir, f"{i}_{file}"))

        quantile = quantile_map(gradient_map)
        overlap_panels = [
            overlay_heatmap(pil_img, full_box_overlap_mask(detections, quantile, t, FACE_CLASSES))
            for t in THRESHOLDS
        ]
        concat_images_horizontally(overlap_panels).save(os.path.join(out_dir, f"{i}_overlap_{file}"))

        print(f"{i}: {file} -> {len(detections)} nudenet detections")


if __name__ == "__main__":
    main()
