'''
Sanity check for the "attribution keeps highlighting the face, not the
explicit content" problem: since we don't have a genitalia/body-part
detector, the closest thing we can build without one is a face-only crop vs.
the same image with the face region blacked out ("everything but the face").

For each image with a detected face, scores three variants with the same
nsfw_model used everywhere else in this repo (rewards.get_nsfw_model):
  - original            (full image, unmodified)
  - face_only           (crop of just the detected face region)
  - face_blacked_out     (full image, face region zeroed out)

If face_only alone scores meaningfully high (close to or above the original,
and clearly above what you'd expect from a face crop with no explicit
content), that's evidence the nsfw_model's decision boundary itself keys off
face/context rather than the specific content - in which case no attribution
method on top of it was ever going to localize genitalia, since the signal
it's explaining doesn't live there.
'''

import os
import csv

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from rewards import get_nsfw_model

image_src_dir = "artificial_nsfw"
extension = "jpeg"
limit = 50
face_padding = 0.2  # extra margin around the detected face box, as a fraction of its size
out_dir = "face_bias_check"


def get_face_box(img_np_bgr):
    '''Largest detected face (x, y, w, h) in pixel coords, or None.'''
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])  # largest by area


def score_pil(pil_img, nsfw_model, processor, clip_model, device):
    inputs = {k: v.to(device) for k, v in processor(images=pil_img, return_tensors="pt").items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        score = nsfw_model(image_embeds)
    return float(score.detach().cpu())


def main():
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nsfw_model = get_nsfw_model()
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    files = [f for f in os.listdir(image_src_dir) if f.endswith(extension)]
    files = files[:limit] if limit >= 0 else files

    rows = []
    for n, file in enumerate(files):
        pil_img = Image.open(os.path.join(image_src_dir, file)).convert("RGB")
        img_np_rgb = np.array(pil_img)
        img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

        face_box = get_face_box(img_np_bgr)
        if face_box is None:
            continue

        h_img, w_img = img_np_rgb.shape[:2]
        x, y, w, h = face_box
        pad_x, pad_y = int(w * face_padding), int(h * face_padding)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)

        face_only = Image.fromarray(img_np_rgb[y0:y1, x0:x1])

        blacked_out_np = img_np_rgb.copy()
        blacked_out_np[y0:y1, x0:x1] = 0
        face_blacked_out = Image.fromarray(blacked_out_np)

        original_score = score_pil(pil_img, nsfw_model, processor, clip_model, device)
        face_only_score = score_pil(face_only, nsfw_model, processor, clip_model, device)
        face_blacked_out_score = score_pil(face_blacked_out, nsfw_model, processor, clip_model, device)

        rows.append({
            "file": file,
            "original_score": original_score,
            "face_only_score": face_only_score,
            "face_blacked_out_score": face_blacked_out_score,
        })
        print(f"{file}: original={original_score:.4f} face_only={face_only_score:.4f} face_blacked_out={face_blacked_out_score:.4f}")

        if n < 10:
            face_only.save(os.path.join(out_dir, f"{n}_face_only_{file}"))
            face_blacked_out.save(os.path.join(out_dir, f"{n}_face_blacked_out_{file}"))

    if not rows:
        print("no faces detected in any image - can't run this check")
        return

    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    original_scores = np.array([r["original_score"] for r in rows])
    face_only_scores = np.array([r["face_only_score"] for r in rows])
    face_blacked_out_scores = np.array([r["face_blacked_out_score"] for r in rows])

    print(f"\n{len(rows)} images with a detected face")
    print(f"mean original score:          {original_scores.mean():.4f}")
    print(f"mean face_only score:         {face_only_scores.mean():.4f}")
    print(f"mean face_blacked_out score:  {face_blacked_out_scores.mean():.4f}")
    print(f"face_only >= face_blacked_out on {(face_only_scores >= face_blacked_out_scores).mean()*100:.1f}% of images")
    print(f"face_only retains {face_only_scores.mean()/max(original_scores.mean(),1e-8)*100:.1f}% of the original score on average")


if __name__ == "__main__":
    main()
