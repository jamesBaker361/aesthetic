"""
Standalone VQAScore and CLIPScore with Hugging Face transformers (no t2v_metrics / image-reward).

VQAScore = P("Yes" | image, 'Does this figure show "{text}"? Please answer yes or no.')
(Lin et al., ECCV 2024, https://arxiv.org/abs/2404.01291)

CLIPScore = 2.5 * max(cos(image embedding, text embedding), 0)
(Hessel et al., EMNLP 2021, https://arxiv.org/abs/2104.08718)

Requires: pip install "transformers>=4.49" accelerate torch pillow
"""
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, Qwen2_5_VLForConditionalGeneration

# Qwen/Qwen2.5-VL-3B-Instruct is faster; 7B is more accurate.
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


def _load_image(image):
    return Image.open(image).convert("RGB") if isinstance(image, str) else image


class VQAScore:
    def __init__(self, model_id=MODEL_ID, device="cuda", dtype=torch.bfloat16):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        tok = self.processor.tokenizer
        # first token of each answer word, both capitalizations
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", "yes"]})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["No", "no"]})

    @torch.no_grad()
    def __call__(self, image, text):
        """image: PIL.Image or path. text: e.g. 'a cat'. Returns dict of probabilities."""
        image = _load_image(image)
        question = f'Does this figure show "{text}"? Please answer yes or no.'
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)

        logits = self.model(**inputs).logits[0, -1].float()  # next-token logits
        probs = logits.softmax(dim=-1)
        p_yes = probs[self.yes_ids].sum().item()
        p_no = probs[self.no_ids].sum().item()
        return {
            "p_yes": p_yes,                                        # raw VQAScore
            "p_yes_normalized": p_yes / (p_yes + p_no + 1e-12),    # yes vs. no only
        }

    def batch(self, images, texts):
        """Score each image against its own text, one forward pass per pair.
        Returns a list of dicts with "score" (raw VQAScore) and "score_normalized"."""
        out = []
        for image, text in zip(images, texts):
            r = self(image, text)
            out.append({"score": r["p_yes"], "score_normalized": r["p_yes_normalized"]})
        return out


class CLIPScore:
    def __init__(self, model_id=CLIP_MODEL_ID, device="cuda", dtype=torch.float16):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def batch(self, images, texts):
        """Score each image against its own text in one batched pass.
        Returns a list of dicts with "score" (CLIPScore) and "cosine"."""
        images = [_load_image(im) for im in images]
        inputs = self.processor(text=list(texts), images=images, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        img = self.model.get_image_features(pixel_values=inputs["pixel_values"]).float()
        txt = self.model.get_text_features(input_ids=inputs["input_ids"],
                                           attention_mask=inputs["attention_mask"]).float()
        cos = torch.nn.functional.cosine_similarity(img, txt, dim=-1).cpu().tolist()
        return [{"score": 2.5 * max(c, 0.0), "cosine": c} for c in cos]

    def __call__(self, image, text):
        return self.batch([image], [text])[0]


def crop_to_mask(image, mask, pad=0.1):
    """Crop a PIL image to the bounding box of a boolean numpy mask, with padding."""
    import numpy as np
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph, pw = int((y1 - y0) * pad), int((x1 - x0) * pad)
    W, H = image.size
    return image.crop((max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph)))


if __name__ == "__main__":
    scorer = VQAScore()
    print(scorer("example.png", "a cat"))