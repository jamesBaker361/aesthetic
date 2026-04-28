from PIL import Image
import io
import numpy as np
import torch
import os
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
import torch.nn.functional as F

def light_reward():
    def _fn(images, prompts, metadata):
        reward = images.reshape(images.shape[0],-1).mean(1)
        return np.array(reward.cpu().detach()),{}
    return _fn


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from score_words import MLP
    
    aesthetic_model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    state_dict = torch.load("improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
    aesthetic_model.load_state_dict(state_dict)
    aesthetic_model.eval()
    
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").cuda()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = {k: v.cuda() for k, v in processor(images=images, return_tensors="pt").items()}
        outputs = clip_model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        scores = aesthetic_model(image_embeds)
        return scores, {}

    return _fn

def nsfw_score():
    from score_words import NSFWScorer
    
    NSFWScorer.download_weights_on_node(os.getcwd())
    nsfw_model=NSFWScorer(model_dir=os.getcwd())
    nsfw_model.setup()
    
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").cuda()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = {k: v.cuda() for k, v in processor(images=images, return_tensors="pt").items()}
        outputs = clip_model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        scores = nsfw_model(image_embeds)
        return scores, {}
    
    return _fn
    