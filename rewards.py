from PIL import Image
import io
import numpy as np
import torch
import os
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
import torch.nn.functional as F
import functools
from transformers import CLIPTokenizer,CLIPTextModelWithProjection

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

@functools.cache
def get_aesthetic_model():
    from score_words import MLP
    
    aesthetic_model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    if torch.cuda.is_available():
        state_dict = torch.load("improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
    else:
        state_dict = torch.load("improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth",map_location="cpu")   # load the model you trained previously or the model available in this repo
    aesthetic_model.load_state_dict(state_dict)
    aesthetic_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aesthetic_model.to(device)
    
    return aesthetic_model

@functools.cache
def get_nsfw_model():
    from score_words import NSFWScorer
    
    NSFWScorer.download_weights_on_node(os.getcwd())
    nsfw_model=NSFWScorer(model_dir=os.getcwd())
    nsfw_model.setup()
    
    return nsfw_model

class WordSimilarityModel:
    '''
    Callable as model(image_embeds), the same convention get_nsfw_model()'s
    NSFWScorer and get_aesthetic_model()'s MLP use - a drop-in nsfw_model for
    every clip_attribution* variant in attribution.py, including the
    backprop-based ones (grad_cam/smoothgrad/integrated): image_embeds is
    left with its autograd graph intact (no torch.no_grad() here), so
    nsfw_score.backward() upstream still flows back through it. Only the word
    embeddings are precomputed once, without grad, since they're fixed.
    image_embeds must already be a CLIP image projection from the same space
    these word embeddings live in (i.e. from a CLIPVisionModelWithProjection
    of the same model_name).
    '''
    def __init__(self,words:list,model_name:str="openai/clip-vit-large-patch14"):
        self.words=words
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        # CLIPTextModelWithProjection (not the plain CLIPTextModel) so
        # text_embeds lands in the same projected space as image_embeds - a
        # bare CLIPTextModel's pooler_output doesn't.
        text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device)

        inputs = {k: v.to(self.device) for k, v in tokenizer(words, padding=True, return_tensors="pt").items()}
        with torch.no_grad():
            self.embeddings = F.normalize(text_model(**inputs).text_embeds, dim=-1)  # [num_words, D], fixed

    def __call__(self, image_embeds):
        # max cosine similarity between the (fixed) word embeddings and each
        # given image embedding
        image_embeds = F.normalize(image_embeds, dim=-1)
        similarities = image_embeds @ self.embeddings.T  # [batch, num_words]
        return similarities.max(dim=-1).values

def get_nsfw_model_text(words:list):
    return WordSimilarityModel(words)


def aesthetic_score():
    
    aesthetic_model=get_aesthetic_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = {k: v.to(device) for k, v in processor(images=images, return_tensors="pt").items()}
        outputs = clip_model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        scores = aesthetic_model(image_embeds)
        return scores, {}

    return _fn

def nsfw_score():
    
    nsfw_model=get_nsfw_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = {k: v.to(device) for k, v in processor(images=images, return_tensors="pt").items()}
        outputs = clip_model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        scores = nsfw_model(image_embeds)
        return scores, {}
    
    return _fn
    