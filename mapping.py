import datasets
import torch
from regression import get_maps,get_aesthetic_model,get_nsfw_model,CLIPVisionModelWithProjection,CLIPImageProcessor
from PIL import Image
from huggingface_hub import hf_hub_download

nsfw_model=get_nsfw_model()
aesthetic_model=get_aesthetic_model()
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


file_path = hf_hub_download(
    repo_id="deepghs/nsfw_detect",
    filename=f"nsfw_dataset_v1.zip",
    repo_type="dataset"
    )

import zipfile
from pathlib import Path

zip_path = Path(file_path)
target_dir = Path("path/to/extract")

target_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(target_dir)

import os

folder="smutfolder"

os.makedirs(folder,exist_ok=True)

for n in range(10):
    try:
        file_path = hf_hub_download(
        repo_id="wallstoneai/civitai-top-nsfw-images-with-metadata",
        filename=f"images/{n}.jpeg",
        repo_type="dataset"
        )
        img=Image.open(file_path).convert("RGB")
        h,w =img.size
        img=img.resize((h//4,w//4))
        new_img=get_maps(img,nsfw_model,aesthetic_model,device,processor,clip_model)
        new_img.save(f"{folder}/{n}.png")
    except:
        pass
    
    
    
print("all done")