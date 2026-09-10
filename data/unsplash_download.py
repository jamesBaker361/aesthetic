from diffusers.utils.loading_utils import load_image
from datasets import load_dataset
import os

n_images=1000
output_dir="unsplash"

os.makedirs(output_dir,exist_ok=True)

data=load_dataset("1aurent/unsplash-lite",split="train")

for r,row in enumerate(data):
    if r>=n_images:
        break
    row_dict=row["photo"]
    image_url=row_dict["image_url"]
    image=load_image(image_url)
    image.save(f"{output_dir}/{r}.png")