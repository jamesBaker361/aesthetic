import os
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
import tqdm

image_processor=VaeImageProcessor()

src_dir="artificial_nsfw"

size=512

for file in tqdm.tqdm(os.listdir(src_dir)):
    img=Image.open(f"{src_dir}/{file}")
    (h,w)=img.size
    if h!=size or w!=size:
        scale= float(size)/float(max(h,w))
        img = img.resize((int(scale*h),int(scale*w)))
    image_pt=image_processor.preprocess(img)


    new_image=image_processor.postprocess(image_pt)[0]
    new_image.save(f"{src_dir}/{file}")