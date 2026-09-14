import os
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

image_processor=VaeImageProcessor()

src_dir="real_fruit"

for file in os.listdir(src_dir):
    img=Image.open(f"{src_dir}/{file}")
    (h,w)=img.size
    if h!=256 or w!=256:
        scale= 256./float(max(h,w))
        img = img.resize((int(scale*h),int(scale*w)))
    image_pt=image_processor.preprocess(img)


    new_image=image_processor.postprocess(image_pt)[0]
    new_image.save(f"{src_dir}/{file}")