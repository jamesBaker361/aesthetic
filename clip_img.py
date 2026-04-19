import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers.image_utils import load_image
from experiment_helpers.image_helpers import concat_images_horizontally,concat_images_vertically
import torchvision.transforms.functional as F
from diffusers.image_processor import VaeImageProcessor
from torchvision.transforms import InterpolationMode
from datasets import load_dataset

model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

all_images=[]

data=load_dataset("ares1123/celebrity_dataset",split="train")
for r,row in enumerate(data):
    if r>=10:
        break
    image=row["image"]
    image_processor=VaeImageProcessor()
    image_list=[image.resize((224,224))]
    image_pt=image_processor.preprocess(image,224,224)
    inputs = processor(images=image, return_tensors="pt")

    module_dict={}

    for name,module in model.named_modules():
        if name.find("k_proj")!=-1 or name.find("v_proj")!=-1 or name.find("q_proj")!=-1 or name.find("self_attn")!=-1:
            module_dict[name]=module
            def hook(module, input, output):
                setattr(module,"saved_output",output)
                if output is None:
                    return input
                return output
            
            module.register_forward_hook(hook)

    with torch.inference_mode():
        outputs = model(**inputs,output_attentions=True,output_hidden_states=True)
        
    for n in range(12):
        to_k=module_dict[f"vision_model.encoder.layers.{n}.self_attn.k_proj"]
        to_v=module_dict[f"vision_model.encoder.layers.{n}.self_attn.v_proj"]
        to_q=module_dict[f"vision_model.encoder.layers.{n}.self_attn.q_proj"]
        head_dim=module_dict[f"vision_model.encoder.layers.{n}.self_attn"].head_dim
        input_shape = to_k.saved_output.shape[:-1]

        hidden_shape = (*input_shape, -1, head_dim)
        queries = to_q.saved_output.view(hidden_shape).transpose(1, 2)
        keys    = to_k.saved_output.view(hidden_shape).transpose(1, 2)
        values  = to_v.saved_output.view(hidden_shape).transpose(1, 2)
        
        keys=keys.transpose(-2,-1)
        
        attn_weight=queries @ keys
        attn_weight=attn_weight.transpose(-2,-1)
        attn_weight=attn_weight.mean(dim=1)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        first_token_map=attn_weight[:,0,1:] #batch size =1, first token is CLS
        print("min max ",first_token_map.max(),first_token_map.min())
        first_token_map=first_token_map.reshape((7,7)).unsqueeze(0)
        first_token_map=F.resize(first_token_map,(224,224),interpolation=InterpolationMode.NEAREST).unsqueeze(0)

        map_norm = (first_token_map - first_token_map.min()) / (first_token_map.max() - first_token_map.min() + 1e-8)
        gray = image_pt.mean(dim=1, keepdim=True).expand_as(image_pt)
        new_image_pt = gray * (1 - map_norm) + image_pt * map_norm
        new_img=image_processor.postprocess(new_image_pt)[0]
        new_img.save(f"clip_{n}.png")
        image_list.append(new_img)
        
    horiz=concat_images_horizontally(image_list)
    all_images.append(horiz)
    
vert=concat_images_vertically(all_images)
vert.save("clip.png")