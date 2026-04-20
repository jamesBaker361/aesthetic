import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from sdxl_unbox.SAE import SparseAutoencoder
from sdxl_pipe import HookedStableDiffusionXLWithUNetPipeline
from diffusers import UNet2DConditionModel
from transformers import AutoProcessor, CLIPVisionModel,CLIPVisionModelWithProjection
from diffusers.image_processor import VaeImageProcessor
import torch
from PIL import Image
import numpy as np

import time
from tqdm import tqdm

from experiment_helpers.init_helpers import default_parser,repo_api_init

parser=default_parser({"save_dir":"embeddings"})
parser.add_argument("--src_dir",type=str,default="laion")
parser.add_argument("--size",type=int,default=256)
            

def main(args):
    mixed_precision : str = args.mixed_precision
    project_name : str = args.project_name
    gradient_accumulation_steps : int = args.gradient_accumulation_steps
    repo_id : str = args.repo_id
    lr : float = args.lr
    epochs : int = args.epochs
    limit : int = args.limit
    save_dir : str = args.save_dir
    batch_size : int = args.batch_size
    val_interval : int = args.val_interval
    load_hf  = args.load_hf
    src_dir : str = args.src_dir
    size: int = args.size
    

    clip_dir=os.path.join(save_dir,"clip")
    sdxl_dir=os.path.join(save_dir, "sdxl")
    for d in [save_dir,clip_dir,sdxl_dir]:
        os.makedirs(d,exist_ok=True)



    dtype=torch.float16

    if torch.cuda.is_available():

        pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
            'stabilityai/sdxl-turbo',
            torch_dtype=dtype,
            device_map="balanced",
            variant=("fp16" if dtype==torch.float16 else None)
        )
    else:
         pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
            'stabilityai/sdxl-turbo',
            torch_dtype=dtype,
            device_map="cpu",
            variant=("fp16" if dtype==torch.float16 else None)
        )

    path_to_checkpoints = './sdxl_unbox/checkpoints/'

    block_list=[
        "down_blocks.2.attentions.1",
        "mid_block.attentions.0",
        "up_blocks.0.attentions.0",
         "up_blocks.0.attentions.1"
    ]
    
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_processor=VaeImageProcessor()
    
    def hook(module, input, output):
        setattr(module,"saved_input",input)
        setattr(module,"saved_output",output)
        return output
    unet: UNet2DConditionModel =pipe.unet
    vae=pipe.vae
    scheduler=pipe.scheduler
    tokenizer=pipe.tokenizer
    
    block_dict={}
    
    for name,module in unet.named_modules():
        if name in block_list:
            module.register_forward_hook(hook)
            block_dict[name]=module
            
    def get_unet_device_dtype(unet):
        param = next(unet.parameters())
        return param.device, param.dtype
            
    device, dtype = get_unet_device_dtype(pipe.unet)
            
    path_list=[f for f in os.listdir(src_dir) if f.endswith(".jpg")]
    count=len([p for p in os.listdir(save_dir) if p.endswith(".npz")])
    
    print(f"processed {count}/{len(path_list)} images")
    
    session_count=0
    
    for r,jpg_path in enumerate(tqdm(path_list)):
        if r<count:
            continue
        
        if r==limit:
            break
        
        npz_path=os.path.join(save_dir,jpg_path+".npz")
        if os.path.exists(npz_path):
            continue
        
        image=Image.open(os.path.join(src_dir,jpg_path)).convert("RGB")
        (h,w)=image.size
        if h<4 or w<4:
            print("hella small ",jpg_path)
            continue
        with torch.no_grad():
            clip_inputs = processor(images=image, return_tensors="pt")
            clip_outputs=model(**clip_inputs,return_dict=True,output_attentions=True,output_hidden_states=True)
            
            result_dict={}
            for attr in ["image_embeds", "last_hidden_state","hidden_states"]:
                try:
                    result_dict[attr]=getattr(clip_outputs,attr).cpu().detach().numpy()
                except AttributeError:
                    hidden_states=getattr(clip_outputs,attr)
                    hidden_states=np.stack([h.cpu().detach().numpy() for h in hidden_states])
                    result_dict[attr]=hidden_states
                    
            
            image_pt=image_processor.preprocess(image,size,size).to(device=device,dtype=dtype) #all images have to be the same size so we can do batching

            latents=vae.encode(image_pt).latent_dist.sample()
            noise = torch.randn_like(latents)
            
            timesteps = torch.randint(
                0, 10, (latents.shape[0],), device=latents.device #only 10, we want this to be very low noise
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = scheduler.add_noise(latents, noise, timesteps)
            
            (prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            )=pipe.encode_prompt("image","image",device,1,False," "," ")
            timestep_cond=None
            add_text_embeds = pooled_prompt_embeds

            if pipe.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

            original_size = (size, size)
            target_size =(size, size)
            crops_coords_top_left=(0,0)
            add_time_ids = pipe._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,)

            actual_batch_size = noisy_model_input.shape[0]
            prompt_embeds = prompt_embeds.expand(actual_batch_size, -1, -1).contiguous()
            add_text_embeds = add_text_embeds.expand(actual_batch_size, -1).contiguous()
            add_time_ids = add_time_ids.expand(actual_batch_size, -1).contiguous()
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            
            model_pred = unet.forward(
                        noisy_model_input,timesteps,
                                            encoder_hidden_states=prompt_embeds,
                                            timestep_cond=timestep_cond,
                                            added_cond_kwargs=added_cond_kwargs,
                                            return_dict=False,)[0]
            
            
            
            for name,block in block_dict.items():
                for key in ["saved_output","saved_input"]:
                    value=getattr(block,key)
                    if type(value)==tuple:
                        value=value[0]
                    if torch.isnan(value).any():
                        print(npz_path,"nan value ",key)
                    result_dict[f"{key}.{name}"]=value.cpu().detach().numpy()
            
            np.savez(npz_path,**result_dict)
            session_count+=1
            if session_count%250==0:
                print(f"processed {session_count}+{count}={session_count+count} / {len(path_list)}")
                


    
        


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")