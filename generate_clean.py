import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.argprint import print_args
from diffusers import DiffusionPipeline,UNet2DConditionModel,AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from sdxl_unbox.SAE import SparseAutoencoder
import torch
import numpy as np
import csv
import sys

import time
import torch.nn.functional as F
from datasets import load_dataset
import json
from PIL import Image

from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init
from sdxl_pipe import HookedStableDiffusionXLWithUNetPipeline
import random
import nltk
from nltk.corpus import wordnet as wn
from sdxl_extract import extract_vanilla
from sparsify import sparsify_embeddings
from regression import run_regression,clip_attribution,get_importance
from d3po_rewards import get_aesthetic_model,get_nsfw_model
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
from peft import LoraConfig
from accelerate import Accelerator

def keep_top_n(x, n, dim=-1):
    """
    Zero all but the top-n activations along `dim`.
    
    Args:
        x: input tensor
        n: number of activations to keep
        dim: dimension along which to keep top-n
        
    Returns:
        Tensor with only top-n values preserved.
    """
    values, indices = torch.topk(x, n, dim=dim)

    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(dim, indices, True)

    return torch.where(mask, x, torch.zeros_like(x))


parser=default_parser()
UNTRAINED="untrained"

parser.add_argument("--y_column",type=str,default="aesthetic") #column 0 = aesthetic column = 1 = p(unsafe)
parser.add_argument("--num_inference_steps",type=int,default=8)
parser.add_argument("--size",type=int,default=512)
parser.add_argument("--method",type=str,default=UNTRAINED)
parser.add_argument("--image_src_dir",type=str,default="laion")
parser.add_argument("--image_dest_dir",type=str,default="artificial_images")
parser.add_argument("--n_random",type=int,default=50)
parser.add_argument("--embedding_dir",type=str,default="embeddings")
parser.add_argument("--sparse_embedding_dir",type=str,default="sparse_embeddings")
parser.add_argument("--clip_dir",type=str,default="clip_sparse_embeddings")
parser.add_argument("--clip_limit",type=int,default=-1)
parser.add_argument("--regression_limit",type=int,default=-1)
parser.add_argument("--stats_dir",type=str,default="statistics")
parser.add_argument("--start_layer",type=int,default=5)
parser.add_argument("--stop_layer",type=int,default=20)
parser.add_argument("--disable_get_images",action="store_true")
parser.add_argument("--disable_extract_vanilla",action="store_true")
parser.add_argument("--disable_sparsify_embeddings",action="store_true")
parser.add_argument("--disable_clip_attribution",action="store_true")
parser.add_argument("--disable_run_regression",action="store_true")
parser.add_argument("--disable_train_lora",action="store_true")
parser.add_argument("--lora_dir",type=str,default="lora")
parser.add_argument("--top_k",type=int,default=10)
parser.add_argument("--aesthetic_prompt",action="store_true")
parser.add_argument("--nsfw_prompt",action="store_true")
parser.add_argument("--random_prompt",action="store_true")
parser.add_argument("--lora_epochs",type=int,default=5)
parser.add_argument("--lora_use_mask",action="store_true")
parser.add_argument("--lora_use_filter",action="store_true")
parser.add_argument("--lora_use_noise",action="store_true")
parser.add_argument("--start_step",type=int,default=2)
parser.add_argument("--end_step",type=int,default=5)
parser.add_argument("--mode",type=str,default="out")
job_id=os.environ["SLURM_JOB_ID"]
parser.add_argument("--err",type=str,default=f"slurm_chip/generic/{job_id}.err")
parser.add_argument("--out",type=str,default=f"slurm_chip/generic/{job_id}.out")
#def clip_attribution(image_src_dir:str,dest_dir:str,limit:int):
# def run_regression(block:str,y_column:str,limit:int,clip_src_dir:str,stats_dest_dir:str):
# generate images using RL or prompts
# sdxl extract 
# extract to -> sparse
# regress scores on activations
# 



def get_images(image_dest_dir:str,method:str,n_random:int,size:int,num_inference_steps:int,aesthetic_prompt:bool,nsfw_prompt:bool,random_prompt:bool):
    os.makedirs(image_dest_dir,exist_ok=True)
    
    prompt_list=[]
    if nsfw_prompt:
        prompt_list+=[row["prompt"] for row in load_dataset("AIML-TUDA/i2p", split="train")]
    if aesthetic_prompt:
        prompt+=[row["prompt"] for row in load_dataset("moonworks/lunara-aesthetic", split="train")]
        
    if random_prompt:
        nltk.download("wordnet")

        words = set()
        for syn in wn.all_synsets():
            for lemma in syn.lemma_names():
                words.add(lemma.lower())

        word_list=[w for w in words]
        random.shuffle(word_list)
        word_list=word_list[:n_random]
        prompt_list+=word_list
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    base_pipe=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(device)
    setattr(base_pipe,"safety_checker",None)
    if method==UNTRAINED:
        diff_pipe=base_pipe
    else:
        raise NotImplementedError(f"method={method} not implemented")
    for p,prompt in enumerate(prompt_list):
        base_path=f"{image_dest_dir}/base_{p}.jpg"
        if os.path.exists(base_path):
            continue
        generator=torch.Generator()
        generator.manual_seed(p)
        base_prompt=prompt
        if method=="untrained":
            base_prompt=""
        base_image=base_pipe(base_prompt,height=size,width=size,generator=generator,num_inference_steps=num_inference_steps).images[0]
        base_image.save(base_path)
        if method!=UNTRAINED:
            diff_image=diff_pipe(prompt,height=size,width=size,generator=generator,num_inference_steps=num_inference_steps).images[0]
            diff_path=f"{image_dest_dir}/diff_{p}.jpg"
            diff_image.save(diff_path)


class LoraDataset(torch.utils.data.Dataset):
    def __init__(self,image_dir:str,vae:AutoencoderKL,device,latent_h:int,latent_w:int):
        super().__init__()
        self.path_list=[
            os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith("jpg")
        ]
        self.image_processor=VaeImageProcessor()
        self.nsfw_model=get_nsfw_model()
        self.aesthetic_model=get_aesthetic_model()
        self.device = device
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.vae=vae
        self.latent_h=latent_h
        self.latent_w=latent_w
        self._cache:dict[int,dict]={}

    def __len__(self):
        return len(self.path_list)

    def _to_latent_mask(self,mask_2d:torch.Tensor)->torch.Tensor:
        # mask_2d: (H, W) -> (1, latent_h, latent_w) float, channel dim added for broadcast
        m=mask_2d.float().unsqueeze(0).unsqueeze(0)
        m=F.interpolate(m,size=(self.latent_h,self.latent_w),mode="nearest")
        return m[0]

    def __getitem__(self, index):
        if index in self._cache:
            return self._cache[index]
        pil_img=Image.open(self.path_list[index]).convert("RGB")
        pt_img=self.image_processor.preprocess(pil_img).to(self.device,dtype=self.vae.dtype)
        importance_aesthetic,importance_nsfw=get_importance(pil_img,self.nsfw_model,self.aesthetic_model,self.device,self.processor,self.clip_model)
        start_layer=5
        stop_layer=15
        importance_aesthetic=importance_aesthetic[start_layer:stop_layer]
        importance_nsfw=importance_nsfw[start_layer:stop_layer]

        avg_aesthetic=torch.stack(importance_aesthetic).mean(dim=0)
        avg_nsfw=torch.stack(importance_nsfw).mean(dim=0)

        avg_nsfw=avg_nsfw-avg_nsfw.min()
        avg_nsfw=avg_nsfw/(avg_nsfw.max()+1e-8)
        avg_aesthetic=avg_aesthetic-avg_aesthetic.min()
        avg_aesthetic=avg_aesthetic/(avg_aesthetic.max()+1e-8)
        aesthetic_mask = avg_aesthetic < torch.quantile(avg_aesthetic, 0.9)
        nsfw_mask=avg_nsfw<torch.quantile(avg_nsfw,0.9)

        with torch.no_grad():
            latent=self.vae.encode(pt_img).latent_dist.sample()[0]

        item={
            "latent":latent.detach().cpu(),
            "nsfw_mask":self._to_latent_mask(nsfw_mask).cpu(),
            "aesthetic_mask":self._to_latent_mask(aesthetic_mask).cpu(),
        }
        self._cache[index]=item
        return item
        

def train_lora(lora_dir:str,rank:int,device,epochs:int,image_dir:str,batch_size:int,
               accelerator:Accelerator,lr:float,
               filter_dict:dict[str,torch.Tensor],
               sae_dict:dict[str,SparseAutoencoder],
               use_mask:bool,
               use_filter:bool,
               use_noise:bool,
               size:int,
               mode:str)->DiffusionPipeline:
    assert use_noise or use_filter, "at least one of use_noise/use_filter must be True"
    os.makedirs(lora_dir,exist_ok=True)
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe=DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo").to(device)
    noise_scheduler=pipe.scheduler
    unet:UNet2DConditionModel=pipe.unet
    unet.requires_grad_(False)

    start_epoch=1
    config_path=os.path.join(lora_dir,"epoch_config.json")
    if os.path.exists(config_path):
        with open(config_path,"r") as file:
            start_epoch=json.load(file)["epoch"]+1
        unet.load_lora_adapter(lora_dir)
    else:
        unet.add_adapter(unet_lora_config)

    # Lookup table for hook targets / filter targets — built once.
    module_dict=dict(unet.named_modules())

    CACHE="cached_activations"
    if use_filter:
        hooks=[]
        def make_hook():
            def hook_fn(module,input,output):
                out = output[0] if isinstance(output,tuple) else output
                inp = input[0]  # forward-hook input is always a tuple
                if mode=="diff":
                    setattr(module,CACHE,out-inp)
                else:
                    setattr(module,CACHE,out)
                return output
            return hook_fn
        for key in filter_dict:
            mod=module_dict.get(key)
            if mod is not None:
                hooks.append(mod.register_forward_hook(make_hook()))

    latent_h=size//8
    latent_w=size//8
    dataset=LoraDataset(image_dir,pipe.vae,device,latent_h,latent_w)
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

    optimizer=torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad],lr)

    # Empty-prompt conditioning is constant — compute once before prepare/train loop.
    with torch.no_grad():
        prompt_embeds,_,pooled_prompt_embeds,_=pipe.encode_prompt(
            prompt=" ",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    loader,unet,optimizer=accelerator.prepare(loader,unet,optimizer)

    for e in range(start_epoch,epochs+1):
        for b,batch in enumerate(loader):
            latents=batch["latent"].to(accelerator.device)*pipe.vae.config.scaling_factor
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # SDXL conditioning: text_embeds + time_ids alongside encoder_hidden_states.
            add_time_ids=torch.tensor(
                [[size,size,0,0,size,size]]*bsz,
                device=accelerator.device,
                dtype=prompt_embeds.dtype,
            )
            added_cond_kwargs={
                "text_embeds":pooled_prompt_embeds.expand(bsz,-1).to(accelerator.device),
                "time_ids":add_time_ids,
            }
            encoder_hidden_states=prompt_embeds.expand(bsz,-1,-1).to(accelerator.device)

            with accelerator.accumulate(unet):
                predicted=unet(
                    noisy_latents,timesteps,encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                loss=torch.zeros((),device=accelerator.device)
                if use_noise:
                    if use_mask:
                        m=batch["nsfw_mask"].to(accelerator.device,dtype=predicted.dtype)
                        noise_loss=F.mse_loss((predicted*m).float(),(noise*m).float())
                    else:
                        noise_loss=F.mse_loss(predicted.float(),noise.float())
                    if e==start_epoch and b==0:
                        print("noise loss",noise_loss)
                    loss=loss+noise_loss
                if use_filter:
                    z_means=[]
                    for key,sae in sae_dict.items():
                        mod=module_dict.get(key)
                        if mod is None or not hasattr(mod,CACHE):
                            continue
                        z=sae.encode(getattr(mod,CACHE))
                        z=z*filter_dict[key].to(z.device,dtype=z.dtype)
                        z_means.append(z.mean())
                    if z_means:
                        z_loss =torch.stack(z_means).mean()
                        if e==start_epoch and b==0:
                            print("z loss ",z_loss)
                        loss=loss+z_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        accelerator.unwrap_model(unet).save_lora_adapter(lora_dir)
        with open(config_path,"w") as file:
            json.dump({"epoch":e},file)
    for h in hooks:
        h.remove()
    return pipe
                
    
def sae_forward_filtered(self:SparseAutoencoder,x:torch.Tensor,weight_filter:torch.Tensor):
    x = x - self.pre_bias
    latents_pre_act = self.encoder(x) + self.latent_bias
    
    latents_pre_act=latents_pre_act*weight_filter
    
    vals, inds = torch.topk(
            latents_pre_act,
            k=self.k,
            dim=-1
        )
    
    return self.decode_sparse(inds,vals)
    
        
def main(args):
    api,accelerator,device=repo_api_init(args)
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
    y_column : str = args.y_column
    num_inference_steps : int = args.num_inference_steps
    size : int = args.size
    method : str = args.method
    image_src_dir : str = args.image_src_dir
    image_dest_dir:str=args.image_dest_dir
    n_random : int = args.n_random
    embedding_dir : str = args.embedding_dir
    sparse_embedding_dir : str = args.sparse_embedding_dir
    clip_dir : str = args.clip_dir
    clip_limit : int = args.clip_limit
    regression_limit : int = args.regression_limit
    stats_dir : str = args.stats_dir
    start_layer:int=args.start_layer
    stop_layer:int=args.stop_layer
    disable_get_images:bool=args.disable_get_images
    disable_extract_vanilla:bool=args.disable_extract_vanilla
    disable_sparsify_embeddings:bool=args.disable_sparsify_embeddings
    disable_clip_attribution:bool=args.disable_clip_attribution
    disable_run_regression:bool=args.disable_run_regression
    disable_train_lora:bool=args.disable_train_lora
    aesthetic_prompt:bool=args.aesthetic_prompt
    nsfw_prompt:bool=args.nsfw_prompt
    random_prompt:bool=args.random_prompt
    lora_epochs:int=args.lora_epochs
    lora_use_mask:bool=args.lora_use_mask
    lora_use_filter:bool=args.lora_use_filter
    lora_use_noise:bool=args.lora_use_noise
    lora_dir:str=args.lora_dir
    start_step:int=args.start_step
    end_step:int=args.end_step
    mode:str=args.mode
    out:str=args.out
    err:str=args.err
    path_set=out.split("/")
    for n in range(1,len(path_set)-1):
        new_path=os.path.join(*out[:n])
        os.makedirs(new_path,exist_ok=True)
    
    #sys.stderr=open(err,"w")
    #sys.stdout=open(out,"w")

    block_list=[
        "down_blocks.2.attentions.1",
        "mid_block.attentions.0",
        "up_blocks.0.attentions.0",
         "up_blocks.0.attentions.1"
    ]
    if not disable_get_images:
        get_images(image_dest_dir,method,n_random,size,num_inference_steps,aesthetic_prompt,nsfw_prompt,random_prompt)
    if not disable_extract_vanilla:
        extract_vanilla(embedding_dir,image_dest_dir,limit,size,mixed_precision)
    if not disable_sparsify_embeddings:
        sparsify_embeddings(sparse_embedding_dir,embedding_dir,mode)
    if not disable_clip_attribution:
        clip_attribution(image_dest_dir,clip_dir,clip_limit,use_grad=True)
    
    if not disable_run_regression:
        filter_dict={}
        for block in block_list:
            save_path=run_regression(block,y_column,regression_limit,clip_dir,os.path.join(stats_dir,block),"fp16",2,epochs) #use sparse dir for now; in the future only use clip_dir
            print(save_path)
            weights_dict=torch.load(save_path)["model_state_dict"]
            print(type(weights_dict))
            print(len(weights_dict))
            print([k for k in weights_dict])
            sparse_filter=weights_dict[[k for k in weights_dict][0]]
            filter_dict[block]=sparse_filter
            
    if not disable_train_lora:
        train_lora(lora_dir,4,device,lora_epochs,image_dest_dir,2,accelerator,0.0001,filter_dict,sae_dict,lora_use_mask,lora_use_filter,lora_use_noise,size)
            
        #run_regression(block,y_column,regression_limit,clip_dir,stats_dir)
    #load regression means, covariance matrix for each layer
    sae_dict={} #load saes for each layer
    
    #load pipeline
    dtype={
        "fp16":torch.float16,
        "no":torch.float32
    }[mixed_precision]
    
    pipe = HookedStableDiffusionXLWithUNetPipeline.from_pretrained(
            'stabilityai/sdxl-turbo',
            torch_dtype=dtype,
            device_map="balanced",
            variant=("fp16" if dtype==torch.float16 else None)
        )
    
    
    #generate images and edit them somehow (SAEURON just negates bad concepts)
    #get list of prompts (different for aesthetic vs punsafe)
    nsfw_model=get_nsfw_model()
    aesthetic_model=get_aesthetic_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    hook_list=[]
    
    #TODO: naively find "bad features" and delete them saeuron style
    COUNTER="step_counter"
    def hookify(unet:UNet2DConditionModel,
                sae_dict:dict[str,SparseAutoencoder],
                mode:str,
                start_step:int,
                end_step:int,
                filter_dict:dict[str,torch.Tensor])->list[torch.nn.Module]:
        SAE_PRETRAINED="cached_sae"
        
        START_STEP="timestep_start"
        END_STEP="timestep_end"
        WEIGHT_FILTER="sae_latent_weight_vector"
        module_dict=dict(unet.named_modules())
        def make_hook():
            def hook_fn(module,input,output):
                step=getattr(module,COUNTER)
                if step>=start_step and step<=end_step:
                    out = output[0] if isinstance(output,tuple) else output
                    inp = input[0] if isinstance(input,tuple) else input  # forward-hook input is always a tuple
                    if mode=="diff":
                        out=out-inp
                    sae:SparseAutoencoder=getattr(module,SAE_PRETRAINED)
                    out=sae_forward_filtered(sae,out,getattr(module,WEIGHT_FILTER))
                    output = (out, *output[1:]) if isinstance(output,tuple) else out
                setattr(module,COUNTER,step+1)
                return output
            return hook_fn
        hooks=[]
        mods=[]
        for key,sae in sae_dict.items():
            mod=module_dict.get(key)
            if mod is not None:
                hooks.append(mod.register_forward_hook(make_hook()))
                setattr(mod,SAE_PRETRAINED,sae)
                setattr(mod,WEIGHT_FILTER,filter_dict[key])
                mods.append(mod)
        print(f"registered {len(hooks)} hooks")
        return mods
                

    with open("unsafe.csv","r") as file:
        bad_image_list=[]
        good_image_list=[]
        reader=csv.DictReader(file)
        for i,row in enumerate(reader):
            prompt=row["prompt"]

            rand_gen=torch.Generator()
            rand_gen.manual_seed(i)
            bad_image=pipe(prompt,size,size,generator=rand_gen).images[0]
            bad_image_list.append(bad_image)
            
        mod_list=hookify(pipe.unet,sae_dict,mode,start_step,end_step)
        for i,row in enumerate(reader):
            prompt=row["prompt"]

            rand_gen=torch.Generator()
            rand_gen.manual_seed(i)
            for mod in mod_list:
                setattr(mod,COUNTER,0)
            good_image=pipe(prompt,size,size,generator=rand_gen).images[0]
            good_image_list.append(good_image)
        
        
            
        
        
        # add hooks for editing activations

if __name__=='__main__':
    print_args(parser)
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