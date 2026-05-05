import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.argprint import print_args
from diffusers import DiffusionPipeline
import torch
import numpy as np
import csv

import time
import torch.nn.functional as F
from datasets import load_dataset

from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init
from sdxl_pipe import HookedStableDiffusionXLWithUNetPipeline
import random
import nltk
from nltk.corpus import wordnet as wn
from sdxl_extract import extract_vanilla
from sparsify import sparsify_embeddings
from regression import run_regression,clip_attribution


parser=default_parser()
UNTRAINED="untrained"

parser.add_argument("--y_column",type=str,default="aesthetic") #column 0 = aesthetic column = 1 = p(unsafe)
parser.add_argument("--num_inference_steps",type=int,default=8)
parser.add_argument("--size",type=int,default=512)
parser.add_argument("--method",type=str,default=UNTRAINED)
parser.add_argument("--image_src_dir",type=str,default="laion")
parser.add_argument("--n_random",type=int,default=50)
parser.add_argument("--embedding_dir",type=str,default="embeddings")
parser.add_argument("--sparse_embedding_dir",type=str,default="sparse_embeddings")
parser.add_argument("--clip_dir",type=str,default="clip_sparse_embeddings")
parser.add_argument("--clip_limit",type=int,default=-1)
parser.add_argument("--regression_limit",type=int,default=-1)
parser.add_argument("--stats_dir",type=str,default="statistics")
#def clip_attribution(image_src_dir:str,dest_dir:str,limit:int):
# def run_regression(block:str,y_column:str,limit:int,clip_src_dir:str,stats_dest_dir:str):
# generate images using RL or prompts
# sdxl extract 
# extract to -> sparse
# regress scores on activations
# 



def get_images(image_dest_dir:str,method:str,n_random:int,size:int,num_inference_steps:int):
    
    prompt_list=[row["prompt"] for row in load_dataset("AIML-TUDA/i2p", split="train")]+[row["prompt"] for row in load_dataset("moonworks/lunara-aesthetic", split="train")]
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
        pass
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
        diff_image=diff_pipe(prompt,height=size,width=size,generator=generator,num_inference_steps=num_inference_steps).images[0]
        diff_path=f"{image_dest_dir}/diff_{p}.jpg"
        base_image.save(base_path)
        diff_image.save(diff_path)



            
        
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
    n_random : int = args.n_random
    embedding_dir : str = args.embedding_dir
    sparse_embedding_dir : str = args.sparse_embedding_dir
    clip_dir : str = args.clip_dir
    clip_limit : int = args.clip_limit
    regression_limit : int = args.regression_limit
    stats_dir : str = args.stats_dir
    
    block_list=[
        "down_blocks.2.attentions.1",
        "mid_block.attentions.0",
        "up_blocks.0.attentions.0",
         "up_blocks.0.attentions.1"
    ]
    
    #get_images(image_src_dir,method,n_random,size,num_inference_steps)
    #extract_vanilla(embedding_dir,image_src_dir,limit,size,mixed_precision)
    #sparsify_embeddings(sparse_embedding_dir,embedding_dir)
    #clip_attribution(image_src_dir,clip_dir,clip_limit,use_grad=True)
    for block in block_list:
        dim=5000 #idr but itll break and tell us
        run_regression(block,dim,y_column,regression_limit,clip_dir,stats_dir+block,"fp16",2,10) #use sparse dir for now; in the future only use clip_dir
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
    with open("unsafe.csv","r") as file:
        reader=csv.DictReader(file)
        
    for row in reader:
        prompt=row["prompt"]
        
        rand_gen=torch.Generator()
        rand_gen.seed(123)
        
        bad_image=pipe(prompt,size,size,generator=rand_gen).images[0]
        
        mask=None
        if mask=="clip":
            #https://arxiv.org/abs/2210.04610 Red-Teaming the Stable Diffusion Safety Filter
            #https://arxiv.org/abs/2502.18816  Grad-ECLIP: Gradient-based Visual and Textual Explanations for CLIP
            pass
        elif mask=="clip_surgery":
            # https://arxiv.org/abs/2304.05653  CLIP Surgery
            # get some bad text prompts- find where they are MOST activated in image and zero out any bad sparse features like SAEUron
            pass
        else:
            pass
            
        
        
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