# this trains the sae and then does SAEURON stuyle removal I think?

import os
import json
import random
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.argprint import print_args
from experiment_helpers.image_helpers import concat_images_horizontally, concat_images_vertically
from diffusers import DiffusionPipeline,UNet2DConditionModel,AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from sdxl_unbox.SAE import SparseAutoencoder
from diffusers import SanaSprintPipeline,Krea2Pipeline
from huggingface_hub import snapshot_download,hf_hub_download
import torch
import torch.nn.functional as F
import numpy as np
import csv
import sys
import shutil
import time
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score
from generate_clean_patch import get_or_make_partition

from PIL import Image
from torchvision.transforms import v2

import saev.src.saev.data.models
import saev.src.saev.data.shards
import saev.src.saev.nn
import saev.src.saev.viz

from sam3_repo.sam3.model_builder import build_sam3_image_model
from sam3_repo.sam3.model.sam3_image_processor import Sam3Processor

from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init
parser=default_parser(
    {
        "repo_id":"jlbaker361/nsfw"
    }
)

parser.add_argument("--image_src_dir",type=str,default="real_fruit")
parser.add_argument("--embedding_dir",type=str,default="embeddings")
parser.add_argument("--sparse_embedding_dir",type=str,default="sparse_embeddings")
parser.add_argument("--mask_dir",type=str,default="mask_dir")
parser.add_argument("--eval_dir",type=str,default="eval_dir")
parser.add_argument("--npz_dict",type=str,default="platonic.npz")

parser.add_argument("--prompt_file",type=str,default="prompts.txt")

parser.add_argument("--query_list",nargs="*",default=[])

parser.add_argument("--partition_path",type=str,default="partition.json")
parser.add_argument("--train_frac",type=float,default=0.8)
parser.add_argument("--seed",type=int,default=42)

def main(args):
    api,accelerator,device=repo_api_init(args)

    image_src_dir:str=args.image_src_dir
    embedding_dir:str=args.embedding_dir
    sparse_embedding_dir:str=args.sparse_embedding_dir
    mask_dir:str=args.mask_dir
    eval_dir:str=args.eval_dir

    for d in [image_src_dir,embedding_dir,sparse_embedding_dir,mask_dir,eval_dir]:
        os.makedirs(d,exist_ok=True)

    target_query:str=args.target_query

    partition=get_or_make_partition(image_src_dir,args.partition_path,args.train_frac,args.seed)
    train_images,test_images=partition["train"],partition["test"]
    all_images=train_images+test_images
    print(f"partition: {len(train_images)} train images, {len(test_images)} test images")


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
