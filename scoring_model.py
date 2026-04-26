#use some sort of pretrained conv backbone + classification head  to learn scores from laion
# dino, siglip, vae + diffusion  imagenet etc
#  nsfw test set  https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/nsfw_testset.zip
# nsfw image data: yesidobyte/nsfw1024 https://huggingface.co/datasets/deepghs/nsfw_detect
# sfw data: laion + people dataset (oversample)

import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.argprint import print_args
import torch
from datasets import load_dataset
from diffusers.image_processor import VaeImageProcessor

import time
import torch.nn.functional as F

from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init

parser=default_parser()

parser.add_argument("--backbone",type=str)
parser.add_argument("--good_src_list",nargs="*",type=str)
parser.add_argument("--bad_src_list",nargs="*",type=str)
parser.add_argument("--size",type=int,default=256)

class NsfwDataset(torch.utils.data.Dataset):
    def __init__(self,good_src_list:list[str],bad_src_list:list[str],size:int):
        super().__init__()
        self.indices=[]
        self.total_len=0
        self.dataset_dict={}
        self.image_processor=VaeImageProcessor()
        self.size=size
        
        for dataset_name in good_src_list:
            data=load_dataset(dataset_name,split="train")
            self.dataset_dict[dataset_name]=data
            for k in range(len(data)):
                self.indices.append([dataset_name,k,0])
        
        for dataset_name in bad_src_list:
            data=load_dataset(dataset_name,split="train")
            self.dataset_dict[dataset_name]=data
            for k in range(len(data)):
                self.indices.append([dataset_name,k,1])
        

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        [dataset_name,k,label]=self.indices[index]
        img=self.dataset_dict[dataset_name][k]
        img=self.image_processor.preprocess(img,self.size,self.size)
        return {
            "image":img,
            "label":label
        }
            

def main(args):
    api,accelerator,device=repo_api_init(args)
    backbone:str=args.backbone


    dataset= NsfwDataset()
    
    train_loader,test_loader,val_loader=split_data(dataset,0.8,args.batch_size)
    
    for batch in train_loader:
        break

    save_subdir=args.save_dir
    os.makedirs(save_subdir,exist_ok=True)
    
    if backbone=="unet":
        pass
    elif backbone=="dino":
        pass
    elif backbone=="imagenet":
        pass
    elif backbone =="sapiens":
        pass

    params=None

    optimizer=torch.optim.AdamW(params,args.lr)
    
    optimizer,unet,action_encoder,train_loader,test_loader,val_loader = accelerator.prepare(optimizer,unet,action_encoder,train_loader,test_loader,val_loader)

        


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