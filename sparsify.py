import torch
from sdxl_unbox.SAE import SparseAutoencoder
import os
import numpy as np


block_list=[
        "down_blocks.2.attentions.1",
        "mid_block.attentions.0",
        "up_blocks.0.attentions.0",
         "up_blocks.0.attentions.1"
    ]

saes_dict:dict[str,SparseAutoencoder] = {}
means_dict = {}
path_to_checkpoints = './sdxl_unbox/checkpoints/'

for block in block_list:
    sae = SparseAutoencoder.load_from_disk(
        os.path.join(path_to_checkpoints, f"unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final"),
    )
    means = torch.load(
        os.path.join(path_to_checkpoints, f"unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final", "mean.pt"),
        weights_only=True
    )
    
    saes_dict[block]=sae
    means_dict[block]=means
    
dest_dir="sparse_embeddings"
os.makedirs(dest_dir,exist_ok=True)
src_dir="embeddings"

for f,file in enumerate( os.listdir(src_dir)):
    if not file.endswith(".npz"):
        continue
    new_path=os.path.join(dest_dir,file)
    if os.path.exists(new_path):
        continue
    with np.load(os.path.join(src_dir,file)) as data:
        result={}
        for block in block_list:
            sae=saes_dict[block]
            x=torch.tensor(data[block])[0].transpose(1,2,0).flatten(0,1)
            features=sae.encode(x)
            result[block]=features.cpu().detach().numpy()
    np.savez(new_path,**result)
            
            
            
            