import torch
from sdxl_unbox.SAE import SparseAutoencoder
import os
import numpy as np
from tqdm import tqdm


block_list=[
        "down_blocks.2.attentions.1",
        "mid_block.attentions.0",
        "up_blocks.0.attentions.0",
         "up_blocks.0.attentions.1"
    ]

saes_dict:dict[str,SparseAutoencoder] = {}
means_dict = {}
path_to_checkpoints = './sdxl_unbox/checkpoints/'

for block in tqdm(block_list, desc="Loading SAEs"):
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

for file in tqdm(os.listdir(src_dir), desc="Sparsifying"):
    if not file.endswith(".npz"):
        continue
    new_path=os.path.join(dest_dir,file)
    if os.path.exists(new_path):
        continue
    with np.load(os.path.join(src_dir,file)) as data:
        result={}
        for block in block_list:
            sae=saes_dict[block]
            input_data=data["saved_input."+block]
            output_data=data["saved_output."+block]
            x=torch.tensor(output_data-input_data).squeeze(0).permute(1,2,0).flatten(0,1)
            
            features=sae.encode(x)
            features=features.cpu()-means_dict[block].cpu()
            result[block]=features.cpu().detach().numpy()
    np.savez(new_path,**result)