import torch
from sdxl_unbox.SAE import SparseAutoencoder
import os
import numpy as np
from tqdm import tqdm
from PIL import Image



block_list=[
        "down_blocks.2.attentions.1",
        "mid_block.attentions.0",
        "up_blocks.0.attentions.0",
         "up_blocks.0.attentions.1"
    ]


path_to_checkpoints = './sdxl_unbox/checkpoints/'

sparse_dest_dir="sparse_embeddings"
os.makedirs(sparse_dest_dir,exist_ok=True)
embedding_src_dir="embeddings"
image_src_dir= "laion"


def sparsify_embeddings():
    saes_dict:dict[str,SparseAutoencoder] = {}
    means_dict = {}
    for block in tqdm(block_list, desc="Loading SAEs"):
        sae = SparseAutoencoder.load_from_disk(
            os.path.join(path_to_checkpoints, f"unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final"),
        )
        if torch.isnan(sae.decoder.weight).any():
            print("nan decoder weight ",block)
        means = torch.load(
            os.path.join(path_to_checkpoints, f"unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final", "mean.pt"),
            weights_only=True
        )
        
        if torch.isnan(means).any():
            print(" nan mean for ",block)
        
        saes_dict[block]=sae
        means_dict[block]=means
        


    for file in tqdm(os.listdir(embedding_src_dir), desc="Sparsifying"):
        if not file.endswith(".npz"):
            continue
        new_path=os.path.join(sparse_dest_dir,file)
        if os.path.exists(new_path):
            continue
        with np.load(os.path.join(embedding_src_dir,file)) as data:
            result={}
            for block in block_list:
                sae=saes_dict[block]
                input_data=data["saved_input."+block]
                output_data=data["saved_output."+block]
                x=torch.tensor(output_data-input_data).squeeze(0).permute(1,2,0).flatten(0,1)
                if torch.isnan(x).any():
                    print("nan x ",new_path)
                features=sae.encode(x)
                if torch.isnan(features).any():
                    print("nan features ",new_path)
                features=features.cpu()-means_dict[block].cpu()
                result[block]=features.cpu().detach().numpy()
        np.savez(new_path,**result)
        
        
def get_top_k_images(block:str,index:int,k:int=10)->Image.Image:
    rankings=[]
    for file in [f for f in os.listdir(image_src_dir) if f.endswith("jpg")]:
        new_path=os.path.join(sparse_dest_dir,file)
        if os.path.exists(new_path):
            npz_dict=np.load(new_path)
            sparse_embedding=npz_dict[block]
            print(sparse_embedding.shape)
            features=sparse_embedding[:,index]
            print(features.shape)
            largest=max(features)
            rankings.append([largest,file])
            
    rankings.sort(key=lambda x:-x[0])
    rankings=rankings[:k]
    return [Image.open(os.path.join(f[1])) for f in rankings]

if __name__=="__main__":
    get_top_k_images("down_blocks.2.attentions.1",0)
            