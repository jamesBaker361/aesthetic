'''
this is the script used to find the top k images for a particular sparse feature for a particular
layer
'''


import torch
from sdxl_unbox.SAE import SparseAutoencoder
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import time
import heapq
from concurrent.futures import ThreadPoolExecutor
from experiment_helpers.image_helpers import concat_images_horizontally,concat_images_vertically
from experiment_helpers.gpu_details import print_details

def top_n_mask(x, n, dim=-1):
    """
    Build a 1.0/0.0 mask that is 1.0 at the n largest entries of `x` along `dim`.
    """
    values, indices = torch.topk(x, n, dim=dim)

    mask = torch.zeros_like(x)
    mask.scatter_(dim, indices, 1.0)

    return mask, indices


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


def sparsify_embeddings(sparse_dest_dir:str="sparse_embeddings",embedding_src_dir:str="embeddings",mode:str="diff"):
    print("sparsify embeddings")
    saes_dict:dict[str,SparseAutoencoder] = {}
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
                if mode=="diff":
                    x=torch.tensor(output_data-input_data).squeeze(0).permute(1,2,0)
                elif mode=="out":
                    x=torch.tensor(output_data).squeeze(0).permute(1,2,0)
                if torch.isnan(x).any():
                    print("nan x ",new_path)
                features=sae.encode(x)
                if torch.isnan(features).any():
                    print("nan features ",new_path)
                features=features.cpu() #-means_dict[block].cpu()
                result[block]=features.cpu().detach().numpy()
        np.savez(new_path,**result)
        
        
def get_top_k_images(block:str,
                     index:int,
                     k:int=10,
                     sparse_dest_dir:str="sparse_embeddings",
                     image_src_dir:str= "artificial_nsfw",
                     extension:str="jpeg",
                     limit:int=1_000_000)->list[Image.Image]:
    files = [f for f in os.listdir(image_src_dir) if f.endswith(extension)]
    if limit>=0:
        files=files[:limit]
        
    print(f"found {len(files)} images in {image_src_dir}")

    def load_score(file):
        npz_path = os.path.join(sparse_dest_dir, file.replace(extension, ".npz"))
        if not os.path.exists(npz_path):
            npz_path = os.path.join(sparse_dest_dir, file + ".npz")
        if not os.path.exists(npz_path):
            return None
        npz_dict = np.load(npz_path)
        sparse_embedding = npz_dict[block]  # (h, w, num_features)
        return float(np.max(sparse_embedding[..., index])), file
    
    print(f"found {len(files)} images in {image_src_dir}")
    file=files[0]
    npz_path = os.path.join(sparse_dest_dir, file.replace(extension, ".npz"))
    if not os.path.exists(npz_path):
        npz_path = os.path.join(sparse_dest_dir, file + ".npz")
    print(f"{npz_path} might exist")
    if not os.path.exists(npz_path):
        print(f"{npz_path} does not exist")
    else:
        print(f"{npz_path} definitelty exists")
        npz_dict = np.load(npz_path)
        sparse_embedding = npz_dict[block]
        print("sparae embedding shape ",sparse_embedding.shape)
    

    heap = []  # min-heap of (score, file), size <= k
    with ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(load_score, files), total=len(files), desc="Scoring"):
            if result is None:
                continue
            score, file = result
            if len(heap) < k:
                heapq.heappush(heap, (score, file))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, file))

    heap.sort(reverse=True)
    return [Image.open(os.path.join(image_src_dir, f[1])).resize((256, 256)) for f in heap]

if __name__=="__main__":
    print_details()
    """
    down_blocks.2.attentions.1/nsfw: 169962 patches kept (threshold=0.9), mean r2=0.0000 max r2=0.0058
statistics/down_blocks.2.attentions.1/regression_down_blocks.2.attentions.1_nsfw.npz
block down_blocks.2.attentions.1 tensor([ 245, 4001, 4960, 1973, 2586, 3490, 1563, 1648, 2378, 5111])
run regression
len file list 6537
mid_block.attentions.0/nsfw: 169962 patches kept (threshold=0.9), mean r2=0.0000 max r2=0.0007
statistics/mid_block.attentions.0/regression_mid_block.attentions.0_nsfw.npz
block mid_block.attentions.0 tensor([4589, 3454,  661, 2928,  242, 4528, 1127, 4338,  983, 4880])
run regression
len file list 6537
up_blocks.0.attentions.0/nsfw: 169962 patches kept (threshold=0.9), mean r2=0.0000 max r2=0.0071
statistics/up_blocks.0.attentions.0/regression_up_blocks.0.attentions.0_nsfw.npz
block up_blocks.0.attentions.0 tensor([4856, 1991, 4398, 4746,  127, 3985,  572, 1744, 4751,  487])
run regression
len file list 6537
up_blocks.0.attentions.1/nsfw: 169962 patches kept (threshold=0.9), mean r2=0.0000 max r2=0.0044
statistics/up_blocks.0.attentions.1/regression_up_blocks.0.attentions.1_nsfw.npz
block up_blocks.0.attentions.1 tensor([4052, 1888, 3397, 2837, 1861, 3653,  980,  825, 3347, 5106])
    """
    
    
    block_list=[
        "mid_block.attentions.0","down_blocks.2.attentions.1",
        "up_blocks.0.attentions.0","up_blocks.0.attentions.1"
    ]
    feature_list=[
        [4589, 3454,  661, 2928,  242, 4528],
        [ 245, 4001, 4960, 1973, 2586,3490],
        [4856, 1991, 4398, 4746,  127, 3985],
        [4052, 1888, 3397, 2837, 1861, 3653]
    ]
    
    for k,(feature_list, block) in enumerate(zip(feature_list,block_list)):
        big_img_list=[]
        for f in feature_list:
            img_list=get_top_k_images(block,f,5,limit=-1)
            img=concat_images_horizontally([i.resize((256,256)) for i in img_list ])
            big_img_list.append(img)
        concat_images_vertically(big_img_list).save(f"sparse_{block}.png")
    print('all done')
            