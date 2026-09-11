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
import json
import cv2
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
        
        
def _get_top_k_heap(block:str,
                    index:int,
                    k:int,
                    sparse_dest_dir:str,
                    image_src_dir:str,
                    cache_dir:str,
                    extension:str,
                    limit:int)->list[tuple[float,str]]:
    '''
    Scores every image in image_src_dir on feature `index` of `block` and
    returns the top-k (score, file) pairs, sorted descending. Cached to disk
    under cache_dir since the scoring pass (loading every npz in
    sparse_dest_dir) is the expensive part and is identical for
    get_top_k_images and get_top_k_images_highlighted.
    '''
    os.makedirs(cache_dir,exist_ok=True)
    cache_path=os.path.join(cache_dir,f"{block}.{index}.k{k}.limit{limit}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return [tuple(pair) for pair in json.load(f)]

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

    with open(cache_path,"w") as f:
        json.dump(heap,f)

    return heap

def get_top_k_images(block:str,
                     index:int,
                     k:int=10,
                     sparse_dest_dir:str="sparse_embeddings",
                     image_src_dir:str= "artificial_nsfw",
                     cache_dir:str="feature_cache",
                     extension:str="jpeg",
                     limit:int=1_000_000)->list[Image.Image]:
    heap=_get_top_k_heap(block,index,k,sparse_dest_dir,image_src_dir,cache_dir,extension,limit)
    return [Image.open(os.path.join(image_src_dir, f[1])).resize((256, 256)) for f in heap]

def get_top_k_images_highlighted(block:str,
                                 index:int,
                                 k:int=10,
                                 sparse_dest_dir:str="sparse_embeddings",
                                 image_src_dir:str="artificial_nsfw",
                                 cache_dir:str="feature_cache",
                                 extension:str="jpeg",
                                 limit:int=1_000_000,
                                 size:int=256)->list[Image.Image]:
    '''
    Same top-k selection as get_top_k_images, but each returned image has the
    `index` feature's per-patch activation overlaid as a heatmap (same style as
    the nsfw/aesthetic importance maps in regression.py's get_maps), so you can
    see where in the image that feature fires, not just which images score highest.
    '''
    heap=_get_top_k_heap(block,index,k,sparse_dest_dir,image_src_dir,cache_dir,extension,limit)

    highlighted=[]
    for score,file in heap:
        img=Image.open(os.path.join(image_src_dir, file)).convert("RGB").resize((size, size))
        img_np=np.array(img)

        npz_path = os.path.join(sparse_dest_dir, file.replace(extension, ".npz"))
        if not os.path.exists(npz_path):
            npz_path = os.path.join(sparse_dest_dir, file + ".npz")
        activation = np.load(npz_path)[block][..., index]  # (h, w)

        # Normalize to [0,1] and upsample (nearest, since each cell is a whole
        # patch) to the image size.
        activation = activation - activation.min()
        activation = activation / (activation.max() + 1e-8)
        activation = cv2.resize(activation.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)

        heatmap_uint8 = np.uint8(255 * np.clip(activation, 0, 1) ** 0.5)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_BONE)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
        highlighted.append(Image.fromarray(np.uint8(255 - overlay)))

    return highlighted

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
        "down_blocks.2.attentions.1",
        "mid_block.attentions.0",
        "up_blocks.0.attentions.0",
        "up_blocks.0.attentions.1"
    ]
    feature_list=[
        [715, 2994, 962, 518, 1836, 788, 4678],
        [ 4906, 4450, 242, 3454, 3053, 365, 3181],
        [487, 1292, 248, 572, 4137, 4751, 2977],
        [4052, 1042, 2961, 4490, 2837, 1067, 1476]
    ]
    
    for k,(feature_list, block) in enumerate(zip(feature_list,block_list)):
        big_img_list=[]
        for f in feature_list:
            img_list=get_top_k_images_highlighted(block,f,5,limit=-1)
            img=concat_images_horizontally([i.resize((256,256)) for i in img_list ])
            big_img_list.append(img)
        concat_images_vertically(big_img_list).save(f"highlighted_{block}.png")
    print('all done')
            