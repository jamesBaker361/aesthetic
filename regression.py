import numpy as np
import argparse
import os
import csv
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from experiment_helpers.argprint import print_args
from sklearn.linear_model import Ridge,LinearRegression,ElasticNet,Lasso
from rewards import get_nsfw_model,get_aesthetic_model
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

from attribution import get_maps, get_importance, clip_attribution

# For each of the `dim` SAE/UNet features in `block`, fit its OWN univariate
# regression y=a*x+b (closed-form OLS, not gradient descent - there's no joint
# model here, every feature gets an independent single-variable fit) against a
# per-patch target, pooled over every patch (across every image) whose
# clip_attribution importance quantile is >= threshold. weight_by_importance
# picks the target: the whole image's y_column score, or that score scaled by
# the patch's own importance quantile. Saves a, b, r2 and the signed Pearson
# correlation r (= sign(a)*sqrt(r2), scale-invariant, used downstream in
# generate_clean.py to rank features by correlation with y_column) per feature.
def run_regression(block:str,y_column:str,
                   limit:int,clip_src_dir:str,
                   stats_dest_dir:str,
                   threshold:float,
                   weight_by_importance:bool):
    print("run regression")
    score_key=f"{block}.{y_column}"
    image_score_key=f"image_{y_column}_score"
    os.makedirs(stats_dest_dir,exist_ok=True)
    save_path=os.path.join(stats_dest_dir,f"regression_{block}_{y_column}.npz")
    if os.path.exists(save_path):
        return save_path

    file_list=[
        os.path.join(clip_src_dir,f)
        for f in os.listdir(clip_src_dir)
        if f.endswith("npz")
    ]
    
    if limit>=0:
        file_list=file_list[:limit]
    print("len file list", len(file_list))
    # streamed first/second-moment accumulation (per feature) over every kept
    # patch, so we never have to hold every image's patches in memory at once
    x_sum=x_sq_sum=xy_sum=None
    y_sum=y_sq_sum=0.0
    count=0
    for file in file_list:
        with np.load(file) as data:
            if block not in data or score_key not in data or image_score_key not in data:
                continue
            X=data[block].reshape(-1,data[block].shape[-1])   # [patches, dim]
            quantile=data[score_key].reshape(-1)                # [patches], in [0,1]
            image_score=float(data[image_score_key])

        keep=quantile>=threshold
        if not keep.any():
            continue
        X=X[keep]
        y=np.full(X.shape[0],image_score,dtype=np.float64)
        if weight_by_importance:
            y=y*quantile[keep]

        if x_sum is None:
            dim=X.shape[-1]
            x_sum=np.zeros(dim);x_sq_sum=np.zeros(dim);xy_sum=np.zeros(dim)
        x_sum+=X.sum(axis=0)
        x_sq_sum+=(X**2).sum(axis=0)
        xy_sum+=(X*y[:,None]).sum(axis=0)
        y_sum+=y.sum()
        y_sq_sum+=(y**2).sum()
        count+=X.shape[0]

    if count==0:
        raise ValueError(f"no patches for block={block} passed threshold={threshold}")

    x_mean=x_sum/count
    x_var=x_sq_sum/count-x_mean**2
    y_mean=y_sum/count
    y_var=y_sq_sum/count-y_mean**2
    cov=xy_sum/count-x_mean*y_mean

    a=cov/(x_var+1e-12)
    b=y_mean-a*x_mean
    r2=(cov**2)/(x_var*y_var+1e-12)
    r=np.sign(a)*np.sqrt(r2)

    print(f"{block}/{y_column}: {count} patches kept (threshold={threshold}), mean r2={r2.mean():.4f} max r2={r2.max():.4f}")

    
    np.savez(save_path,a=a,b=b,r2=r2,r=r)
    return save_path

        
def run_top_k_features_popularity_contest(block:str,y_column:str,
                         clip_src_dir:str,
                         image_test_dir:str,
                         image_src_dir:str="artificial_images",
                         
                         limit:int=-1,
                         quantile_threshold: float=0.95,
                         
                         k:int=10):
    print("run_top_k_features_popularity_contest")
    score_key=f"{block}.{y_column}"
    image_score_key=f"image_{y_column}_score"

    file_list=[
        os.path.join(clip_src_dir,f)
        for f in os.listdir(clip_src_dir)
        if f.endswith("npz")
    ]

    if limit>=0:
        file_list=file_list[:limit]
    print("len file list", len(file_list))

    nsfw_count_dict=defaultdict(lambda: 0)
    sfw_count_dict=defaultdict(lambda: 0)

    os.makedirs(f"{image_test_dir}/gradient",exist_ok=True)
    n_grad_to_save=10
    grad_saved=0

    nsfw_count =0
    sfw_count=0

    for file in file_list:
        with np.load(file) as data:
            if block not in data:
                print(f"block {block} not in data") 
                continue
            elif score_key not in data:
                print(f"score_key {score_key} not in data")
            elif image_score_key not in data:
                print(f"image score key {image_score_key} not in data")
                continue

            score=data[image_score_key]
            if score >0.8:
                

                # only take the patches whose per-patch importance quantile is
                # in the top quantile_threshold (quantile in [0,1], see
                # get_maps), then rank those kept patches' features by max
                # activation - mirrors the sfw branch below but restricted to
                # the most important patches instead of every patch
                quantiles=data[score_key].reshape(-1)
                keep=quantiles>=quantile_threshold
                if keep.any():
                    kept_features=data[block].reshape(-1,data[block].shape[-1])[keep]
                    feature_max=kept_features.max(axis=0)
                    indices=np.argsort(feature_max)[::-1][:k]
                    for index in indices:
                        nsfw_count_dict[index]+=1
                        
                if grad_saved<n_grad_to_save:
                    # overlay this image's per-patch quantile map (already in
                    # [0,1], see clip_attribution) on the original image, same
                    # nearest-neighbor upsample + COLORMAP_BONE style used
                    # elsewhere in this file/sparsify.py, so the ranking used
                    # below can be sanity-checked visually
                    orig_filename=os.path.basename(file)
                    if orig_filename.endswith(".npz"):
                        orig_filename=orig_filename[:-len(".npz")]
                    orig_path=os.path.join(image_src_dir,orig_filename)
                    if os.path.exists(orig_path):
                        img_np=np.array(Image.open(orig_path).convert("RGB"))
                        img_h,img_w=img_np.shape[:2]

                        heatmap=cv2.resize(data[score_key].astype(np.float32),(img_w,img_h),interpolation=cv2.INTER_NEAREST)
                        heatmap=np.clip(heatmap,0,1)**0.5
                        heatmap_uint8=np.uint8(255*heatmap)
                        heatmap_color=cv2.applyColorMap(heatmap_uint8,cv2.COLORMAP_BONE)
                        heatmap_color=cv2.cvtColor(heatmap_color,cv2.COLOR_BGR2RGB)

                        overlay=cv2.addWeighted(img_np,0.6,heatmap_color,0.4,0)
                        Image.fromarray(np.uint8(255-overlay)).save(
                            os.path.join(image_test_dir,"gradient",f"{block}_{y_column}_{orig_filename}")
                        )

                        # same overlay, but zero out every patch below
                        # quantile_threshold first, so only the patches that
                        # would actually pass the keep filter above show up
                        thresholded=np.where(data[score_key]>=quantile_threshold,data[score_key],0.0)
                        heatmap=cv2.resize(thresholded.astype(np.float32),(img_w,img_h),interpolation=cv2.INTER_NEAREST)
                        heatmap=np.clip(heatmap,0,1)**0.5
                        heatmap_uint8=np.uint8(255*heatmap)
                        heatmap_color=cv2.applyColorMap(heatmap_uint8,cv2.COLORMAP_BONE)
                        heatmap_color=cv2.cvtColor(heatmap_color,cv2.COLOR_BGR2RGB)

                        overlay=cv2.addWeighted(img_np,0.6,heatmap_color,0.4,0)
                        Image.fromarray(np.uint8(255-overlay)).save(
                            os.path.join(image_test_dir,"gradient",f"{block}_{y_column}_thresholded_{orig_filename}")
                        )

                        grad_saved+=1
                    nsfw_count+=1
            
            if score < 0.75:
                sfw_count+=1
                
                all_features=data[block].reshape(-1,data[block].shape[-1])
                for feature in all_features:
                    indices=np.argsort(feature)[::-1][:k]
                    for index in indices:
                        sfw_count_dict[index]+=1

    print(sorted(nsfw_count_dict.items(), key=lambda x: x[1],reverse=True)[:10])
    print(sorted(sfw_count_dict.items(), key=lambda x: x[1],reverse=True)[:10])
    
    print("nsfw count",nsfw_count,nsfw_count/len(file_list))
    print("sfw count", sfw_count,sfw_count/len(file_list))

    nsfw_count_dict={key:value/nsfw_count for key,value in nsfw_count_dict.items()}
    sfw_count_dict={key:value/sfw_count for key,value in sfw_count_dict.items()}
    
    
    relative_dict={}
    
    never_present=[]
    
    for key,value in nsfw_count_dict.items():
        denominator=1e-8
        if key in sfw_count_dict:
            denominator=sfw_count_dict[key]
        else:
            never_present.append(key)
        relative_dict[key]=value/denominator

    print("never present ",never_present)
    
    return dict(sorted(relative_dict.items(), key=lambda x: x[1],reverse=True)),sfw_count_dict
            
            




if __name__=="__main__":
    n=20
    os.makedirs("maps",exist_ok=True)
    nsfw_model=get_nsfw_model()
    aesthetic_model=get_aesthetic_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_list=[]
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    for k,file in enumerate([f for f in os.listdir("artificial_nsfw") if f.endswith("jpeg")][:n]):
        path=os.path.join("artificial_nsfw", file)
        img=Image.open(path)
        concat,_,score=get_maps(img,nsfw_model,
                        aesthetic_model,
                        device,
                        processor,
                        clip_model)
        print(k,score)
        img_list.append(concat)
        concat.save(f"maps/heat_{k}.png")

        print("all done!")