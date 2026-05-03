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
from diffusers.image_processor import VaeImageProcessor
from d3po_rewards import get_nsfw_model,get_aesthetic_model
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from experiment_helpers.image_helpers import concat_images_horizontally
import matplotlib.pyplot as plt
import cv2

def clip_attribution(image_src_dir:str,dest_dir:str,limit:int,sparse_dir:str="sparse_embeddings"):
    #for each image find relevant patches and scores and save them
    os.makedirs(dest_dir,exist_ok=True)
    # get models
    nsfw_model=get_nsfw_model()
    aesthetic_model=get_aesthetic_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    img_pro=VaeImageProcessor()

    for n, file in enumerate([f for f in os.listdir(image_src_dir) if f.endswith("jpg")][:limit]):
        
        # --- Load image ---
        pil_img = Image.open(os.path.join(image_src_dir, file)).convert("RGB")
        og_w, og_h = pil_img.size  # NOTE: PIL = (W, H) supposedly...

        img_tensor = transforms.PILToTensor()(pil_img)  # [C,H,W]

        with torch.enable_grad():
            inputs = {k: v.to(device) for k, v in processor(images=img_tensor, return_tensors="pt").items()}
            inputs['pixel_values'].requires_grad_(True)
            outputs = clip_model(**inputs, output_hidden_states=True, output_attentions=True)

            hidden_states = outputs.hidden_states
            for t in hidden_states:
                t.retain_grad()

            last_hidden_state = outputs.last_hidden_state  # [1, 1+N, D]
            last_hidden_state.retain_grad()

            image_embeds = F.normalize(outputs.image_embeds, dim=-1)

            # --- Score (your aesthetic model or direction) ---
            score = aesthetic_model(image_embeds)
            score.backward()
        img_list=[]
        try:
            npz_dict=dict(np.load(os.path.join(sparse_dir, file.replace("jpg","npz"))))
            npz_dict["aesthetic"]=score.cpu().detach().numpy()
            npz_dict["nsfw"]=0.
            np.savez(os.path.join(dest_dir,file.replace("jpg","npz")), ** npz_dict)
        except FileNotFoundError:
            pass
        for layer_idx,target_hidden_state in enumerate(hidden_states):
            # --- Importance (Grad * Activation) ---
            grads = target_hidden_state.grad[0, 1:, :]        # remove CLS → [N, D]
            acts  = target_hidden_state[0, 1:, :]             # [N, D]
            
            num_patches = acts.shape[0]
            h = w = int(num_patches ** 0.5)
            


            importance = grads * acts                       # [N, D]
            importance = importance.norm(dim=-1)            # [N] should we sum? 

            # --- Reshape to patch grid ---
            num_patches = importance.shape[0]
            h = w = int(num_patches ** 0.5)
            importance = importance.reshape(h, w)

            # --- Normalize ---
            importance = importance - importance.min()
            importance = importance / (importance.max() + 1e-8)

            # --- Upsample to image size ---
            importance = importance.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]

            big_importance = F.interpolate(
                importance,
                size=(og_h, og_w),   # torch = (H, W)
                mode="nearest",
                #align_corners=False
            )[0, 0]

            # --- Convert for plotting ---
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy() / 255.0
            heatmap = big_importance.detach().cpu().numpy()

            # --- Optional sharpening ---
            heatmap = np.clip(heatmap, 0, 1)
            heatmap = heatmap ** 0.5
            

            # convert heatmap → color
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # convert original image
            img_uint8 = np.uint8(img_np * 255)

            # blend
            overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)
            
            
            
            
            overlay=cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            pil_img=VaeImageProcessor.numpy_to_pil(overlay)[0]
            img_list.append(pil_img)
        path=os.path.join(dest_dir,file)
        concat=concat_images_horizontally(img_list)
        concat.save(path)
        
        

def run_regression(block:str,y_column:str,limit:int,clip_src_dir:str,stats_dest_dir:str):
    pass




info_path="laion/info.csv"
sparse_dir="sparse_embeddings"
dest_dir="statistics"

os.makedirs(dest_dir,exist_ok=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--y_column",type=str,default="aesthetic") #column 0 = aesthetic column = 1 = p(unsafe)
    parser.add_argument("--block",type=str,default="down_blocks.2.attentions.1")
    parser.add_argument("--limit",type=int,default=-1)
    
    clip_attribution("test_imgs","test_maps",-1)
    
    exit(0)
    print_args(parser)
    args=parser.parse_args()
    print(args)
    indep_chunks=[]
    dependent=[]
    with open(info_path,"r") as file:
        for l,line in enumerate(tqdm(file)):
            if l==args.limit:
                break
            [imgpath,aesthetic,punsafe]=line.strip().split(",")
            imgpath=imgpath.split("/")[1]
            aesthetic=float(aesthetic)
            punsafe=float(punsafe)
            target={
                "aesthetic":aesthetic,
                "punsafe":punsafe
            }[args.y_column]
            if l<10:
                print(target)
            npz_file=os.path.join(sparse_dir,imgpath+".npz")
            if os.path.exists(npz_file):
                features=np.load(npz_file)[args.block]
                if l<10:
                    print(features.shape)
                mask=np.isfinite(features).all(axis=1)
                features=features[mask]
                if len(features):
                    indep_chunks.append(features)
                    dependent.extend([target]*len(features))
            elif l<10:
                print(npz_file,"doesnt exists")

    print(" len samples",len(dependent))

    independent=np.vstack(indep_chunks)
    del indep_chunks
    dependent=np.array(dependent)

    indep_mean = independent.mean(axis=0)
    indep_std = independent.std(axis=0)
    indep_std[indep_std == 0] = 1
    independent = (independent - indep_mean) / indep_std
    
    t0=time.time()
    covariance=np.cov(independent,rowvar=False)
    print(f"covariance: {time.time()-t0:.2f}s")
    
    independent = np.hstack([independent, np.ones((independent.shape[0], 1))])

    dep_mean = dependent.mean()
    dep_std = dependent.std()
    dependent = (dependent - dep_mean) / dep_std

    indep_train, indep_test, dep_train, dep_test = train_test_split(
        independent, dependent, test_size=0.05, random_state=42)

    for var,name in zip([indep_train, indep_test, dep_train, dep_test,independent,dependent],
                        ["indep_train", "indep_test", "dep_train", "dep_test","independent","dependent"]):
        print(name,var.shape)

    npz_dict={}
    for solver_class,name in zip(
            [LinearRegression,ElasticNet,Ridge,Lasso],
            ["LinearRegression","ElasticNet","Ridge","Lasso"]):
        model=solver_class()
        t0=time.time()
        model.fit(indep_train,dep_train)
        preds=model.predict(indep_test)
        mse=mean_squared_error(dep_test,preds)
        r2=r2_score(dep_test,preds)
        print(f"{name} {time.time()-t0:.2f}s  mse={mse:.4f}  r2={r2:.4f}")
        npz_dict[f"{name}_coef"]=model.coef_
        for key,value in model.get_params().items():
            npz_dict[f"{name}_{key}"]=value

    save_dir=os.path.join(dest_dir,args.block)
    os.makedirs(save_dir,exist_ok=True)
    np.savez(os.path.join(save_dir,args.y_column),
             covar=covariance,
             indep_mean=indep_mean,
             dep_mean=dep_mean,
             indep_std=indep_std,
             dep_std=dep_std,**npz_dict)
    