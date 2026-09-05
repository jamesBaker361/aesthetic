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
from rewards import get_nsfw_model,get_aesthetic_model
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from experiment_helpers.image_helpers import concat_images_horizontally,concat_images_vertically
import matplotlib.pyplot as plt
import cv2

def get_maps(pil_img: Image.Image,
             nsfw_model,
             aesthetic_model,
             device,
             processor,
             clip_model):
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
        #score = aesthetic_model(image_embeds)
        score=-nsfw_model(image_embeds)
        score.backward()
    img_list=[]
    try:
        pass
        npz_dict=dict(np.load(os.path.join(sparse_dir, file.replace("jpg","npz"))))
        npz_dict["aesthetic"]=score.cpu().detach().numpy()
        npz_dict["nsfw"]=0.
        np.savez(os.path.join(dest_dir,file.replace("jpg","npz")), ** npz_dict)
    except (FileNotFoundError,NameError):
        pass
    clip_grad_maps=[]
    for layer_idx,target_hidden_state in enumerate(hidden_states): # so the middle 4 layers seem to be the only not totally dogshit- maybe we should pool
        #if use_grad:
        # --- Importance (Grad * Activation) ---
        grads = target_hidden_state.grad[0, 1:, :]        # remove CLS → [N, D]
        grads=torch.nn.ReLU()(grads)
        acts  = target_hidden_state[0, 1:, :]             # [N, D]
        
        num_patches = acts.shape[0]
        h = w = int(num_patches ** 0.5)
        


        importance = grads #* acts                       # [N, D]
        #importance = torch.abs(importance).sum(dim=-1)            # [N] should we sum? 
        importance=importance.norm(dim=-1)

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
        
        clip_grad_maps.append(big_importance)

        # --- Convert for plotting ---
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        heatmap = big_importance.detach().cpu().numpy()

        # --- Optional sharpening ---
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = heatmap ** 0.5
        

        # convert heatmap → color
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_BONE)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # convert original image
        img_uint8 = np.uint8(img_np)

        # blend
        overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)

        pil_img=VaeImageProcessor.numpy_to_pil(255-overlay)[0]

        heat_map_pil=VaeImageProcessor.numpy_to_pil(255-heatmap_color)[0]
        
        big_img=concat_images_vertically([pil_img,heat_map_pil])
        
        #img_list.append(pil_img)
        
        #second importance
        
        '''grads = target_hidden_state.grad[0, 1:, :]        # remove CLS → [N, D]
        acts  = target_hidden_state[0, 1:, :]             # [N, D]
        
        num_patches = acts.shape[0]
        h = w = int(num_patches ** 0.5)
        


        importance = grads * acts                       # [N, D]
        #importance = torch.abs(importance).sum(dim=-1)            # [N] should we sum? 
        importance=importance.norm(dim=-1)'''
        
        cls=target_hidden_state[0,0, :]
        acts  = target_hidden_state[0, 1:, :]
        importance = torch.stack([torch.dot(cls, a) for a in acts])

        
        

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
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_BONE)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # convert original image
        img_uint8 = np.uint8(img_np * 255)

        # blend
        overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)

        pil_img=VaeImageProcessor.numpy_to_pil(255-overlay)[0]

        heat_map_pil=VaeImageProcessor.numpy_to_pil(255-heatmap_color)[0]
        
        pil_img=concat_images_vertically([big_img,pil_img,heat_map_pil])
        
        img_list.append(pil_img)
        
    
    concat=concat_images_horizontally(img_list)
    arr = np.array(concat)

    arr = np.ascontiguousarray(arr)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr).convert("RGB")
    
    avg_importance=torch.stack(clip_grad_maps).mean(0)
    
    heatmap = avg_importance.detach().cpu().numpy()

    # --- Optional sharpening ---
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = heatmap ** 0.5
    

    # convert heatmap → color
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_BONE)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    avg_pil=VaeImageProcessor.numpy_to_pil(255-heatmap_color)[0]
    max_importance=torch.stack(clip_grad_maps).max(dim=0).values
    heatmap = max_importance.detach().cpu().numpy()

    # --- Optional sharpening ---
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = heatmap ** 0.5
    

    # convert heatmap → color
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_BONE)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    max_pil=VaeImageProcessor.numpy_to_pil(255-heatmap_color)[0]
    
    concat2=concat_images_horizontally([avg_pil,max_pil])

    return img,concat2

def get_importance(pil_img: Image.Image,
             nsfw_model,
             aesthetic_model,
             device,
             processor,
             clip_model)->tuple[list[torch.Tensor],list[torch.Tensor],float,float]:
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

        # one forward pass, two backward passes off the same activations
        nsfw_score=nsfw_model(image_embeds)
        nsfw_score.backward(retain_graph=True)
        nsfw_grads=[t.grad.clone() for t in hidden_states]
        for t in hidden_states:
            t.grad=None

        aesthetic_score=aesthetic_model(image_embeds)
        aesthetic_score.backward()
        aesthetic_grads=[t.grad.clone() for t in hidden_states]

    def build_importance(grads_list):
        importance_list=[]
        for layer_idx,(target_hidden_state,grad) in enumerate(zip(hidden_states,grads_list)): # so the middle 4 layers seem to be the only not totally dogshit- maybe we should pool
            # --- Importance (Grad * Activation) ---
            grads = grad[0, 1:, :]        # remove CLS → [N, D]
            grads=torch.nn.ReLU()(grads)
            acts  = target_hidden_state[0, 1:, :]             # [N, D]

            importance = grads * acts                       # [N, D]
            importance=importance.norm(dim=-1)

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
            )[0, 0]
            importance_list.append(big_importance)
        return importance_list

    importance_nsfw=build_importance(nsfw_grads)
    importance_aesthetic=build_importance(aesthetic_grads)

    # Whole-image scores (scalars), needed downstream as the regression target
    # in run_regression - as opposed to the per-patch importance maps above.
    return importance_aesthetic,importance_nsfw,float(aesthetic_score.detach().cpu()),float(nsfw_score.detach().cpu())

def clip_attribution(image_src_dir:str,dest_dir:str,limit:int,
                     sparse_dir:str="sparse_embeddings",
                     start_layer=5,
                     stop_layer=15,):
    # Step 1 of the intended pipeline: rank each spatial patch by how much it
    # drives the nsfw/aesthetic score (via get_importance's grad*activation maps),
    # then convert that ranking to a [0,1] quantile per patch (see below). Also
    # stash the whole-image scores themselves - run_regression needs both: the
    # quantile to threshold/weight patches, the whole-image score as the target.
    print("clip attributuon")
    os.makedirs(dest_dir,exist_ok=True)
    # get models
    nsfw_model=get_nsfw_model()
    aesthetic_model=get_aesthetic_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    img_pro=VaeImageProcessor()

    files=[f for f in os.listdir(image_src_dir) if f.endswith("jpg")]
    if limit>=0:
        files=files[:limit]
    for n, file in enumerate(files):
        npz_file=file+".npz"
        if os.path.exists(os.path.join(sparse_dir,npz_file)):
        
            # --- Load image ---
            pil_img = Image.open(os.path.join(image_src_dir, file)).convert("RGB")
            importance_aesthetic,importance_nsfw,aesthetic_score,nsfw_score=get_importance(pil_img,nsfw_model,aesthetic_model,device,processor,clip_model)
            importance_aesthetic=importance_aesthetic[start_layer:stop_layer]
            importance_nsfw=importance_nsfw[start_layer:stop_layer]

            avg_aesthetic=torch.stack(importance_aesthetic).mean(dim=0)
            avg_nsfw=torch.stack(importance_nsfw).mean(dim=0)

            with np.load(os.path.join(sparse_dir,npz_file)) as old_npz:
                # whole-image scores, constant across all patches/blocks of this image
                save_dict={
                    "image_aesthetic_score":aesthetic_score,
                    "image_nsfw_score":nsfw_score,
                }
                for block in [
                    "down_blocks.2.attentions.1",
                    "mid_block.attentions.0",
                    "up_blocks.0.attentions.0",
                    "up_blocks.0.attentions.1"
                ]:
                    features=torch.tensor(old_npz[block])
                    (h,w,c)=features.size()
                    save_dict[block]=features.cpu().numpy()
                    for y_value,importance in zip(
                        ["nsfw","aesthetic"],
                        [avg_nsfw,avg_aesthetic]
                    ):
                        # Resize the importance map to match the spatial dimensions (h, w).
                        # Two dimensions are added first to represent batch and channel dimensions,
                        # which F.interpolate expects: [H, W] -> [1, 1, H, W].
                        resized = F.interpolate(
                            importance.unsqueeze(0).unsqueeze(0),
                            size=(h, w)
                        )[0, 0]  # Remove the batch and channel dimensions: [1, 1, h, w] -> [h, w]

                        # Flatten the 2D importance map into a 1D vector so all pixels can be ranked.
                        flat = resized.flatten()

                        # Compute the rank of every value.
                        # flat.argsort() gives the indices that would sort the values.
                        # Applying argsort() again converts those sorted indices into each
                        # element's rank, ranging from 0 (smallest) to N-1 (largest).
                        ranks = flat.argsort().argsort().float()

                        # Normalize ranks to the range [0, 1], producing the percentile/quantile
                        # of each pixel rather than using its raw importance value.
                        # max(..., 1) avoids division by zero if there is only one element.
                        quantile = (ranks / max(flat.numel() - 1, 1)).reshape(h, w)
                        save_dict[f"{block}.{y_value}"]=quantile.cpu().numpy()

            np.savez(os.path.join(dest_dir,npz_file), **save_dict)

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
        






if __name__=="__main__":
    info_path="laion/info.csv"
    sparse_dir="sparse_embeddings"
    dest_dir="statistics"

    os.makedirs(dest_dir,exist_ok=True)
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
    