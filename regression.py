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
from experiment_helpers.data_helpers import split_data
from sklearn.linear_model import Ridge,LinearRegression,ElasticNet,Lasso
from diffusers.image_processor import VaeImageProcessor
from d3po_rewards import get_nsfw_model,get_aesthetic_model
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from experiment_helpers.image_helpers import concat_images_horizontally,concat_images_vertically
import matplotlib.pyplot as plt
import cv2
from accelerate import Accelerator

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
        score=nsfw_model(image_embeds)
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

        # --- Convert for plotting ---
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy() / 255.0
        heatmap = big_importance.detach().cpu().numpy()

        # --- Optional sharpening ---
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = heatmap ** 0.5
        

        # convert heatmap → color
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_BONE)

        # convert original image
        img_uint8 = np.uint8(img_np * 255)

        # blend
        overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)
        
        
        
        
        #overlay=cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_img=VaeImageProcessor.numpy_to_pil(overlay)[0]
        
        heat_map_pil=VaeImageProcessor.numpy_to_pil(heatmap_color)[0]
        
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

        # convert original image
        img_uint8 = np.uint8(img_np * 255)

        # blend
        overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)
        
        
        
        
        overlay=cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_img=VaeImageProcessor.numpy_to_pil(overlay)[0]
        
        heat_map_pil=VaeImageProcessor.numpy_to_pil(heatmap_color)[0]
        
        pil_img=concat_images_vertically([big_img,pil_img,heat_map_pil])
        
        img_list.append(pil_img)
        
    
    concat=concat_images_horizontally(img_list)
    arr = np.array(concat)

    arr = np.ascontiguousarray(arr)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr).convert("RGB")
    return img

def get_importance(pil_img: Image.Image,
             nsfw_model,
             aesthetic_model,
             device,
             processor,
             clip_model)->torch.Tensor:
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
        score=nsfw_model(image_embeds)
        score.backward()
        
    target_layers=[]
    for layer_idx,target_hidden_state in enumerate(hidden_states): # so the middle 4 layers seem to be the only not totally dogshit- maybe we should pool
        #if use_grad:
        # --- Importance (Grad * Activation) ---
        grads = target_hidden_state.grad[0, 1:, :]        # remove CLS → [N, D]
        grads=torch.nn.ReLU()(grads)
        acts  = target_hidden_state[0, 1:, :]             # [N, D]
        
        num_patches = acts.shape[0]
        h = w = int(num_patches ** 0.5)
        


        importance = grads * acts                       # [N, D]
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


def clip_attribution(image_src_dir:str,dest_dir:str,limit:int,sparse_dir:str="sparse_embeddings",use_grad:bool=False):
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
        img=get_maps(pil_img,nsfw_model,
             aesthetic_model,
             device,
             processor,
             clip_model)
        path=os.path.join(dest_dir,file).replace("jpg","png")
        img.save(path, quality=65)
        
import numpy as np
import os
import torch

def compute_stats(file_list, block, y_column):
    X_sum, X_sq_sum, count = None, None, 0
    y_sum, y_sq_sum = None, None

    for file in file_list:
        data = np.load(file)

        X = data[block].reshape(-1, data[block].shape[-1])
        y = data[y_column].reshape(-1, 1)

        if X_sum is None:
            X_sum = X.sum(axis=0)
            X_sq_sum = (X**2).sum(axis=0)
            y_sum = y.sum(axis=0)
            y_sq_sum = (y**2).sum(axis=0)
        else:
            X_sum += X.sum(axis=0)
            X_sq_sum += (X**2).sum(axis=0)
            y_sum += y.sum(axis=0)
            y_sq_sum += (y**2).sum(axis=0)

        count += X.shape[0]

    X_mean = X_sum / count
    X_std = np.sqrt(X_sq_sum / count - X_mean**2) + 1e-6

    y_mean = y_sum / count
    y_std = np.sqrt(y_sq_sum / count - y_mean**2) + 1e-6

    return X_mean, X_std, y_mean, y_std

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, block, y_column, X_mean, X_std, y_mean, y_std):
        self.file_list = file_list
        self.block = block
        self.y_column = y_column

        self.X_mean = torch.tensor(X_mean, dtype=torch.float32)
        self.X_std = torch.tensor(X_std, dtype=torch.float32)
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
        self.y_std = torch.tensor(y_std, dtype=torch.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = np.load(self.file_list[index])

        X = torch.tensor(data[self.block], dtype=torch.float32)
        y = torch.tensor(data[self.y_column], dtype=torch.float32)

        X = X.flatten(0, -2)  # ensure (N, dim)
        y = y.view(-1, 1)

        # 🔹 normalize
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        return {"indep": X, "dep": y}
        

def run_regression(block:str,dim:int,y_column:str,
                   limit:int,clip_src_dir:str,
                   stats_dest_dir:str,
                   mixed_precision:str,
                   gradient_accumulation_steps:int,
                   epochs:int):
    for file in os.listdir(clip_src_dir):
        if file.endswith("npz"):
            embeds=np.load(os.path.join(clip_src_dir,file))[block]
            dim=embeds.shape[-1]
            break
    
    linear=torch.nn.Linear(dim,1)
    optimizer=torch.optim.AdamW(linear.parameters())
    file_list = [
        os.path.join(clip_src_dir, f)
        for f in os.listdir(clip_src_dir)
        if f.endswith("npz")
    ]

    X_mean, X_std, y_mean, y_std = compute_stats(file_list, block, y_column)

    dataset = RegressionDataset(
        file_list, block, y_column,
        X_mean, X_std, y_mean, y_std
    )
    train,test,val=split_data(dataset,0.9,1)
    
    accelerator:Accelerator=Accelerator(mixed_precision=mixed_precision,gradient_accumulation_steps=gradient_accumulation_steps)
    save_path=os.path.join(stats_dest_dir, f"regression_{block}_{y_column}.pt")
    try:
        checkpoint=torch.load(save_path,map_location="cpu")
        linear.load_state_dict(checkpoint["model_state_dict"])
        start_epoch=checkpoint["e"]+1
    except:
        start_epoch=0
    
    linear,optimizer,train,test,val=accelerator.prepare(linear,optimizer,train,test,val)
    
    
    for e in range(start_epoch,epochs):
        loss_list=[]
        for b,batch in enumerate(train):
            if b==limit:
                break
            x=batch["indep"].flatten(0,1)
            y=batch["dep"]
            with accelerator.accumulate(linear):
                with accelerator.autocast():
                    optimizer.zero_grad()
                    predicted=linear(x)
                    loss=F.mse_loss(predicted,y)
                    accelerator.backward(loss)
                    optimizer.step()
                    loss_list.append(loss.cpu().detach().float().numpy())
        print(e,np.mean(loss_list))
        if accelerator.is_main_process:
            os.makedirs(stats_dest_dir, exist_ok=True)

            unwrapped = accelerator.unwrap_model(linear)

            accelerator.save({
                "model_state_dict": unwrapped.state_dict(),
                "e":e
            }, save_path)
    
        






if __name__=="__main__":
    info_path="laion/info.csv"
    sparse_dir="sparse_embeddings"
    dest_dir="statistics"

    os.makedirs(dest_dir,exist_ok=True)
    parser=argparse.ArgumentParser()
    parser.add_argument("--y_column",type=str,default="aesthetic") #column 0 = aesthetic column = 1 = p(unsafe)
    parser.add_argument("--block",type=str,default="down_blocks.2.attentions.1")
    parser.add_argument("--limit",type=int,default=-1)
    
    clip_attribution("test_imgs","test_maps",-1,use_grad=True)
    
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
    