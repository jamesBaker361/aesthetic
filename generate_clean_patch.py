# this trains the sae and then does SAEURON stuyle removal I think?

import os
import json
import random
import re
import argparse
import urllib.request
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.argprint import print_args
from experiment_helpers.image_helpers import concat_images_horizontally, concat_images_vertically
from diffusers import DiffusionPipeline,UNet2DConditionModel,AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from sdxl_unbox.SAE import SparseAutoencoder
from diffusers import SanaSprintPipeline,Krea2Pipeline
from huggingface_hub import snapshot_download,hf_hub_download
import torch
import torch.nn.functional as F
import numpy as np
import csv
import sys

sys.path.append("/umbc/rs/pi_donengel/users/jbaker15/aesthetic/saev_repo/src")

import shutil
import time
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score

from PIL import Image
from torchvision.transforms import v2

import saev.src.saev.data.models
import saev.src.saev.data.shards
import saev.src.saev.nn
import saev.src.saev.viz

from sam3_repo.sam3.model_builder import build_sam3_image_model
from sam3_repo.sam3.model.sam3_image_processor import Sam3Processor

from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init
parser=default_parser(
    {
        "repo_id":"jlbaker361/nsfw"
    }
)

parser.add_argument("--image_src_dir",type=str,default="real")
parser.add_argument("--embedding_dir",type=str,default="embeddings")
parser.add_argument("--sparse_embedding_dir",type=str,default="sparse_embeddings")
parser.add_argument("--mask_dir",type=str,default="mask_dir")
parser.add_argument("--eval_dir",type=str,default="eval_dir")
parser.add_argument("--npz_dict",type=str,default="platonic.npz")

parser.add_argument("--target_query",type=str,default="banana")

parser.add_argument("--partition_path",type=str,default="partition.json")
parser.add_argument("--train_frac",type=float,default=0.8)
parser.add_argument("--seed",type=int,default=42)

parser.add_argument("--vit_family",type=str,default="dinov2")
parser.add_argument("--vit_ckpt",type=str,default="dinov2_vitb14_reg")
parser.add_argument("--sae_hf_repo",type=str,default="osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K")
parser.add_argument("--layer",type=int,default=10)
parser.add_argument("--n_content_tokens",type=int,default=256)
parser.add_argument("--resize_ratio",type=float,default=256/224)

parser.add_argument("--top_k",type=int,default=10)
parser.add_argument("--n_visualize",type=int,default=5)

IMAGE_EXTENSIONS=(".jpg",".jpeg",".png",".bmp",".webp")


def list_images(image_src_dir:str)->list:
    return sorted(f for f in os.listdir(image_src_dir) if f.lower().endswith(IMAGE_EXTENSIONS))


def get_or_make_partition(image_src_dir:str,partition_path:str,train_frac:float,seed:int)->dict:
    if os.path.exists(partition_path):
        with open(partition_path) as f:
            return json.load(f)

    images=list_images(image_src_dir)
    rng=random.Random(seed)
    rng.shuffle(images)
    n_train=int(round(len(images)*train_frac))
    partition={"train":images[:n_train],"test":images[n_train:]}
    with open(partition_path,"w") as f:
        json.dump(partition,f,indent=2)
    return partition


# commit pinned by the notebook's own `pip install saev@<sha>` cell, so the
# constants parsed out of it can't silently drift if main changes later
_SAEV_NOTEBOOK_COMMIT="6d6eff52c4ae04f5153badc0a553adddc8d3e3cc"
_SAEV_NOTEBOOK_URL=f"https://raw.githubusercontent.com/Imageomics/saev/{_SAEV_NOTEBOOK_COMMIT}/examples/inference.ipynb"
# which notebook cell (matched by its `..._SCALAR = ` variable name) holds the
# norm constants for each ckpt we know how to fetch
_DINO_NORM_VAR_BY_CKPT={
    "dinov2_vitb14_reg":"DINOV2_IMAGENET1K",
}


def download_dino_norm_file(vit_ckpt:str,norm_path:str):
    var_prefix=_DINO_NORM_VAR_BY_CKPT.get(vit_ckpt)
    if var_prefix is None:
        raise ValueError(
            f"no known source for '{vit_ckpt}' normalization constants - "
            f"add it to _DINO_NORM_VAR_BY_CKPT or drop a {os.path.basename(norm_path)} file in dino_norms/ by hand"
        )

    with urllib.request.urlopen(_SAEV_NOTEBOOK_URL) as resp:
        notebook=json.load(resp)

    source=""
    for cell in notebook["cells"]:
        text="".join(cell.get("source",[]))
        if f"{var_prefix}_SCALAR" in text and f"{var_prefix}_MEAN" in text:
            source=text
            break
    if not source:
        raise RuntimeError(f"couldn't find {var_prefix}_SCALAR/{var_prefix}_MEAN in {_SAEV_NOTEBOOK_URL}")

    scalar=float(re.search(rf"{var_prefix}_SCALAR\s*=\s*([0-9.eE+-]+)",source).group(1))
    list_src=source[source.index(f"{var_prefix}_MEAN"):]
    inner=list_src[list_src.index("[")+1:list_src.index("]")]
    mean=[float(x.strip()) for x in inner.split(",") if x.strip()]

    os.makedirs(os.path.dirname(norm_path),exist_ok=True)
    with open(norm_path,"w") as f:
        json.dump({"scalar":scalar,"mean":mean},f)


def load_dino_normalize_fn(vit_ckpt:str):
    norm_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"dino_norms",f"{vit_ckpt}_in1k.json")
    if not os.path.exists(norm_path):
        print(f"{norm_path} not found, downloading normalization constants from {_SAEV_NOTEBOOK_URL}...")
        download_dino_norm_file(vit_ckpt,norm_path)
    with open(norm_path) as f:
        data=json.load(f)
    mean=torch.tensor(data["mean"])
    scalar=data["scalar"]

    def _normalize(x):
        # DINOv2 SAEs expect activations normalized by the IN1K mean/scalar
        # they were trained on (see Imageomics/saev examples/inference.ipynb)
        return (x.clamp(-1e5,1e5)-mean.to(x.device))/scalar

    return _normalize


def load_vit_and_sae(vit_family:str,vit_ckpt:str,sae_hf_repo:str,layer:int,n_content_tokens:int,device):
    vit_cls=saev.data.models.load_model_cls(vit_family)
    img_tr,_=vit_cls.make_transforms(vit_ckpt,n_content_tokens)
    vit=vit_cls(vit_ckpt).to(device)
    vit.eval()
    recorded_vit=saev.data.shards.RecordedTransformer(vit,n_content_tokens,True,[layer])

    sae_fpath=hf_hub_download(sae_hf_repo,"sae.pt")
    sae=saev.nn.load(sae_fpath,device=device)
    sae.eval()

    return recorded_vit,sae,img_tr


def extract_patch_acts(recorded_vit,img:Image.Image,img_tr,device):
    x=img_tr(img)[None,...].to(device)
    with torch.no_grad():
        _,vit_acts=recorded_vit(x)
    # vit_acts: [batch, n_layers, tokens_per_example, d_model]; strip CLS (index 0)
    return vit_acts[0,0,1:,:]


def cache_embeddings(images:list,image_src_dir:str,embedding_dir:str,recorded_vit,img_tr,device):
    for name in images:
        out_path=os.path.join(embedding_dir,name+".npz")
        if os.path.exists(out_path):
            continue
        img=Image.open(os.path.join(image_src_dir,name)).convert("RGB")
        patch_acts=extract_patch_acts(recorded_vit,img,img_tr,device)
        np.savez(out_path,patch_acts=patch_acts.cpu().numpy())


def cache_sparse_embeddings(images:list,embedding_dir:str,sparse_embedding_dir:str,sae,normalize_fn,device):
    for name in images:
        out_path=os.path.join(sparse_embedding_dir,name+".npz")
        if os.path.exists(out_path):
            continue
        data=np.load(os.path.join(embedding_dir,name+".npz"))
        patch_acts=torch.from_numpy(data["patch_acts"]).to(device)
        if normalize_fn is not None:
            patch_acts=normalize_fn(patch_acts)
        with torch.no_grad():
            sae_out=sae(patch_acts)
        np.savez(out_path,f_x=sae_out.f_x.cpu().numpy())


def mask_out_path(mask_dir:str,name:str,target_query:str)->str:
    safe_query=target_query.replace(" ","_")
    return os.path.join(mask_dir,f"{name}.{safe_query}.npz")


def cache_masks(images:list,image_src_dir:str,mask_dir:str,target_query:str,processor,grid_size:int,image_size:int,resize_size:int):
    # NEAREST so the resize/crop that puts the mask into the same pixel frame
    # as img_tr's ViT input doesn't blur its binary 0/1 edges
    mask_transform=v2.Compose([
        v2.Resize(resize_size,interpolation=v2.InterpolationMode.NEAREST),
        v2.CenterCrop(image_size),
    ])
    patch_size=image_size//grid_size

    for name in images:
        out_path=mask_out_path(mask_dir,name,target_query)
        if os.path.exists(out_path):
            continue

        image=Image.open(os.path.join(image_src_dir,name)).convert("RGB")
        inference_state=processor.set_image(image)
        output=processor.set_text_prompt(state=inference_state,prompt=target_query)
        masks,scores=output["masks"],output["scores"]
        w,h=image.size

        if len(scores)==0:
            big_seg=np.zeros((h,w),dtype=bool)
        else:
            masks_np=masks.squeeze(1).cpu().numpy()
            # union of every returned mask - True wherever ANY mask is nonzero
            big_seg=np.any(masks_np,axis=0)

        mask_img=Image.fromarray((big_seg.astype(np.uint8))*255,mode="L")
        mask_img=mask_transform(mask_img)
        mask_cropped=np.array(mask_img)>127

        # a patch counts as "on-target" if the majority of its pixels fall
        # inside the SAM3 mask
        patch_mask=mask_cropped.reshape(grid_size,patch_size,grid_size,patch_size).mean(axis=(1,3))>0.5

        np.savez(out_path,patch_mask=patch_mask,pixel_mask=big_seg)


def discover_top_features(train_images:list,sparse_embedding_dir:str,mask_dir:str,target_query:str,top_k:int):
    feats_list,labels_list=[],[]
    for name in train_images:
        feats_list.append(np.load(os.path.join(sparse_embedding_dir,name+".npz"))["f_x"])
        labels_list.append(np.load(mask_out_path(mask_dir,name,target_query))["patch_mask"].reshape(-1))

    feats=np.concatenate(feats_list,axis=0)
    labels=np.concatenate(labels_list,axis=0)

    n_pos=int(labels.sum())
    n_neg=len(labels)-n_pos
    if n_pos==0 or n_neg==0:
        raise ValueError(
            f"'{target_query}' has no positive/negative patch contrast across the train "
            f"partition ({n_pos} positive / {n_neg} negative patches) - can't discover features"
        )

    # per-latent AUROC of "does this latent's activation separate on-target
    # from off-target patches", computed for every latent at once via the
    # rank-sum form of the Mann-Whitney U statistic instead of one
    # roc_auc_score call per latent (there can be tens of thousands of them)
    ranks=rankdata(feats,axis=0)
    sum_ranks_pos=ranks[labels].sum(axis=0)
    auc=(sum_ranks_pos-n_pos*(n_pos+1)/2)/(n_pos*n_neg)

    top_idx=np.argsort(auc)[::-1][:top_k]
    return top_idx,auc[top_idx]


def update_npz_dict(npz_dict_path:str,target_query:str,top_idx:np.ndarray,top_scores:np.ndarray):
    existing={}
    if os.path.exists(npz_dict_path):
        with np.load(npz_dict_path,allow_pickle=True) as data:
            existing={k:data[k] for k in data.files}
    existing[target_query]=top_idx.astype(np.int64)
    existing[f"{target_query}__auc"]=top_scores.astype(np.float32)
    np.savez(npz_dict_path,**existing)


def evaluate_on_test(test_images:list,sparse_embedding_dir:str,mask_dir:str,target_query:str,top_idx:np.ndarray,
                      eval_dir:str,image_src_dir:str,img_tr,image_size:int,resize_size:int,patch_size:int,n_visualize:int):
    os.makedirs(eval_dir,exist_ok=True)
    safe_query=target_query.replace(" ","_")

    all_scores,all_labels=[],[]
    per_image_metrics={}
    vis_transform=v2.Compose([v2.Resize(resize_size),v2.CenterCrop(image_size)])

    for i,name in enumerate(test_images):
        sparse=np.load(os.path.join(sparse_embedding_dir,name+".npz"))["f_x"]
        mask=np.load(mask_out_path(mask_dir,name,target_query))["patch_mask"].reshape(-1)
        # combine the discovered latents by taking, per patch, the strongest
        # of the top_k activations as the "is this the target" score
        patch_score=sparse[:,top_idx].max(axis=1)

        all_scores.append(patch_score)
        all_labels.append(mask)

        if mask.any() and not mask.all():
            per_image_metrics[name]={
                "auroc":float(roc_auc_score(mask,patch_score)),
                "ap":float(average_precision_score(mask,patch_score)),
            }

        if i<n_visualize:
            img=Image.open(os.path.join(image_src_dir,name)).convert("RGB")
            vis_img=vis_transform(img)
            highlighted=saev.viz.add_highlights(vis_img,patch_score,patch_size,upper=float(patch_score.max()))
            highlighted.save(os.path.join(eval_dir,f"{safe_query}_{name}.png"))

    scores=np.concatenate(all_scores)
    labels=np.concatenate(all_labels)

    pooled_auroc=float(roc_auc_score(labels,scores)) if 0<labels.sum()<len(labels) else None
    pooled_ap=float(average_precision_score(labels,scores)) if 0<labels.sum()<len(labels) else None

    metrics={
        "n_test_images":len(test_images),
        "pooled_auroc":pooled_auroc,
        "pooled_ap":pooled_ap,
        "per_image":per_image_metrics,
    }
    with open(os.path.join(eval_dir,f"{safe_query}_metrics.json"),"w") as f:
        json.dump(metrics,f,indent=2)

    return metrics


def main(args):
    api,accelerator,device=repo_api_init(args)

    image_src_dir:str=args.image_src_dir
    embedding_dir:str=args.embedding_dir
    sparse_embedding_dir:str=args.sparse_embedding_dir
    mask_dir:str=args.mask_dir
    eval_dir:str=args.eval_dir

    for d in [image_src_dir,embedding_dir,sparse_embedding_dir,mask_dir,eval_dir]:
        os.makedirs(d,exist_ok=True)

    target_query:str=args.target_query

    partition=get_or_make_partition(image_src_dir,args.partition_path,args.train_frac,args.seed)
    train_images,test_images=partition["train"],partition["test"]
    all_images=train_images+test_images
    print(f"partition: {len(train_images)} train images, {len(test_images)} test images")

    recorded_vit,sae,img_tr=load_vit_and_sae(
        args.vit_family,args.vit_ckpt,args.sae_hf_repo,args.layer,args.n_content_tokens,device
    )
    normalize_fn=load_dino_normalize_fn(args.vit_ckpt) if args.vit_family=="dinov2" else None

    print("caching ViT patch embeddings...")
    cache_embeddings(all_images,image_src_dir,embedding_dir,recorded_vit,img_tr,device)

    print("caching sparse (SAE) embeddings...")
    cache_sparse_embeddings(all_images,embedding_dir,sparse_embedding_dir,sae,normalize_fn,device)

    patch_size=recorded_vit.model.patch_size
    grid_size=int(round(args.n_content_tokens**0.5))
    image_size=grid_size*patch_size
    resize_size=round(image_size*args.resize_ratio)

    print(f"extracting SAM3 masks for '{target_query}'...")
    if device=="cuda" or (hasattr(device,"type") and device.type=="cuda"):
        # SAM3 is meant to run under bfloat16 autocast end-to-end - without
        # it, parts of the model produce bf16 intermediates while plain
        # float32 inputs stay float32, causing a dtype-mismatch crash.
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.autocast("cuda",dtype=torch.bfloat16).__enter__()
    sam3_model=build_sam3_image_model()
    sam3_processor=Sam3Processor(sam3_model,device=device)
    cache_masks(all_images,image_src_dir,mask_dir,target_query,sam3_processor,grid_size,image_size,resize_size)

    print(f"discovering top {args.top_k} features for '{target_query}' from the train partition...")
    top_idx,top_scores=discover_top_features(train_images,sparse_embedding_dir,mask_dir,target_query,args.top_k)
    for idx,score in zip(top_idx,top_scores):
        print(f"  latent {idx}: train AUROC {score:.3f}")
    update_npz_dict(args.npz_dict,target_query,top_idx,top_scores)

    print("evaluating discovered features on the held-out test partition...")
    metrics=evaluate_on_test(
        test_images,sparse_embedding_dir,mask_dir,target_query,top_idx,
        eval_dir,image_src_dir,img_tr,image_size,resize_size,patch_size,args.n_visualize,
    )
    print(json.dumps({k:v for k,v in metrics.items() if k!="per_image"},indent=2))

if __name__=='__main__':
    print_args(parser)
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
