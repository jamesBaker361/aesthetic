'''
Alternative sparse-feature source: instead of the UNet-activation SAEs in
sdxl_unbox/checkpoints (used throughout sparsify.py), this extracts CLIP ViT
patch-token features and encodes them with a pretrained SAE from
mateuszpach/sae-for-vlm (paper: "Sparse Autoencoders Learn Monosemantic
Features in Vision-Language Models", github.com/ExplainableML/sae-for-vlm).
download.py fetches the checkpoints into vlm_checkpoints_dir.

That repo's checkpoints are keyed by (base model, layer, attachment point,
sae variant); the layout on disk (per its HF repo file listing) is:
    {sae_variant}/imagenet_train_activations_{model}_{layer}_post_mlp_residual_{sae_variant}/trainer_0/checkpoints/ae_100000.pt
Only two base models have checkpoints: clip-vit-large-patch14-336 (NOT the
clip-vit-large-patch14 used elsewhere in this project - different input
resolution, so its own model instance is loaded here) and
siglip-so400m-patch14-384. "post_mlp_residual" activations are the residual
stream after each ViT encoder layer's MLP block, i.e. CLIPVisionModel's own
hidden_states - available layers for CLIP: 11, 17, 22, 23 (0-indexed into
encoder.layers). hidden_states[0] is the embedding output, so
encoder.layers[layer]'s output is hidden_states[layer+1].

sparsify_vlm_embeddings mirrors sparsify_embeddings: for each image already
present in sparse_dest_dir's npz (from sparsify_embeddings), it adds one more
block key holding this SAE's sparse features, in the same (H, W, dict_size)
layout as every UNet block - so get_top_k_images/run_regression/
run_top_k_features_popularity_contest/clip_attribution all work against it
unmodified, just by adding VLM_BLOCK_NAME to whatever block_list they're
given. It only covers that analysis pipeline, not train_lora's generation-
time suppression hooks (hookify) - those hook into unet.named_modules() by
block name, and a CLIP ViT layer was never part of the UNet's forward pass to
begin with, so there's nothing there to hook.
'''

import functools
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

VLM_MODEL_NAME = "openai/clip-vit-large-patch14-336"
VLM_CHECKPOINT_DIR = "vlm_checkpoints_dir"
VLM_LAYER = 22             # one of 11, 17, 22, 23 for clip-vit-large-patch14-336
VLM_SAE_VARIANT = "batch_top_k_20_x8"   # batch_top_k_20_x{1,2,4,8,16,64} or matroyshka_batch_top_k_20_x{...}
VLM_BLOCK_NAME = f"vlm_clip_l14_336_layer{VLM_LAYER}"


class BatchTopKSAE(nn.Module):
    '''
    Minimal reimplementation of dictionary_learning's BatchTopKSAE, just
    enough to load mateuszpach/sae-for-vlm's checkpoints and encode/decode -
    parameter names/shapes have to match the checkpoint's state_dict exactly
    (encoder, decoder, b_dec, threshold, k) for load_state_dict to work.
    At inference it thresholds each feature at its own learned scalar
    threshold (not a top-k op - that's only used during training).
    '''
    def __init__(self, activation_dim:int, dict_size:int, k:int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(self, x:torch.Tensor)->torch.Tensor:
        pre_acts = torch.relu(self.encoder(x - self.b_dec))
        return pre_acts * (pre_acts > self.threshold)

    def decode(self, f:torch.Tensor)->torch.Tensor:
        return self.decoder(f) + self.b_dec

    @property
    def n_dirs_local(self):
        # matches sdxl_unbox.SAE.SparseAutoencoder's attribute name, so code
        # written against that (e.g. generate_clean.py's dim=sae.n_dirs_local)
        # doesn't need a special case for this SAE
        return self.dict_size

    @classmethod
    def from_pretrained(cls, path:str, device=None)->"BatchTopKSAE":
        state_dict = torch.load(path, map_location="cpu")
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        k = int(state_dict["k"].item())
        sae = cls(activation_dim, dict_size, k)
        sae.load_state_dict(state_dict)
        if device is not None:
            sae.to(device)
        return sae


def _vlm_checkpoint_path(checkpoint_dir:str, layer:int, sae_variant:str)->str:
    model_short = VLM_MODEL_NAME.rsplit("/", 1)[-1]
    run_name = f"imagenet_train_activations_{model_short}_{layer}_post_mlp_residual_{sae_variant}"
    return os.path.join(checkpoint_dir, sae_variant, run_name, "trainer_0", "checkpoints", "ae_100000.pt")


@functools.cache
def get_vlm_clip_model(device:str):
    model = CLIPVisionModelWithProjection.from_pretrained(VLM_MODEL_NAME).to(device)
    processor = CLIPImageProcessor.from_pretrained(VLM_MODEL_NAME)
    return model, processor


@functools.cache
def get_vlm_sae(device:str, layer:int=VLM_LAYER, sae_variant:str=VLM_SAE_VARIANT, checkpoint_dir:str=VLM_CHECKPOINT_DIR)->BatchTopKSAE:
    path = _vlm_checkpoint_path(checkpoint_dir, layer, sae_variant)
    return BatchTopKSAE.from_pretrained(path, device=device)


def extract_vlm_patch_features(pil_img:Image.Image, clip_model, processor, device:str, layer:int=VLM_LAYER)->torch.Tensor:
    '''
    Returns this image's patch-token activations at encoder.layers[layer]'s
    output, reshaped to (1, H, W, C) - channel-last already (unlike the UNet
    features elsewhere in this project, ViT hidden_states have no permute to
    do), CLS token dropped, H=W=sqrt(num_patches).
    '''
    inputs = {k: v.to(device) for k, v in processor(images=pil_img, return_tensors="pt").items()}
    with torch.no_grad():
        outputs = clip_model(**inputs, output_hidden_states=True)
    # hidden_states[0] is the embedding output, so encoder.layers[layer]'s
    # output is hidden_states[layer + 1]
    hidden_state = outputs.hidden_states[layer + 1]  # [1, 1+num_patches, C]
    patch_tokens = hidden_state[:, 1:, :]             # drop CLS -> [1, num_patches, C]
    num_patches = patch_tokens.shape[1]
    h = w = int(num_patches ** 0.5)
    c = patch_tokens.shape[-1]
    return patch_tokens.reshape(1, h, w, c)


def sparsify_vlm_embeddings(image_src_dir:str,
                            sparse_dest_dir:str="sparse_embeddings",
                            extension:str="jpg",
                            layer:int=VLM_LAYER,
                            sae_variant:str=VLM_SAE_VARIANT,
                            checkpoint_dir:str=VLM_CHECKPOINT_DIR,
                            block_name:str=None):
    '''
    Adds this SAE's sparse features as one more block key to the npz files
    sparsify_embeddings already produced in sparse_dest_dir (matched by
    filename: "<image file>.npz") - skips images that don't have one yet
    (run sparsify_embeddings first) and images that already have this block.
    '''
    block_name = block_name or VLM_BLOCK_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, processor = get_vlm_clip_model(device)
    sae = get_vlm_sae(device, layer, sae_variant, checkpoint_dir)

    files = [f for f in os.listdir(image_src_dir) if f.endswith(extension) or f.endswith("jpeg")]
    for file in tqdm(files, desc="Sparsifying (VLM SAE)"):
        npz_path = os.path.join(sparse_dest_dir, file + ".npz")
        if not os.path.exists(npz_path):
            continue
        with np.load(npz_path) as existing:
            if block_name in existing:
                continue
            data = dict(existing)

        pil_img = Image.open(os.path.join(image_src_dir, file)).convert("RGB")
        features = extract_vlm_patch_features(pil_img, clip_model, processor, device, layer)  # [1,H,W,C]
        sparse = sae.encode(features)[0]  # -> [H,W,dict_size]
        data[block_name] = sparse.cpu().detach().numpy()
        np.savez(npz_path, **data)
