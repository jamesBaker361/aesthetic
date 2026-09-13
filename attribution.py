'''
CLIP-gradient attribution: ranks each spatial patch of an image by how much it
drives the nsfw/aesthetic score, via grad*activation (Grad-CAM style) maps on
CLIPVisionModelWithProjection's hidden states. clip_attribution is the
production entrypoint (used by generate_clean.py to build the per-patch
quantile maps run_regression thresholds on); clip_attribution_smoothgrad and
clip_attribution_integrated_gradients are drop-in alternatives that trade
extra forward/backward passes for a less noisy attribution map.
'''

import functools
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer, CLIPTextModelWithProjection

from nudenet import NudeDetector

from rewards import get_nsfw_model, get_aesthetic_model,get_nsfw_model_text
from experiment_helpers.image_helpers import concat_images_horizontally, concat_images_vertically
import cv2

# NudeNet classes that don't help localize "explicit content" for
# clip_attribution_nudenet's purposes - faces, and body regions that are
# common but not themselves nsfw-specific.
NUDENET_EXCLUDE_CLASSES = {"FACE_FEMALE", "FACE_MALE", "BELLY_COVERED","BELLY_EXPOSED","ARMPITS_EXPOSED","FEET_EXPOSED","FEET_COVERED"}

# The 4 UNet cross-attention blocks every clip_attribution* variant builds a
# quantile map for by default. Pass a different block_list (e.g. with
# vlm_sae.VLM_BLOCK_NAME added) to also cover a block whose sparse features
# came from a different source (see vlm_sae.sparsify_vlm_embeddings) - the
# only requirement is that sparse_dir's npz files already have that block key.
DEFAULT_BLOCK_LIST = [
    "down_blocks.2.attentions.1",
    "mid_block.attentions.0",
    "up_blocks.0.attentions.0",
    "up_blocks.0.attentions.1",
]


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
    clip_grad_maps=[]
    for layer_idx,target_hidden_state in enumerate(hidden_states): # so the middle 4 layers seem to be the only not totally dogshit- maybe we should pool
        #if use_grad:
        # --- Importance (Grad * Activation) ---
        grads = target_hidden_state.grad[0, 1:, :]        # remove CLS → [N, D]
        grads=torch.nn.ReLU()(grads)
        acts  = target_hidden_state[0, 1:, :]             # [N, D]

        num_patches = acts.shape[0]
        h = w = int(num_patches ** 0.5)



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

        img_list.append(big_img)

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

    heatmap = np.clip(heatmap, 0, 1)
    heatmap = heatmap ** 0.5
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_BONE)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    max_pil=VaeImageProcessor.numpy_to_pil(255-heatmap_color)[0]

    concat2=concat_images_horizontally([avg_pil,max_pil])

    return img,concat2,score


def _patch_grid_importance(target_hidden_state, grad, og_h, og_w):
    '''
    Shared Grad-CAM core: grad*activation -> per-patch norm -> reshape to the
    ViT's own patch grid -> normalize to [0,1] -> nearest-neighbor upsample to
    the original image size. Used by every per-layer importance method
    (vanilla and smoothgrad) so the CAM math stays identical between them -
    only how `grad` was obtained differs.
    '''
    grads = grad[0, 1:, :]        # remove CLS → [N, D]
    grads = torch.nn.ReLU()(grads)
    acts = target_hidden_state[0, 1:, :]  # [N, D]

    importance = grads * acts   # [N, D] #Possibility: do normalized acts
    importance = importance.norm(dim=-1)

    num_patches = importance.shape[0]
    h = w = int(num_patches ** 0.5)
    importance = importance.reshape(h, w)

    importance = importance - importance.min()
    importance = importance / (importance.max() + 1e-8)

    importance = importance.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    big_importance = F.interpolate(
        importance,
        size=(og_h, og_w),
        mode="nearest",
    )[0, 0]
    return big_importance


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
        return [
            _patch_grid_importance(target_hidden_state, grad, og_h, og_w)
            for target_hidden_state, grad in zip(hidden_states, grads_list)
        ]

    importance_nsfw=build_importance(nsfw_grads)
    importance_aesthetic=build_importance(aesthetic_grads)

    # Whole-image scores (scalars), needed downstream as the regression target
    # in run_regression - as opposed to the per-patch importance maps above.
    return importance_aesthetic,importance_nsfw,float(aesthetic_score.detach().cpu()),float(nsfw_score.detach().cpu())


def get_importance_smoothgrad(pil_img: Image.Image,
             nsfw_model,
             aesthetic_model,
             device,
             processor,
             clip_model,
             n_samples:int=15,
             noise_std:float=0.15)->tuple[list[torch.Tensor],list[torch.Tensor],float,float]:
    '''
    SmoothGrad: averages the same grad*activation CAM used by get_importance
    over n_samples noisy copies of the input (Gaussian noise added to the
    processed pixel_values), which tends to cancel out the noisy, spiky part
    of the gradient and leave the more consistently-important regions. The
    first sample uses zero noise so the actual image is always included.
    '''
    og_w, og_h = pil_img.size
    img_tensor = transforms.PILToTensor()(pil_img)
    base_inputs = {k: v.to(device) for k, v in processor(images=img_tensor, return_tensors="pt").items()}
    clean_pixel_values = base_inputs['pixel_values']

    nsfw_maps_sum=None
    aesthetic_maps_sum=None
    nsfw_score_sum=0.0
    aesthetic_score_sum=0.0

    for i in range(n_samples):
        with torch.enable_grad():
            noise = torch.randn_like(clean_pixel_values)*noise_std if i>0 else torch.zeros_like(clean_pixel_values)
            pixel_values = (clean_pixel_values+noise).clone().requires_grad_(True)
            inputs = {**base_inputs, 'pixel_values': pixel_values}
            outputs = clip_model(**inputs, output_hidden_states=True, output_attentions=True)

            hidden_states = outputs.hidden_states
            for t in hidden_states:
                t.retain_grad()

            image_embeds = F.normalize(outputs.image_embeds, dim=-1)

            nsfw_score=nsfw_model(image_embeds)
            nsfw_score.backward(retain_graph=True)
            nsfw_grads=[t.grad.clone() for t in hidden_states]
            for t in hidden_states:
                t.grad=None

            aesthetic_score=aesthetic_model(image_embeds)
            aesthetic_score.backward()
            aesthetic_grads=[t.grad.clone() for t in hidden_states]

        nsfw_maps=[
            _patch_grid_importance(hs,g,og_h,og_w).detach()
            for hs,g in zip(hidden_states,nsfw_grads)
        ]
        aesthetic_maps=[
            _patch_grid_importance(hs,g,og_h,og_w).detach()
            for hs,g in zip(hidden_states,aesthetic_grads)
        ]

        if nsfw_maps_sum is None:
            nsfw_maps_sum=nsfw_maps
            aesthetic_maps_sum=aesthetic_maps
        else:
            nsfw_maps_sum=[s+m for s,m in zip(nsfw_maps_sum,nsfw_maps)]
            aesthetic_maps_sum=[s+m for s,m in zip(aesthetic_maps_sum,aesthetic_maps)]

        nsfw_score_sum+=float(nsfw_score.detach().cpu())
        aesthetic_score_sum+=float(aesthetic_score.detach().cpu())

    importance_nsfw=[m/n_samples for m in nsfw_maps_sum]
    importance_aesthetic=[m/n_samples for m in aesthetic_maps_sum]

    return importance_aesthetic,importance_nsfw,aesthetic_score_sum/n_samples,nsfw_score_sum/n_samples


def get_importance_integrated_gradients(pil_img: Image.Image,
             nsfw_model,
             aesthetic_model,
             device,
             processor,
             clip_model,
             n_steps:int=20)->tuple[list[torch.Tensor],list[torch.Tensor],float,float]:
    '''
    Integrated Gradients (Sundararajan et al.): integrates the gradient of the
    score w.r.t. the processed pixel_values along a straight-line path from a
    black baseline to the actual image, then scales by (image - baseline).
    Unlike get_importance/get_importance_smoothgrad this is computed directly
    in pixel space rather than per hidden-state layer (there's no well-defined
    "layer" to integrate an intermediate activation along), so it returns a
    single-element list per score instead of one map per layer - callers that
    slice/average over layers (see clip_attribution_integrated_gradients) just
    use that one map.
    '''
    og_w, og_h = pil_img.size
    img_tensor = transforms.PILToTensor()(pil_img)
    base_inputs = {k: v.to(device) for k, v in processor(images=img_tensor, return_tensors="pt").items()}
    input_pixel_values = base_inputs['pixel_values']
    baseline = torch.zeros_like(input_pixel_values)
    delta = input_pixel_values-baseline

    nsfw_grad_sum=torch.zeros_like(input_pixel_values)
    aesthetic_grad_sum=torch.zeros_like(input_pixel_values)
    nsfw_score=aesthetic_score=None

    for step in range(1,n_steps+1):
        alpha=step/n_steps
        with torch.enable_grad():
            pixel_values=(baseline+alpha*delta).clone().requires_grad_(True)
            inputs={**base_inputs,'pixel_values':pixel_values}
            outputs=clip_model(**inputs, output_hidden_states=True, output_attentions=True)
            image_embeds=F.normalize(outputs.image_embeds,dim=-1)

            step_nsfw_score=nsfw_model(image_embeds)
            step_nsfw_score.backward(retain_graph=True)
            nsfw_grad_sum=nsfw_grad_sum+pixel_values.grad
            pixel_values.grad=None

            step_aesthetic_score=aesthetic_model(image_embeds)
            step_aesthetic_score.backward()
            aesthetic_grad_sum=aesthetic_grad_sum+pixel_values.grad

        if step==n_steps:  # alpha=1 - the real image, used as the reported score
            nsfw_score=step_nsfw_score
            aesthetic_score=step_aesthetic_score

    avg_nsfw_grad=nsfw_grad_sum/n_steps
    avg_aesthetic_grad=aesthetic_grad_sum/n_steps

    nsfw_ig=(delta*avg_nsfw_grad).norm(dim=1)[0]        # [H,W] at CLIP input resolution
    aesthetic_ig=(delta*avg_aesthetic_grad).norm(dim=1)[0]

    def to_map(ig):
        ig=ig-ig.min()
        ig=ig/(ig.max()+1e-8)
        ig=ig.unsqueeze(0).unsqueeze(0)
        # bilinear here (not nearest) since this attribution is already
        # per-pixel/continuous, unlike the patch-grid maps elsewhere in this
        # file which need a blocky upsample
        big=F.interpolate(ig,size=(og_h,og_w),mode="bilinear",align_corners=False)[0,0]
        return big

    importance_nsfw=[to_map(nsfw_ig)]
    importance_aesthetic=[to_map(aesthetic_ig)]

    return importance_aesthetic,importance_nsfw,float(aesthetic_score.detach().cpu()),float(nsfw_score.detach().cpu())


@functools.cache
def get_sam2_mask_generator(model_id: str = "facebook/sam2-hiera-large", device: str = None):
    '''
    Loads (and caches, since the checkpoint download/build is expensive) a
    SAM2AutomaticMaskGenerator - "segment anything": no point/box/text prompt,
    it just proposes every object-like segment it finds in the image. Text
    conditioning happens downstream in _sam2_word_importance_map, which scores
    each proposed segment against target_words in CLIP space.
    '''
    from sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SAM2AutomaticMaskGenerator.from_pretrained(model_id, device=device)


def _sam2_word_importance_map(pil_img, target_words, mask_generator, clip_model, processor, device, similarity_threshold=0.2):
    '''
    "Segment anything" + text conditioning: mask_generator.generate proposes
    every candidate segment in pil_img with no notion of target_words, then
    each segment's bounding-box crop is CLIP-image-embedded and compared
    against the CLIP text embeddings of target_words (same cosine-similarity
    scoring as rewards.WordSimilarityModel, just applied per-segment instead
    of whole-image). A segment is kept - its pixels marked important - iff its
    best similarity to any target word clears similarity_threshold; segments
    are unioned where they overlap. Everything SAM2 didn't segment, or
    segmented but didn't match a target word, stays 0.
    '''
    w_img, h_img = pil_img.size
    img_np = np.array(pil_img)

    masks = mask_generator.generate(img_np)
    importance_map = np.zeros((h_img, w_img), dtype=np.float32)
    if not masks:
        return torch.from_numpy(importance_map)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)

    with torch.no_grad():
        text_inputs = {k: v.to(device) for k, v in tokenizer(target_words, padding=True, return_tensors="pt").items()}
        text_embeds = F.normalize(text_model(**text_inputs).text_embeds, dim=-1)  # [num_words, D]

        for m in masks:
            x, y, w, h = m["bbox"]
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w_img, int(x + w)), min(h_img, int(y + h))
            if x1 <= x0 or y1 <= y0:
                continue

            crop = pil_img.crop((x0, y0, x1, y1))
            inputs = {k: v.to(device) for k, v in processor(images=crop, return_tensors="pt").items()}
            crop_embed = F.normalize(clip_model(**inputs).image_embeds, dim=-1)  # [1, D]
            similarity = (crop_embed @ text_embeds.T).max().item()

            if similarity >= similarity_threshold:
                importance_map = np.maximum(importance_map, m["segmentation"].astype(np.float32))

    return torch.from_numpy(importance_map)


def get_importance_sam2(pil_img: Image.Image,
             nsfw_model,
             aesthetic_model,
             device,
             processor,
             clip_model,
             target_words: list,
             mask_generator=None,
             similarity_threshold: float = 0.2)->tuple[list[torch.Tensor],list[torch.Tensor],float,float]:
    '''
    SAM2 drop-in for get_importance: the per-patch importance doesn't come
    from a CLIP gradient CAM at all, it comes from Segment Anything - every
    segment SAM2 proposes for pil_img is kept (importance=1 over its pixels)
    iff its content is close enough in CLIP space to one of target_words,
    otherwise discarded (importance=0), via _sam2_word_importance_map. Same
    convention as get_importance_integrated_gradients/clip_attribution_nudenet:
    there's no per-layer map to slice, so each returned list has one element,
    and (like clip_attribution_nudenet) nsfw/aesthetic share the identical
    map since SAM2's segmentation isn't score-dependent. nsfw_model/
    aesthetic_model are only used for the whole-image scores run_regression
    needs as its target, exactly like clip_attribution_nudenet.
    '''
    mask_generator = mask_generator or get_sam2_mask_generator(device=device)

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in processor(images=pil_img, return_tensors="pt").items()}
        outputs = clip_model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        nsfw_score = nsfw_model(image_embeds)
        aesthetic_score = aesthetic_model(image_embeds)

    importance = _sam2_word_importance_map(
        pil_img, target_words, mask_generator, clip_model, processor, device, similarity_threshold
    ).to(device)

    return [importance], [importance], float(aesthetic_score.detach().cpu()), float(nsfw_score.detach().cpu())


def _save_attribution_npz(dest_path, sparse_npz_path, importance_aesthetic, importance_nsfw, aesthetic_score, nsfw_score, block_list=None):
    '''
    Shared by every clip_attribution* variant (gradient-based or NudeNet-
    based): given a single (H, W) importance map per score, already at the
    original image resolution, resizes it down to each SAE block's own patch
    grid and saves the per-patch [0,1] quantile (rank-normalized, not the raw
    importance value) alongside that block's sparse features and the
    whole-image scores. See run_regression for how these quantiles get
    thresholded.
    '''
    block_list = block_list or DEFAULT_BLOCK_LIST
    with np.load(sparse_npz_path) as old_npz:
        # whole-image scores, constant across all patches/blocks of this image
        save_dict={
            "image_aesthetic_score":aesthetic_score,
            "image_nsfw_score":nsfw_score,
        }
        for block in block_list:
            if block not in old_npz:
                continue
            features=torch.tensor(old_npz[block])
            (h,w,c)=features.size()
            save_dict[block]=features.cpu().numpy()
            for y_value,importance in zip(
                ["nsfw","aesthetic"],
                [importance_nsfw,importance_aesthetic]
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

    np.savez(dest_path, **save_dict)


def _select_nsfw_model(banned_words:list):
    '''
    get_nsfw_model()'s NSFWScorer by default, or a WordSimilarityModel over
    banned_words when given - both are callable as model(image_embeds) and
    both backprop, so every clip_attribution* variant can use either
    interchangeably.
    '''
    if banned_words:
        return get_nsfw_model_text(banned_words)
    return get_nsfw_model()


def _clip_attribution_core(image_src_dir:str,dest_dir:str,limit:int,
                     sparse_dir:str,
                     start_layer:int,
                     stop_layer:int,
                     importance_fn,
                     banned_words:list=None,
                     block_list:list=None):
    # Step 1 of the intended pipeline: rank each spatial patch by how much it
    # drives the nsfw/aesthetic score (via importance_fn's grad*activation
    # maps), then convert that ranking to a [0,1] quantile per patch (see
    # _save_attribution_npz). Also stash the whole-image scores themselves -
    # run_regression needs both: the quantile to threshold/weight patches, the
    # whole-image score as the target. Shared by clip_attribution and its
    # smoothgrad/integrated-gradients variants - they only differ in
    # importance_fn (which grad they feed the same CAM/quantile pipeline).
    print("clip attributuon")
    os.makedirs(dest_dir,exist_ok=True)
    # get models
    nsfw_model=_select_nsfw_model(banned_words)
    aesthetic_model=get_aesthetic_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    files=[f for f in os.listdir(image_src_dir) if f.endswith("jpg") or f.endswith("jpeg")]
    if limit>=0:
        files=files[:limit]
    for file in files:
        npz_file=file+".npz"
        sparse_path=os.path.join(sparse_dir,npz_file)
        if not os.path.exists(sparse_path):
            continue
        dest_path=os.path.join(dest_dir,npz_file)
        if os.path.exists(dest_path):
            continue

        # --- Load image ---
        pil_img = Image.open(os.path.join(image_src_dir, file)).convert("RGB")
        importance_aesthetic,importance_nsfw,aesthetic_score,nsfw_score=importance_fn(pil_img,nsfw_model,aesthetic_model,device,processor,clip_model)
        importance_aesthetic=importance_aesthetic[start_layer:stop_layer]
        importance_nsfw=importance_nsfw[start_layer:stop_layer]

        avg_aesthetic=torch.stack(importance_aesthetic).mean(dim=0)
        avg_nsfw=torch.stack(importance_nsfw).mean(dim=0)
        _save_attribution_npz(dest_path,sparse_path,avg_aesthetic,avg_nsfw,aesthetic_score,nsfw_score,block_list)


def clip_attribution(image_src_dir:str,dest_dir:str,limit:int,
                     sparse_dir:str="sparse_embeddings",
                     start_layer=5,
                     stop_layer=15,
                     banned_words:list=None,
                     block_list:list=None,
                     ):
    return _clip_attribution_core(image_src_dir,dest_dir,limit,sparse_dir,start_layer,stop_layer,get_importance,banned_words,block_list)


def clip_attribution_smoothgrad(image_src_dir:str,dest_dir:str,limit:int,
                     sparse_dir:str="sparse_embeddings",
                     start_layer=5,
                     stop_layer=15,
                     n_samples:int=15,
                     noise_std:float=0.15,
                     banned_words:list=None,
                     block_list:list=None):
    importance_fn=functools.partial(get_importance_smoothgrad,n_samples=n_samples,noise_std=noise_std)
    return _clip_attribution_core(image_src_dir,dest_dir,limit,sparse_dir,start_layer,stop_layer,importance_fn,banned_words,block_list)


def clip_attribution_integrated_gradients(image_src_dir:str,dest_dir:str,limit:int,
                     sparse_dir:str="sparse_embeddings",
                     n_steps:int=20,
                     banned_words:list=None,
                     block_list:list=None):
    importance_fn=functools.partial(get_importance_integrated_gradients,n_steps=n_steps)
    # get_importance_integrated_gradients returns a single-element list (no
    # per-layer maps to slice), so always take layer 0
    return _clip_attribution_core(image_src_dir,dest_dir,limit,sparse_dir,0,1,importance_fn,banned_words,block_list)


def clip_attribution_sam2(image_src_dir:str,dest_dir:str,limit:int,
                     target_words:list,
                     sparse_dir:str="sparse_embeddings",
                     similarity_threshold:float=0.2,
                     sam2_model_id:str="facebook/sam2-hiera-large",
                     block_list:list=None):
    '''
    Same output format/pipeline as clip_attribution, but the per-patch
    importance comes from get_importance_sam2 (SAM2's "segment anything"
    proposals, kept where they match target_words in CLIP space) instead of a
    CLIP gradient CAM - the SAM2 analogue of clip_attribution_nudenet, which
    similarly swaps in a non-gradient localization source.
    '''
    print("clip attribution (sam2)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_generator = get_sam2_mask_generator(sam2_model_id, device=device)
    importance_fn = functools.partial(
        get_importance_sam2,
        target_words=target_words,
        mask_generator=mask_generator,
        similarity_threshold=similarity_threshold,
    )
    # get_importance_sam2 returns a single-element list (no per-layer maps to
    # slice), so always take layer 0 - same convention as
    # clip_attribution_integrated_gradients.
    return _clip_attribution_core(image_src_dir,dest_dir,limit,sparse_dir,0,1,importance_fn,target_words,block_list)


def nudenet_importance_map(detections, h_img, w_img, exclude_classes):
    '''(H, W) map: 0 everywhere except inside a detected box whose class isn't
    in exclude_classes, filled with that detection's score (max where boxes overlap).'''
    map_np = np.zeros((h_img, w_img), dtype=np.float32)
    for d in detections:
        if d["class"] in exclude_classes:
            continue
        x, y, w, h = d["box"]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w_img, int(x + w)), min(h_img, int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        map_np[y0:y1, x0:x1] = np.maximum(map_np[y0:y1, x0:x1], d["score"])
    return map_np


def clip_attribution_nudenet(image_src_dir:str,dest_dir:str,limit:int,
                     sparse_dir:str="sparse_embeddings",
                     exclude_classes=NUDENET_EXCLUDE_CLASSES,
                     banned_words:list=None,
                     block_list:list=None):
    '''
    Same output format/pipeline as clip_attribution, but the per-patch
    importance comes directly from NudeDetector's boxes instead of a CLIP
    gradient: any detected box whose class isn't in exclude_classes is filled
    with its detection score (max where boxes overlap), everything else is 0
    (see nudenet_importance_map). No forward/backward CAM needed for the
    importance itself - nsfw_model/aesthetic_model/clip_model/processor are
    only used for the whole-image scores run_regression needs as its target
    (nsfw_model swaps to a WordSimilarityModel over banned_words when given,
    same as every other clip_attribution* variant); NudeDetector does the
    actual localization.
    '''
    print("clip attribution (nudenet)")
    os.makedirs(dest_dir,exist_ok=True)
    nsfw_model=_select_nsfw_model(banned_words)
    aesthetic_model=get_aesthetic_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    detector = NudeDetector()

    files=[f for f in os.listdir(image_src_dir) if f.endswith("jpg") or f.endswith("jpeg")]
    if limit>=0:
        files=files[:limit]
    for file in files:
        npz_file=file+".npz"
        sparse_path=os.path.join(sparse_dir,npz_file)
        if not os.path.exists(sparse_path):
            continue
        dest_path=os.path.join(dest_dir,npz_file)
        if os.path.exists(dest_path):
            continue

        path=os.path.join(image_src_dir,file)
        pil_img=Image.open(path).convert("RGB")
        w_img,h_img=pil_img.size

        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in processor(images=pil_img, return_tensors="pt").items()}
            outputs = clip_model(**inputs)
            image_embeds = F.normalize(outputs.image_embeds, dim=-1)
            nsfw_score=nsfw_model(image_embeds)
            aesthetic_score=aesthetic_model(image_embeds)

        detections=detector.detect(path)
        importance=torch.tensor(nudenet_importance_map(detections,h_img,w_img,exclude_classes))

        # NudeNet doesn't distinguish "important for nsfw" vs "important for
        # aesthetic" - the same box-derived map is used for both targets
        _save_attribution_npz(dest_path,sparse_path,importance,importance,
                               float(aesthetic_score.detach().cpu()),float(nsfw_score.detach().cpu()),block_list)
