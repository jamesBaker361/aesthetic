# Grad-ECLIP ("Gradient-based Visual Explanation for CLIP", Zhao et al.,
# ICML 2024 - https://proceedings.mlr.press/v235/zhao24p.html) image heat maps,
# ported from Grad-Eclip/generate_emap.py's clip_encode_dense + grad_eclip
# (last-layer, withksim=True). That file can't be imported directly - it
# pulls in Game_MM_CLIP/CLIP_Surgery/M2IB at module level - and it uses
# openai's `clip` package; this uses open_clip's ViT-B-16 "openai" weights
# instead (same checkpoint as clip.load("ViT-B/16")).
#
# map for (image, text): cosine c between the CLS image embedding and the
# text embedding -> gradient of c w.r.t. the last block's attention output,
# taken at the CLS token -> per patch: relu(sum_d grad_cls * v_patch *
# cos(q_cls, k_patch)), with cos min-max normalized over patches.

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

import open_clip

_transform = Compose([
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def imgprocess_keepsize(img: Image.Image, patch_size=(16, 16), scale_factor=1) -> torch.Tensor:
    # Grad-Eclip/generate_emap.py's own preprocessing: keep the image's size
    # (rounded to a multiple of the patch size) instead of CLIP's 224 center
    # crop, and interpolate the position embedding to match - so a 512x512
    # image gets a 32x32 map covering the whole frame
    w, h = img.size
    ph, pw = patch_size
    nw = int(w * scale_factor / pw + 0.5) * pw
    nh = int(h * scale_factor / ph + 0.5) * ph
    img = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)(img).convert("RGB")
    return _transform(img)


def load_clip(device, model_name: str = "ViT-B-16", pretrained: str = "openai"):
    # openai's CLIP weights were trained with QuickGELU - open_clip's plain
    # "ViT-B-16" config uses nn.GELU, so without forcing it the activations
    # (and heat maps) are silently off
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained,
                                                         force_quick_gelu=(pretrained == "openai"))
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


def attention_layer(q, k, v, num_heads=1):
    # verbatim from generate_emap.py (LND layout). Note the repo calls this
    # with num_heads=1 (all heads treated as one) rather than the model's real
    # head count - kept as-is so the maps match the paper's code.
    tgt_len, bsz, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    q = q * float(head_dim) ** -0.5
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    attn_weights = F.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
    attn_output = torch.bmm(attn_weights, v).transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    return attn_output


def clip_encode_dense(model, x: torch.Tensor):
    '''
    generate_emap.clip_encode_dense on open_clip's VisionTransformer: runs
    every block but the last normally, then the last block's attention by
    hand so its q/k/v and attention output are available for grad_eclip.
    Returns (projected tokens (N,L,D), v, q_out, k_out, attn_output, (feah, feaw)).
    '''
    visual = model.visual
    x = visual.conv1(x)
    feah, feaw = x.shape[-2:]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    class_embedding = visual.class_embedding.to(x.dtype)
    x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)

    pos_embedding = visual.positional_embedding.to(x.dtype)
    tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
    pos_h, pos_w = visual.grid_size
    img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
    img_pos = F.interpolate(img_pos, size=(feah, feaw), mode="bicubic", align_corners=False)
    img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
    x = x + torch.cat((tok_pos[None, ...], img_pos), dim=1)
    x = visual.ln_pre(x)

    # newer open_clip transformers are batch-first (NLD), older ones LND
    batch_first = getattr(visual.transformer, "batch_first", False)
    if not batch_first:
        x = x.permute(1, 0, 2)
    for block in visual.transformer.resblocks[:-1]:
        x = block(x)
    x_in = x.permute(1, 0, 2) if batch_first else x  # -> LND

    target = visual.transformer.resblocks[-1]
    ls_1 = getattr(target, "ls_1", torch.nn.Identity())
    ls_2 = getattr(target, "ls_2", torch.nn.Identity())
    linear = torch._C._nn.linear
    q, k, v = linear(target.ln_1(x_in), target.attn.in_proj_weight, target.attn.in_proj_bias).chunk(3, dim=-1)
    attn_output = attention_layer(q, k, v, 1)
    x = x_in + ls_1(linear(attn_output, target.attn.out_proj.weight, target.attn.out_proj.bias))
    x = x + ls_2(target.mlp(target.ln_2(x)))

    x = visual.ln_post(x.permute(1, 0, 2))  # LND -> NLD
    x = x @ visual.proj

    with torch.no_grad():
        q_out = linear(q, target.attn.out_proj.weight, target.attn.out_proj.bias)
        k_out = linear(k, target.attn.out_proj.weight, target.attn.out_proj.bias)
    return x, v, q_out, k_out, attn_output, (feah, feaw)


def grad_eclip(c, q_out, k_out, v, attn_output, map_size):
    # verbatim generate_emap.grad_eclip with withksim=True
    grad = torch.autograd.grad(c, attn_output, retain_graph=True)[0].detach()
    grad_cls = grad[:1, 0, :]
    q_cls = F.normalize(q_out[:1, 0, :], dim=-1)
    k_patch = F.normalize(k_out[1:, 0, :], dim=-1)
    cosine_qk = (q_cls * k_patch).sum(-1)
    cosine_qk = (cosine_qk - cosine_qk.min()) / (cosine_qk.max() - cosine_qk.min())
    emap = F.relu_((grad_cls * v[1:, 0, :] * cosine_qk[:, None]).detach().sum(-1))
    return emap.reshape(*map_size)


def grad_eclip_pixel_map(model, tokenizer, image: Image.Image, text: str, device) -> np.ndarray:
    '''
    Grad-ECLIP heat map of `text` on `image`, bilinearly upsampled to the
    image's own pixel size and min-max normalized to [0,1]: (H, W) float32.
    '''
    with torch.no_grad():
        text_emb = F.normalize(model.encode_text(tokenizer([text]).to(device)).float(), dim=-1)

    x = imgprocess_keepsize(image).to(device).unsqueeze(0)
    x.requires_grad_(True)  # so attn_output is on the autograd graph even if the weights are frozen
    with torch.enable_grad():
        outputs, v, q_out, k_out, attn_output, map_size = clip_encode_dense(model, x)
        img_emb = F.normalize(outputs[:, 0].float(), dim=-1)
        c = (img_emb @ text_emb.T)[0, 0]
        emap = grad_eclip(c, q_out, k_out, v, attn_output, map_size)

    w, h = image.size
    emap = F.interpolate(emap[None, None].float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    emap = emap - emap.min()
    emap = emap / (emap.max() + 1e-8)
    return emap.cpu().numpy().astype(np.float32)


def top_frac_patch_mask(pixel_map: np.ndarray, grid_h: int, grid_w: int, top_frac: float) -> np.ndarray:
    '''
    Average the pixel heat map down to a (grid_h, grid_w) patch grid (same
    exact-divide reshape as generate_clean_inference.resize_mask_to_grid),
    then keep exactly the ceil(top_frac * n) highest patches - thresholding
    at each block's own resolution rather than downsampling a binary mask, so
    it's always the top top_frac regardless of the block's grid size.
    '''
    h, w = pixel_map.shape
    scale_h, scale_w = h // grid_h, w // grid_w
    cropped = pixel_map[:grid_h * scale_h, :grid_w * scale_w]
    patch_map = cropped.reshape(grid_h, scale_h, grid_w, scale_w).mean(axis=(1, 3)).reshape(-1)
    n_keep = int(np.ceil(top_frac * patch_map.size))
    mask = np.zeros(patch_map.size, dtype=bool)
    mask[np.argsort(patch_map)[::-1][:n_keep]] = True
    return mask.reshape(grid_h, grid_w)
