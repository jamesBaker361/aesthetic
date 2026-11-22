# CASL (He et al., arXiv:2601.15441) on SDXL in the paper's own setting, to see whether it can work on SDXL at all
# apart from UnlearnCanvas's prompts and objects: real CelebA-HQ faces, an unconditional model (the empty prompt,
# no guidance), the paper's attribute text pairs (its Table 4) and its evaluation (EPR from a 40-attribute
# classifier's logits, Sec. 4.4 / App. 7.8; CLIP-Score with ViT-L/14, App. 7.8).
#
#   trajectory  each real image -> fp16-fix VAE latent z_0 -> x_{t_start}: --start invert (DDIM inversion with the
#               empty prompt, as the paper's App. 7.4) or noise (forward noise seeded by the image's index, SDEdit);
#               then --n_steps DDIM steps (trailing spacing) back to x_0. --model turbo (sdxl-turbo, the SAEs' own
#               model) or base (stable-diffusion-xl-base-1.0, the SAEs applied off their training model).
#   stage 2     (Sec. 4.2, Eqs. 6-8) dh = W z + b added to --block's output, z = the frozen SAE code of the block's
#               residual update. Trained as the paper's Eq. 7: a trajectory step t >= --t_edit of a training image,
#               the one-step x0 prediction from x_t with dh added, decoded;
#                   L = lambda_clip (1 - cos(E_I(x_edit) - E_I(x_ref), E_T(y_ref) - E_T(y_origin)))
#                       + lambda_recon |x_edit - x_ref|_1
#               x_ref = the real image (--ref real, Eq. 8) or the unedited x0 prediction (--ref recon). --n_train
#               train-split images (the paper's 1,000, Sec. 5.1), shared by every attribute. W snapshots at
#               --snapshots steps; latents ranked by W's column norm (as evaluate_unlearncanvas.py's run_casl).
#   CASL-Steer  (Eqs. 9-11, App. 7.4) the top-k latents I; h' = h + W[:, I] (alpha z_I) at every step with
#               t >= --t_edit of the full trajectory from x_{t_start}, on --n_test test-split images (Sec. 5.3's
#               32, one set for every attribute). alpha = 0 is the unedited reconstruction.
#   metrics     EPR = mean |f_c(x') - f_c(x)| / (mean over the other 39 attributes + 1e-8), f = the classifier's
#               logits, x = the reconstruction (the edit alone; EPR) or the real image (EPR_real, as the paper's
#               "Recon." row, which is the k = 0 row here); d_target_signed = the mean signed change of the target
#               (towards the attribute: No_Beard's sign flipped for Beards); clip = CLIP ViT-L/14 cosine of x' with
#               y_ref, d_clip = its change from the reconstruction; dL_clip = the training CLIP loss's mean over its
#               first 50 steps minus its mean over the 50 steps before the snapshot (> 0: the loss fell).
# -> {out_dir}/results.csv (one row per snapshot x attribute x k x alpha; rows already there are skipped),
# {out_dir}/train/{attr}.json (loss curve) + {attr}_s{step}.npz (ranking, W columns), {out_dir}/grids/ (real |
# reconstruction | each alpha at k = 1, for the first --n_grid test images). The trajectory cache
# ({cache_dir}/{pipeline}/: trajectories, reconstructions) is shared by every run with the same pipeline settings:
# fill it with one --stage prepare job first (run jobs fill what is missing too, but would race).
# Needs prepare_celebahq.py's index and train_celeba_classifier.py's classifier. Run: scripts/casl_faces_sweep.sh
import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

from experiment_helpers.gpu_details import print_details
from generate_clean_swap import load_sae
from grad_eclip_mask import load_clip
from evaluate_sae_features import load_json, save_json, save_npz, save_image
from train_celeba_classifier import classifier_input, attr_net

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=str, default="run", choices=["prepare", "run"],
                    help="prepare: only fill the trajectory cache for these pipeline settings")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--cache_dir", type=str, default="evaluation/casl_faces/cache")
parser.add_argument("--index", type=str, default="data/celebahq/attrs.csv")
parser.add_argument("--classifier", type=str, default="data/celebahq/attr_r50.pth")
parser.add_argument("--seed", type=int, default=0)
# pipeline
parser.add_argument("--model", type=str, default="turbo", choices=["turbo", "base"])
parser.add_argument("--start", type=str, default="invert", choices=["invert", "noise"])
parser.add_argument("--t_start", type=int, default=999)
parser.add_argument("--n_steps", type=int, default=4)
parser.add_argument("--t_edit", type=int, default=500, help="edit (and train on) the steps with t >= this")
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--uncond", type=str, default="empty", choices=["empty", "zeros"],
                    help="the unconditional embedding: the empty prompt's, or zeros (SDXL's CFG negative)")
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_test", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=8, help="images per batch in prepare / CASL-Steer")
# stage 2
parser.add_argument("--block", type=str, default="down_blocks.2.attentions.1")
parser.add_argument("--mode", type=str, default="diff", choices=["diff", "out"])
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--snapshots", type=int, nargs="+", default=[1000, 3000, 5000])
parser.add_argument("--lambda_clip", type=float, default=3.0)
parser.add_argument("--lambda_recon", type=float, default=1.0)
parser.add_argument("--ref", type=str, default="real", choices=["real", "recon"])
parser.add_argument("--clip_model", type=str, default="ViT-B-16")
parser.add_argument("--clip_pretrained", type=str, default="openai")
parser.add_argument("--keep", type=int, default=64, help="latents kept in each snapshot's ranking")
# CASL-Steer
parser.add_argument("--attributes", type=str, nargs="+", default=["Smiling", "Blond_Hair"])
parser.add_argument("--k_list", type=int, nargs="+", default=[1])
parser.add_argument("--alphas", type=float, nargs="+", default=[1, 32, 64, 96, 128, 160, 192])
parser.add_argument("--eval_steps", type=int, nargs="*", default=None, help="snapshots to evaluate (default all)")
parser.add_argument("--eval_clip_model", type=str, default="ViT-L-14")
parser.add_argument("--eval_clip_pretrained", type=str, default="openai")
parser.add_argument("--n_grid", type=int, default=4)

MODEL_IDS = {"turbo": "stabilityai/sdxl-turbo", "base": "stabilityai/stable-diffusion-xl-base-1.0"}
# the paper's Table 4: attribute -> (CelebA label, sign towards the attribute, y_origin, y_ref)
ATTRIBUTES = {
    "Smiling": ("Smiling", 1, "face", "smiling face"),
    "Young": ("Young", 1, "person", "young person"),
    "Beards": ("No_Beard", -1, "person", "person with beards"),
    "Big_Nose": ("Big_Nose", 1, "person", "person with big nose"),
    "Blond_Hair": ("Blond_Hair", 1, "person", "person with blond hair"),
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MIN_DIRECTION = 1e-2  # as evaluate_unlearncanvas.py's CASL_MIN_DIRECTION
CONFIG_COLUMNS = ["model", "start", "t_start", "n_steps", "t_edit", "uncond", "block", "optimizer", "lr",
                  "lambda_clip", "lambda_recon", "ref", "n_train", "n_test"]


def pipeline_key(args) -> str:
    return f"{args.model}_{args.start}_t{args.t_start}_n{args.n_steps}_s{args.size}_{args.uncond}"


def load_split(args):
    '''(train rows, test rows) of the index: fixed random samples of the official splits.'''
    df = pd.read_csv(args.index)
    root = os.path.dirname(args.index)
    df["path"] = [os.path.join(root, f) for f in df["file"]]
    out = []
    for split, n, seed in (("train", args.n_train, args.seed), ("test", args.n_test, args.seed + 1)):
        rows = df[df["split"] == split].reset_index(drop=True)
        keep = np.sort(np.random.default_rng(seed).choice(len(rows), n, replace=False))
        out.append(rows.iloc[keep].reset_index(drop=True))
    return out


def load_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return torchvision.transforms.functional.to_tensor(img)


def to_pil(img: torch.Tensor) -> Image.Image:
    return Image.fromarray((img.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))


def clip_embed(model, img: torch.Tensor) -> torch.Tensor:
    '''(B, 3, H, W) in [0, 1] -> normalized CLIP image embeddings (differentiable).'''
    x = torch.nn.functional.interpolate(img.float(), size=(224, 224), mode="bicubic", antialias=True,
                                        align_corners=False)
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    e = model.encode_image((x - mean) / std)
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_text(model, tokenizer, texts: list, device) -> torch.Tensor:
    t = model.encode_text(tokenizer(texts).to(device)).float()
    return t / t.norm(dim=-1, keepdim=True)


class FaceSDXL:
    '''
    SDXL's UNet + the fp16-fix VAE with the unconditional embedding, DDIM over the trajectory self.T, and one forward
    hook on --block whose edit is self.edit: None, {"kind": "train", "W", "b"} (dh = W z + b) or {"kind": "steer",
    "W", "latents", "alpha"} (dh = alpha W z_I); it only edits steps with t >= --t_edit (self.t, set by eps()).
    '''

    def __init__(self, args, device):
        from diffusers import StableDiffusionXLPipeline, AutoencoderKL
        self.args, self.device = args, device
        dtype = torch.float16
        pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_IDS[args.model], torch_dtype=dtype, variant="fp16")
        pipe.to(device)
        with torch.no_grad():
            if args.uncond == "zeros":
                pe = torch.zeros(1, 77, pipe.unet.config.cross_attention_dim, dtype=dtype, device=device)
                ppe = torch.zeros(1, pipe.text_encoder_2.config.projection_dim, dtype=dtype, device=device)
            else:
                pe, _, ppe, _ = pipe.encode_prompt(prompt="", device=device, num_images_per_prompt=1,
                                                   do_classifier_free_guidance=False)
        s = args.size
        self.cond = {"pe": pe, "ppe": ppe, "ids": torch.tensor([[s, s, 0, 0, s, s]], dtype=dtype, device=device)}
        self.unet = pipe.unet.requires_grad_(False)
        self.ac = pipe.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)
        self.vae.requires_grad_(False)
        self.sf = self.vae.config.scaling_factor
        self.T = [int(t) for t in np.round(np.linspace(args.t_start, 0, args.n_steps, endpoint=False))]
        self.edit, self.t, self.stats, self.sae = None, None, [], None

    def attach(self, block: str, sae):
        self.sae = sae
        return self.unet.get_submodule(block).register_forward_hook(self.hook)

    def hook(self, module, inp, out):
        e = self.edit
        if e is None or self.t < self.args.t_edit:
            return out
        o = out[0] if isinstance(out, tuple) else out
        x = o - inp[0] if self.args.mode == "diff" else o
        with torch.no_grad():
            z = self.sae.encode(x.permute(0, 2, 3, 1).float())  # frozen encoder (Sec. 4.2)
        if e["kind"] == "train":
            dh = z @ e["W"].T + e["b"]
        else:
            zi = z[..., e["latents"]]
            dh = e["alpha"] * (zi @ e["W"].T)  # Eq. 11: W (alpha 1_I ⊙ z)
            with torch.no_grad():  # does the steered latent fire, and how big is the edit?
                self.stats.append(((zi > 0).any(-1).float().mean().item(),
                                   (dh.norm(dim=-1).mean() / o.permute(0, 2, 3, 1).float().norm(dim=-1).mean()
                                    .clamp_min(1e-12)).item()))
        o = o + dh.permute(0, 3, 1, 2).to(o.dtype)
        return (o, *out[1:]) if isinstance(out, tuple) else o

    def eps(self, x: torch.Tensor, t: int) -> torch.Tensor:
        self.t = t
        n, c = len(x), self.cond
        return self.unet(x.to(self.unet.dtype), torch.full((n,), t, device=self.device),
                         encoder_hidden_states=c["pe"].expand(n, -1, -1),
                         added_cond_kwargs={"text_embeds": c["ppe"].expand(n, -1), "time_ids": c["ids"].expand(n, -1)},
                         return_dict=False)[0].float()

    def x0(self, x: torch.Tensor, t: int, eps: torch.Tensor) -> torch.Tensor:
        a = self.ac[t]
        return (x - (1 - a).sqrt() * eps) / a.sqrt()

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        return self.vae.encode((img * 2 - 1).to(self.vae.dtype)).latent_dist.mean.float() * self.sf

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (self.vae.decode((z / self.sf).to(self.vae.dtype), return_dict=False)[0].float() / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def start_latents(self, z0: torch.Tensor, seeds: list) -> torch.Tensor:
        '''x_{T[0]}: DDIM inversion of z0 (--start invert) or z0 forward-noised with each image's own seed.'''
        if self.args.start == "noise":
            noise = torch.stack([torch.randn(z0.shape[1:], generator=torch.Generator().manual_seed(int(s)))
                                 for s in seeds]).to(z0)
            a = self.ac[self.T[0]]
            return a.sqrt() * z0 + (1 - a).sqrt() * noise
        x, a_prev = z0, torch.tensor(1.0, device=self.device)
        for t in reversed(self.T):
            e = self.eps(x, t)
            x0 = (x - (1 - a_prev).sqrt() * e) / a_prev.sqrt()
            a_prev = self.ac[t]
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
        return x

    def denoise(self, x: torch.Tensor):
        '''DDIM from x_{T[0]} to x_0 -> (x_0, the latents at every T[i], (B, n_steps, 4, h, w)).'''
        traj = []
        for i, t in enumerate(self.T):
            traj.append(x)
            e = self.eps(x, t)
            x0 = self.x0(x, t, e)
            a_next = self.ac[self.T[i + 1]] if i + 1 < len(self.T) else torch.tensor(1.0, device=self.device)
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * e
        return x, torch.stack(traj, 1)


class AttrClassifier:
    def __init__(self, path: str, device):
        ck = torch.load(path, map_location="cpu")
        self.attrs, self.size, self.device = ck["attrs"], ck["size"], device
        self.net = attr_net(len(self.attrs))
        self.net.load_state_dict(ck["state_dict"])
        self.net.to(device).eval()

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> np.ndarray:
        return self.net(classifier_input(img.to(self.device), self.size)).float().cpu().numpy()


def cache_paths(args, idx: int):
    root = os.path.join(args.cache_dir, pipeline_key(args))
    return os.path.join(root, "traj", f"{idx}.npy"), os.path.join(root, "recon", f"{idx}.png")


@torch.no_grad()
def prepare(args, m: FaceSDXL, rows: pd.DataFrame):
    '''Fills the trajectory cache: each image's unedited trajectory latents and its reconstruction.'''
    todo = [r for r in rows.itertuples() if not all(os.path.exists(p) for p in cache_paths(args, r.idx))]
    print(f"prepare {pipeline_key(args)}: {len(todo)} of {len(rows)} images to do, T = {m.T}")
    for d in ("traj", "recon"):
        os.makedirs(os.path.join(args.cache_dir, pipeline_key(args), d), exist_ok=True)
    m.edit = None
    for n in range(0, len(todo), args.batch_size):
        batch = todo[n:n + args.batch_size]
        img = torch.stack([load_image(r.path, args.size) for r in batch]).to(m.device)
        x0, traj = m.denoise(m.start_latents(m.encode(img), [r.idx for r in batch]))
        recon = m.decode(x0)
        for r, tr, rc in zip(batch, traj, recon):
            traj_path, recon_path = cache_paths(args, r.idx)
            tmp = f"{traj_path[:-len('.npy')]}.tmp{os.getpid()}.npy"
            np.save(tmp, tr.half().cpu().numpy())
            os.replace(tmp, traj_path)
            save_image(to_pil(rc), recon_path)
        if n % (25 * args.batch_size) == 0:
            print(f"  prepare {n}/{len(todo)}")


def load_traj(args, idx: int) -> torch.Tensor:
    return torch.from_numpy(np.load(cache_paths(args, idx)[0]).astype(np.float32))


def snapshot_path(args, attr: str, step: int) -> str:
    return os.path.join(args.out_dir, "train", f"{attr}_s{step}.npz")


def train(args, m: FaceSDXL, attr: str, train_rows: pd.DataFrame, clip, tokenizer) -> list:
    '''CASL stage 2 for one attribute (see the header); saves every snapshot and returns the loss log.'''
    log_path = os.path.join(args.out_dir, "train", f"{attr}.json")
    if os.path.exists(log_path) and all(os.path.exists(snapshot_path(args, attr, s)) for s in args.snapshots):
        return load_json(log_path, {})["log"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _, _, y_origin, y_ref = ATTRIBUTES[attr]
    text = clip_text(clip, tokenizer, [y_origin, y_ref], m.device)
    text_dir = (text[1] - text[0]) / (text[1] - text[0]).norm()
    d_model, n_dirs = m.sae.decoder.weight.shape
    # zero init, as run_casl: a latent that never fires gets no gradient and stays out of the ranking
    W = torch.zeros(d_model, n_dirs, device=m.device, requires_grad=True)
    b = torch.zeros(d_model, device=m.device, requires_grad=True)
    opt = (torch.optim.SGD([W, b], lr=args.lr, momentum=args.momentum) if args.optimizer == "sgd"
           else torch.optim.Adam([W, b], lr=args.lr))
    edit_steps = [j for j, t in enumerate(m.T) if t >= args.t_edit]
    assert edit_steps, f"no step of T = {m.T} has t >= --t_edit {args.t_edit}"
    rng = np.random.default_rng(args.seed)
    log, skipped = [], 0
    m.unet.enable_gradient_checkpointing(); m.vae.enable_gradient_checkpointing()
    m.unet.train(); m.vae.train()  # some diffusers versions only checkpoint in train mode (no dropout in either)
    try:
        for step in range(max(args.snapshots)):
            r = train_rows.iloc[int(rng.integers(len(train_rows)))]
            j = int(rng.choice(edit_steps))
            t, x = m.T[j], load_traj(args, r.idx)[j:j + 1].to(m.device)
            with torch.no_grad():
                if args.ref == "real":
                    ref = load_image(r.path, args.size)[None].to(m.device)
                else:
                    m.edit = None
                    ref = m.decode(m.x0(x, t, m.eps(x, t)))
                ref_emb = clip_embed(clip, ref)
            m.edit = {"kind": "train", "W": W, "b": b}
            W.grad, b.grad = None, None
            with torch.enable_grad():
                img = m.decode(m.x0(x, t, m.eps(x, t)))
                d_img = (clip_embed(clip, img) - ref_emb)[0]
                loss_clip = 1 - (d_img @ text_dir) / d_img.norm().clamp_min(MIN_DIRECTION)
                loss_recon = (img * 2 - ref * 2).abs().mean()  # [-1, 1]
                (args.lambda_clip * loss_clip + args.lambda_recon * loss_recon).backward()
            m.edit = None
            if torch.isfinite(W.grad).all() and torch.isfinite(b.grad).all():
                opt.step()
            else:
                skipped += 1  # fp16 overflow in the UNet backward: drop this step
            log.append([loss_clip.item(), loss_recon.item(), t])
            if step % 50 == 0:
                print(f"  {attr} step {step} t {t}: clip {log[-1][0]:.4f} recon {log[-1][1]:.4f} "
                      f"|W| {W.detach().norm().item():.3g}" + (f" ({skipped} non-finite skipped)" if skipped else ""))
            if step + 1 in args.snapshots:
                scores = W.detach().norm(dim=0).double().cpu().numpy()
                order = [int(i) for i in np.argsort(-scores) if scores[i] > 0][:args.keep]
                save_npz(snapshot_path(args, attr, step + 1), latents=np.asarray(order, dtype=np.int64),
                         W=W.detach()[:, order].cpu().numpy(), scores=scores[order])
                save_json(log_path, {"log": log, "skipped": skipped, "y_origin": y_origin, "y_ref": y_ref})
                print(f"  {attr} snapshot {step + 1}: top latents {order[:8]}")
    finally:
        m.edit = None
        m.unet.disable_gradient_checkpointing(); m.vae.disable_gradient_checkpointing()
        m.unet.eval(); m.vae.eval()
    del W, b, opt
    torch.cuda.empty_cache()
    return log


def epr(f_new: np.ndarray, f_old: np.ndarray, ti: int):
    '''(EPR, mean |target change|, mean |non-target change|) over the images, logits (N, n_attrs).'''
    d = np.abs(f_new - f_old).mean(0)
    other = np.delete(d, ti).mean()
    return float(d[ti] / (other + 1e-8)), float(d[ti]), float(other)


def save_grid(path: str, columns: list, n: int, tile: int = 256):
    '''columns: [(n, 3, H, W)] -> one row per image: real | reconstruction | each alpha.'''
    grid = Image.new("RGB", (tile * len(columns), tile * n))
    for c, col in enumerate(columns):
        for i in range(n):
            grid.paste(to_pil(col[i]).resize((tile, tile), Image.BICUBIC), (c * tile, i * tile))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(grid, path)


@torch.no_grad()
def steer(args, m: FaceSDXL, x_start: torch.Tensor, W: torch.Tensor, latents: list, alpha: float):
    '''CASL-Steer from each image's x_{t_start} -> (edited images on the CPU, fire rate, |dh| / |h|).'''
    m.edit, m.stats, out = {"kind": "steer", "W": W, "latents": latents, "alpha": alpha}, [], []
    for n in range(0, len(x_start), args.batch_size):
        x0, _ = m.denoise(x_start[n:n + args.batch_size].to(m.device))
        out.append(m.decode(x0).cpu())
    m.edit = None
    stats = np.asarray(m.stats)
    return torch.cat(out), float(stats[:, 0].mean()), float(stats[:, 1].mean())


def main(args):
    device = "cuda"
    train_rows, test_rows = load_split(args)
    m = FaceSDXL(args, device)
    prepare(args, m, pd.concat([train_rows, test_rows]))
    if args.stage == "prepare":
        return
    assert args.out_dir, "--out_dir is needed to run"
    os.makedirs(args.out_dir, exist_ok=True)
    save_json(os.path.join(args.out_dir, "config.json"), vars(args))
    sae = load_sae(args.block).to(device).requires_grad_(False)
    m.attach(args.block, sae)
    clip, tokenizer = load_clip(device, args.clip_model, args.clip_pretrained)
    clip.requires_grad_(False)
    eclip, etokenizer = load_clip(device, args.eval_clip_model, args.eval_clip_pretrained)
    clf = AttrClassifier(args.classifier, device)

    real = torch.stack([load_image(p, args.size) for p in test_rows["path"]])
    recon = torch.stack([load_image(cache_paths(args, i)[1], args.size) for i in test_rows["idx"]])
    x_start = torch.stack([load_traj(args, i)[0] for i in test_rows["idx"]])
    f_real, f_recon = clf(real), clf(recon)
    e_recon = torch.cat([clip_embed(eclip, recon[n:n + 16].to(device)) for n in range(0, len(recon), 16)])

    results_path = os.path.join(args.out_dir, "results.csv")
    rows = pd.read_csv(results_path).to_dict("records") if os.path.exists(results_path) else []
    done = {(r["attribute"], int(r["step"]), int(r["k"]), float(r["alpha"])) for r in rows}
    config = {"run": os.path.basename(os.path.normpath(args.out_dir)),
              **{c: getattr(args, c) for c in CONFIG_COLUMNS}}

    def add(row):
        rows.append({**config, **row})
        tmp = f"{results_path}.tmp{os.getpid()}"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, results_path)

    for attr in args.attributes:
        label, sign, _, y_ref = ATTRIBUTES[attr]
        ti = clf.attrs.index(label)
        log = np.asarray(train(args, m, attr, train_rows, clip, tokenizer))
        e_text = clip_text(eclip, etokenizer, [y_ref], device)[0]
        clip_recon = (e_recon @ e_text).mean().item()
        for step in args.eval_steps or args.snapshots:
            stats = {"clip_loss_first": float(log[:50, 0].mean()),
                     "clip_loss_last": float(log[max(0, step - 50):step, 0].mean())}
            stats["dL_clip"] = stats["clip_loss_first"] - stats["clip_loss_last"]
            if (attr, step, 0, 0.0) not in done:  # the paper's "Recon." row: the reconstruction against the real image
                e, dt, dn = epr(f_recon, f_real, ti)
                add({"attribute": attr, "step": step, "k": 0, "alpha": 0.0, "latents": "", "EPR": np.nan,
                     "EPR_real": e, "d_target": np.nan, "d_target_signed": np.nan, "d_nontarget": np.nan,
                     "clip": clip_recon, "d_clip": 0.0, "fire": np.nan, "dh_ratio": np.nan, **stats})
            with np.load(snapshot_path(args, attr, step)) as d:
                order, W_all = [int(j) for j in d["latents"]], torch.from_numpy(d["W"]).float().to(device)
            for k in args.k_list:
                if k > len(order):
                    print(f"  ! {attr} s{step}: only {len(order)} latents with a nonzero column, k = {k} skipped")
                    continue
                grid = [real, recon] if k == 1 else None
                for alpha in args.alphas:
                    if (attr, step, k, float(alpha)) in done:
                        grid = None  # this run only has part of the row: no grid
                        continue
                    edited, fire, dh_ratio = steer(args, m, x_start, W_all[:, :k], order[:k], alpha)
                    f_edit = clf(edited)
                    e, dt, dn = epr(f_edit, f_recon, ti)
                    e_real, _, _ = epr(f_edit, f_real, ti)
                    e_edit = torch.cat([clip_embed(eclip, edited[n:n + 16].to(device))
                                        for n in range(0, len(edited), 16)])
                    clip_edit = (e_edit @ e_text).mean().item()
                    add({"attribute": attr, "step": step, "k": k, "alpha": float(alpha),
                         "latents": " ".join(map(str, order[:k])), "EPR": e, "EPR_real": e_real, "d_target": dt,
                         "d_target_signed": float(sign * (f_edit[:, ti] - f_recon[:, ti]).mean()),
                         "d_nontarget": dn, "clip": clip_edit, "d_clip": clip_edit - clip_recon, "fire": fire,
                         "dh_ratio": dh_ratio, **stats})
                    print(f"  {attr} s{step} k{k} alpha {alpha:g}: EPR {e:.3f} (target {dt:.3f}, other {dn:.3f}), "
                          f"dCLIP {clip_edit - clip_recon:+.4f}, fires on {fire:.1%} of patches, |dh|/|h| "
                          f"{dh_ratio:.2e}")
                    if grid is not None:
                        grid.append(edited)
                if grid is not None:
                    save_grid(os.path.join(args.out_dir, "grids", f"{attr}_s{step}.jpg"), grid,
                              min(args.n_grid, len(real)))


if __name__ == "__main__":
    print_details()
    start = time.time()
    args = parser.parse_args()
    print(args)
    main(args)
    print(f"all done! {(time.time() - start) / 3600:.2f} hours")
