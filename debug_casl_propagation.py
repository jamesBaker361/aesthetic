# Does an edit at a UNet block reach the image when CASL trains? scripts/uc_casl_debug_down21.sh showed, at down.2.1,
# |dh| / |h| of 2-8 at the block but |dI| 0 and a pixel change of 0.0002 on every training step, while CASL-Steer's
# generation (make_steer_hook) can destroy images from the same block. This adds one fixed vector b to the block's
# output at every patch (W = 0, so dh = b, no SAE ranking involved) and measures the pixel change against the
# unedited image, five ways:
#   train   CaslTrainer.generate with grad on - exactly as CaslTrainer.step (gradient checkpointing, train mode)
#   nockpt  the same with gradient checkpointing off
#   nograd  the same under no_grad (checkpointing inactive)
#   eval    the same with the UNet / VAE in eval mode (checkpointing still on)
#   pipe    the eval-time path: generate() with a forward hook adding the same b (as make_steer_hook)
# at the bottleneck and at down.2.1. A change near 0 for "train" but not "pipe" means training never sees its edit.
# Run on a GPU from the repo root: python debug_casl_propagation.py   (scripts/uc_casl_debug_propagation.sh)
import numpy as np
import torch

from evaluate_unlearncanvas import parser, UCModels, CaslTrainer
from evaluate_sae_features import generate

parser.add_argument("--edit_norm", type=float, default=50.0, help="|b| per patch")
parser.add_argument("--prompt", type=str, default="A Cats image in Van_Gogh style.")
args = parser.parse_args()
args.block_list = args.block_list or ["mid_block.attentions.0", "down_blocks.2.attentions.1"]
device = torch.device("cuda")
models = UCModels(args, device)
seed = 188


def change(a, b):
    '''mean |pixel difference| in [0, 1] units'''
    return float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean())


def as_np(img):
    '''(1, 3, H, W) tensor in [0, 1] -> (H, W, 3) float array'''
    return img.detach().float()[0].permute(1, 2, 0).cpu().numpy()


results = []
with CaslTrainer(args, models) as tr:
    embeds = tr.embed_prompt(args.prompt)
    tr.offload_text_encoders()
    with torch.no_grad():
        orig = as_np(tr.generate(embeds, seed))
    for block in args.block_list:
        sae = models.get_sae(block).requires_grad_(False)
        d_model, n_dirs = sae.decoder.weight.shape
        g = torch.Generator().manual_seed(0)
        direction = torch.randn(d_model, generator=g)
        W = torch.zeros(d_model, n_dirs, device=device, requires_grad=True)
        b = (direction / direction.norm() * args.edit_norm).to(device).requires_grad_(True)

        def run(name, grad=True):
            handle = tr.attach(block, sae, W, b)
            try:
                with torch.set_grad_enabled(grad):
                    img = tr.generate(embeds, seed)
                    if grad:
                        img.float().mean().backward()  # as step(): the backward re-runs checkpointed blocks
            finally:
                handle.remove()
            stats = getattr(tr, "stats", {}) or {}
            results.append((block, name, change(as_np(img), orig), stats.get("dh_ratio", float("nan")), tr.calls,
                            None if b.grad is None else float(b.grad.norm())))
            W.grad, b.grad = None, None

        run("train")
        run("nograd", grad=False)
        tr.unet.disable_gradient_checkpointing(); tr.vae.disable_gradient_checkpointing()
        run("nockpt")
        tr.unet.enable_gradient_checkpointing(); tr.vae.enable_gradient_checkpointing()
        tr.unet.eval(); tr.vae.eval()
        run("eval")
        tr.unet.train(); tr.vae.train()

# the eval-time path (make_steer_hook's): a plain forward hook through generate()
pipe = models.get_pipe()
plain = generate(pipe, args.prompt, seed, args)
for block in args.block_list:
    d_model = models.get_sae(block).decoder.weight.shape[0]
    g = torch.Generator().manual_seed(0)
    direction = torch.randn(d_model, generator=g)
    vec = (direction / direction.norm() * args.edit_norm).to(device)
    seen = {"ratio": float("nan"), "calls": 0}

    def hook(module, inp, out, vec=vec, seen=seen):
        o = out[0] if isinstance(out, tuple) else out
        seen["calls"] += 1
        seen["ratio"] = float(vec.norm() / o.permute(0, 2, 3, 1).float().norm(dim=-1).mean().clamp_min(1e-12))
        o = (o.float() + vec.view(1, -1, 1, 1)).to(o.dtype)
        return (o, *out[1:]) if isinstance(out, tuple) else o

    with torch.no_grad():
        img = generate(pipe, args.prompt, seed, args, {f"unet.{block}": hook})
    results.append((block, "pipe", change(img, plain) / 255.0, seen["ratio"], seen["calls"], None))

print(f"\nedit: a fixed vector, |b| = {args.edit_norm:g} per patch, prompt {args.prompt!r}, seed {seed}, "
      f"{args.num_inference_steps} step(s)")
print(f"{'block':30s} {'path':7s} {'pixel change':>13s} {'|dh|/|h|':>9s} {'hook calls':>10s} {'|grad b|':>9s}")
for block, name, ch, ratio, calls, gb in results:
    print(f"{block:30s} {name:7s} {ch:13.5f} {ratio:9.2f} {calls:10d} {'-' if gb is None else f'{gb:9.3g}':>9s}")
print("\npixel change: mean |edited - unedited| in [0, 1]. 'train' ~0 while 'pipe' is not -> CASL training never "
      "sees its edit at that block; compare nockpt / nograd / eval to find which setting causes it.")
