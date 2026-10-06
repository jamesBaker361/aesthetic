# Like-for-like compute of the per-edit hyperparameter search: our --auto_k + --auto_gamma vs SAeUron's
# grid search (arXiv:2501.18052, SAeUron/scripts/sweep_cls_distr.py + find_best_params_cls_sweep.py),
# both in THIS pipeline (sdxl-turbo, --size, --num_inference_steps, --guidance_scale, the UnlearnCanvas
# answer set, the ViT-L/16 classifiers).
#
# SAeUron's search: every (gamma, percentile) setting of the grid (7 x 3 = 21 by default) is generated on
# the whole answer set and scored (UA + IRA) for every concept x block, then the best setting is kept:
#     C_grid = |grid| x N_answer x (F_gen(block) + F_cls)            per concept x block
# F_gen = one generate() call with the SAE removal hook at the block (text encoders, UNet, SAE, VAE),
# F_cls = the style + object classifiers on one image, both measured with torch's FlopCounterMode.
#
# Ours, per concept x mask method x block (no images generated):
#   auto-k   the per-latent 1-D probes (sparse_probe.select_bce_and_f1, counted analytically) and every
#            LogisticRegression the search fits/predicts with (re-run here on the cached codes, iterations
#            read from n_iter_), for the run's --rules
#   auto-γ   StandardScaler + dense probe (re-fit here for n_iter_), one SAE encode of the held-out
#            patches, and per γ tried (the curve length cached in {out_dir}/auto_gamma/) a dense SAE decode
#            + probe predict
#   masks    SAM3 on each discovery image of the concept (FlopCounterMode on one image), shared by the
#            concept's blocks and split evenly over them
# Shared by both and left out: discovery images/activations, the unedited answer set, the final scoring.
#
# FlopCounterMode only counts matmul / conv / attention FLOPs, while the CPU side counts every multiply
# and add with upper-bound iteration constants (--cg_per_newton, --cd_passes, --lbfgs_evals), so the
# ratio printed is a lower bound on how much cheaper our search is.
#
# Run with the same flags as the run being costed, e.g. for scripts/uc_top6_all_objects.sh's f1/accuracy:
#   python search_flops.py --target_styles --object_discover_prompt_file prompt_dir/dream_prompts.txt \
#     --discover_seeds 0 --eval_seeds 188 --cache_dir evaluation/uc/cache \
#     --out_dir evaluation/uc_auto/f1_all_autok_accuracy_autog --object_mask_methods sam --rules f1 \
#     --eval_scope all --n_random_controls 0 --auto_k --auto_k_metric accuracy --auto_gamma --remove_mode saeuron
# Writes {out_dir}/search_flops.json.

import os
import json
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

from attribution import DEFAULT_BLOCK_LIST
from evaluate_sae_features import generate, make_zero_hook, load_block_codes, load_json, sam_mask
from evaluate_unlearncanvas import (
    parser, CLASSES, STYLES, UCModels, discover_entries, concept_entries, concept_methods,
    answer_entries, load_features, variants, auto_gamma_path, auto_gamma_key, auto_rules,
    saeuron_scores, image_mean_codes, top_n, patch_labels, sam_query, load_attribution,
)
from sparse_probe import select_bce_and_f1, smallest_k

parser.add_argument("--sweep_gammas", nargs="*", type=float, default=[-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0],
                    help="SAeUron's multiplier grid (sweep_cls_distr.py)")
parser.add_argument("--sweep_percentiles", nargs="*", type=float, default=[99.99, 99.995, 99.999],
                    help="SAeUron's feature-percentile grid, i.e. its tau (sweep_cls_distr.py)")
parser.add_argument("--cg_per_newton", type=int, default=10,
                    help="liblinear L2 (trust-region Newton): CG steps per Newton iteration, upper bound")
parser.add_argument("--cd_passes", type=int, default=10,
                    help="liblinear L1 (newGLMNET): coordinate-descent passes per outer iteration, upper bound")
parser.add_argument("--lbfgs_evals", type=int, default=2, help="lbfgs: loss + gradient evaluations per iteration")


# ---------------------------------------------------------------- CPU FLOP counting

class Counter:
    def __init__(self):
        self.flops = 0.0


COUNTER = Counter()


def nnz_of(X) -> int:
    return int(X.nnz) if hasattr(X, "nnz") else int(np.prod(X.shape))


@contextmanager
def count_sklearn(args):
    '''Adds every LogisticRegression fit / decision_function inside the block to COUNTER.'''
    from sklearn.linear_model import LogisticRegression
    fit, decision = LogisticRegression.fit, LogisticRegression.decision_function

    def counted_fit(self, X, y, *a, **kw):
        out = fit(self, X, y, *a, **kw)
        nnz, iters = nnz_of(X), int(np.max(self.n_iter_))
        l1 = self.penalty == "l1" or (getattr(self, "l1_ratio", None) or 0) > 0
        if self.solver == "liblinear" and l1:
            per_iter = (1 + args.cd_passes) * 4 * nnz
        elif self.solver == "liblinear":
            per_iter = (1 + args.cg_per_newton) * 4 * nnz  # f + g, then Hessian-vector products
        else:
            per_iter = args.lbfgs_evals * 4 * nnz  # X @ w and X.T @ r per evaluation
        COUNTER.flops += iters * per_iter
        return out

    def counted_decision(self, X, *a, **kw):
        COUNTER.flops += 2 * nnz_of(X)
        return decision(self, X, *a, **kw)

    LogisticRegression.fit, LogisticRegression.decision_function = counted_fit, counted_decision
    try:
        yield
    finally:
        LogisticRegression.fit, LogisticRegression.decision_function = fit, decision


def per_latent_probe_flops(val: np.ndarray, n_dirs: int, n_steps: int) -> float:
    '''fit_sparse_1d_ridge_logistic + the confusion counts: ~20 ops per nonzero and ~30 per latent a step.'''
    nnz = int((val > 0).sum())
    return n_steps * (20 * nnz + 30 * n_dirs) + 25 * nnz


# ---------------------------------------------------------------- GPU FLOPs (FlopCounterMode)

def gpu_flops(fn) -> float:
    with FlopCounterMode(display=False) as fc:
        fn()
    return float(fc.get_total_flops())


def measure_gpu(args, models, entries, targets, block_list):
    e = entries[concept_entries(args, entries, *targets[0])[0]]
    prompt = answer_entries(args)[0]["prompt"]
    out = {"gen": {}}
    pipe = models.get_pipe()
    for block in block_list:
        sae = models.get_sae(block)
        hook = make_zero_hook(sae, [0, 1, 2], args.mode, args.start_step, args.end_step, models.device, scale=-10.0)
        with torch.no_grad():
            out["gen"][block] = gpu_flops(lambda: generate(pipe, prompt, 0, args, {f"unet.{block}": hook}))
    uc = models.get_uc()
    out["cls"] = gpu_flops(lambda: uc.batch([e["image"]]))
    out["sam"] = None
    if any("sam" in concept_methods(args, t) for t, _ in targets):
        try:
            sam = models.get_sam()
            image = Image.open(e["image"]).convert("RGB")
            with torch.no_grad():
                out["sam"] = gpu_flops(lambda: sam_mask(sam, image, sam_query(targets[0][1]), models.device))
        except Exception as err:  # SAM3 ops FlopCounterMode can't trace
            print(f"! could not measure SAM3 ({err!r}) - mask cost left out")
    models.free()
    return out


# ---------------------------------------------------------------- our search, re-run with counting

def auto_k_flops(args, entries, targets, block_list) -> dict:
    '''{(concept, method, block): FLOPs of the probe stage incl. the auto-k search} for the run's rules.'''
    out = {}
    for block in block_list:
        idx_all, val_all, owner, (gh, gw), n_dirs = load_block_codes(entries, block)
        means = image_mean_codes(idx_all, val_all, owner, len(entries), n_dirs) if "saeuron" in args.rules else None
        for ctype, concept in targets:
            for method in concept_methods(args, ctype):
                own = concept_entries(args, entries, ctype, concept)
                labels = np.zeros(len(owner), dtype=bool)
                for n in own:
                    labels[owner == n] = patch_labels(args, entries[n], ctype, concept, method, gh, gw).reshape(-1)
                rows = np.ones(len(owner), dtype=bool) if args.negatives == "all" else np.isin(owner, own)
                labels = labels[rows]
                if labels.all() or not labels.any():
                    continue
                COUNTER.flops = per_latent_probe_flops(val_all[rows], n_dirs, args.bce_newton_steps)
                with count_sklearn(args):
                    auto = {"rules": auto_rules(args), "groups": owner[rows], "frac": args.auto_k_frac,
                            "metric": args.auto_k_metric, "max_k": args.auto_k_max, "seed": args.seed,
                            "top_n": top_n(args)}
                    select_bce_and_f1(idx_all[rows], val_all[rows], labels, n_dirs, args.bce_ridge,
                                      args.bce_newton_steps, auto=auto)
                    orders = []
                    if "saeuron" in args.rules:
                        orders.append(saeuron_scores(args, entries, means, ctype, concept)[0])
                    if "attribution" in args.rules and load_attribution(args, concept).get(block):
                        orders.append(load_attribution(args, concept)[block]["order"])  # ranking cost not counted
                    for order in orders:
                        smallest_k(idx_all[rows], val_all[rows], labels, owner[rows], n_dirs, order[:args.auto_k_max],
                                   frac=args.auto_k_frac, metric=args.auto_k_metric, seed=args.seed)
                out[(concept, method, block)] = COUNTER.flops
        print(f"auto-k @ {block}: done")
    return out


def auto_gamma_flops(args, entries, var_list, block_list) -> dict:
    '''{(concept, method, block): FLOPs of run_auto_gamma for that group's edits} (same split and probe).'''
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    learned = [v for v in var_list if v["kind"] != "base" and v["method"] != "random"]
    out = {}
    for block in block_list:
        groups = {}
        for v in learned:
            if v["block"] == block:
                groups.setdefault((v["concept_type"], v["subject"], v["method"]), []).append(v)
        n_dirs = None
        for (ctype, concept, method), vs in groups.items():
            images = []
            for n in concept_entries(args, entries, ctype, concept):
                e = entries[n]
                with np.load(e["embedding"]) as d:
                    x = d[f"saved_output.{block}"][0]
                    x = x - d[f"saved_input.{block}"][0] if args.mode == "diff" else x
                if n_dirs is None:
                    with np.load(e["sparse"]) as s:
                        n_dirs = int(s[f"{block}__n_dirs"])
                x = x.transpose(1, 2, 0).astype(np.float32)
                labels = patch_labels(args, e, ctype, concept, method, x.shape[0], x.shape[1]).reshape(-1)
                images.append((x, labels.astype(bool)))
            rng = np.random.default_rng(args.seed)
            test_ids = set(rng.choice(len(images), max(1, int(round(0.2 * len(images)))), replace=False).tolist())
            train = [im for i, im in enumerate(images) if i not in test_ids] or images
            n_te = sum(x.shape[0] * x.shape[1] for i, (x, _) in enumerate(images) if i in test_ids)
            Xtr = np.concatenate([x.reshape(-1, x.shape[-1]) for x, _ in train])
            ytr = np.concatenate([y for _, y in train])
            d = Xtr.shape[1]
            if ytr.all() or not ytr.any():
                continue
            COUNTER.flops = 5 * Xtr.size  # scaler fit + transform
            with count_sklearn(args):
                LogisticRegression(C=1.0, max_iter=2000).fit(StandardScaler().fit_transform(Xtr), ytr)
            flops = COUNTER.flops
            flops += 2 * n_te * d * n_dirs + 4 * n_te * d  # SAE encode of the held-out patches, "before" predict
            cache = load_json(auto_gamma_path(args, concept, method), {})
            for v in vs:
                hit = cache.get(auto_gamma_key(args, v))
                n_evals = len(hit["curve"]) if hit and "curve" in hit else 2 + args.auto_gamma_steps
                # per gamma: dense decode of the edited codes, relative-change norms, scaler + probe predict
                flops += n_evals * (2 * n_te * n_dirs * d + 8 * n_te * d)
            out[(concept, method, block)] = flops
        print(f"auto-gamma @ {block}: done")
    return out


# ---------------------------------------------------------------- main

def fmt(f: float) -> str:
    for s, unit in [(1e18, "EFLOP"), (1e15, "PFLOP"), (1e12, "TFLOP"), (1e9, "GFLOP")]:
        if f >= s:
            return f"{f / s:.3g} {unit}"
    return f"{f:.3g} FLOP"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cache_dir = args.cache_dir or args.out_dir
    block_list = args.block_list if args.block_list else list(DEFAULT_BLOCK_LIST)
    args.object_list = args.object_list or list(CLASSES)
    if args.n_objects > 0:
        args.object_list = args.object_list[:args.n_objects]
    args.style_list = args.style_list or list(STYLES)
    args.eval_objects = args.eval_objects or args.object_list
    args.eval_styles = args.eval_styles or args.style_list
    target_objects = args.object_list if args.target_objects is None else args.target_objects
    target_styles = args.style_list if args.target_styles is None else args.target_styles
    targets = [("object", o) for o in target_objects] + [("style", s) for s in target_styles]
    if args.limit > 0:
        targets = targets[:args.limit]
    entries = discover_entries(args, args.object_list, args.style_list,
                               style_targets=any(t == "style" for t, _ in targets))
    assert args.auto_k and args.auto_gamma, "cost a run made with --auto_k --auto_gamma"
    if "attribution" in args.rules:
        print("! attribution rule: its ranking (gradients through SDXL + classifier) is not counted, only the search")
    if any(m == "grad_eclip" for t, _ in targets for m in concept_methods(args, t)):
        print("! grad_eclip masks are not counted")

    var_list = variants(args, load_features(args, targets), {}, targets, block_list)
    gpu = measure_gpu(args, UCModels(args, device), entries, targets, block_list)
    ak = auto_k_flops(args, entries, targets, block_list)
    ag = auto_gamma_flops(args, entries, var_list, block_list)

    grid = len(args.sweep_gammas) * len(args.sweep_percentiles)
    n_answer = len(answer_entries(args))
    keys = sorted(set(ak) | set(ag))
    sam_per_key = {}
    if gpu["sam"]:
        for concept, method, block in keys:
            if method == "sam":
                n_img = len(concept_entries(args, entries, *next(t for t in targets if t[1] == concept)))
                sam_per_key[(concept, method, block)] = n_img * gpu["sam"] / len(block_list)
    # SAeUron needs no masks: one grid search per concept x block (summed over the run's mask methods here,
    # as our side is, so a run with two mask methods compares two searches with two)
    saeuron = {k: grid * n_answer * (gpu["gen"][k[2]] + gpu["cls"]) for k in keys}
    ours = {k: ak.get(k, 0.0) + ag.get(k, 0.0) + sam_per_key.get(k, 0.0) for k in keys}
    tot = {"saeuron_grid": sum(saeuron.values()), "auto_k": sum(ak.values()), "auto_gamma": sum(ag.values()),
           "sam_masks": sum(sam_per_key.values()), "ours": sum(ours.values())}
    n = len(keys)

    print(f"\nlike-for-like search cost - {n} edits (concept x mask method x block), rules {args.rules}")
    print(f"pipeline: sdxl-turbo {args.size}px, {args.num_inference_steps} step(s), guidance {args.guidance_scale:g}; "
          f"one grid setting = {n_answer} answer images; per image: generate "
          + ", ".join(f"{b.replace('_blocks', '').replace('.attentions', '')} {fmt(f)}" for b, f in gpu["gen"].items())
          + f", classifiers {fmt(gpu['cls'])}" + (f", SAM3 {fmt(gpu['sam'])}" if gpu["sam"] else ""))
    rows = [(f"SAeUron grid ({len(args.sweep_gammas)} γ x {len(args.sweep_percentiles)} τ = {grid} settings)",
             tot["saeuron_grid"]),
            ("ours: auto-k", tot["auto_k"]), ("ours: auto-γ", tot["auto_gamma"]),
            ("ours: SAM masks", tot["sam_masks"]), ("ours: total", tot["ours"])]
    print(f"\n{'':52s}{'per edit':>14s}{'run total':>14s}")
    for name, f in rows:
        print(f"{name:52s}{fmt(f / n):>14s}{fmt(f):>14s}")
    print(f"\nSAeUron grid / ours: {tot['saeuron_grid'] / tot['ours']:.3g}x  "
          f"(without masks: {tot['saeuron_grid'] / (tot['auto_k'] + tot['auto_gamma']):.3g}x)")

    path = os.path.join(args.out_dir, "search_flops.json")
    with open(path, "w") as f:
        json.dump({"totals": tot, "n_edits": n, "grid": grid, "n_answer": n_answer, "per_image": gpu,
                   "per_edit": {"|".join(k): {"saeuron_grid": saeuron[k], "auto_k": ak.get(k, 0.0),
                                              "auto_gamma": ag.get(k, 0.0), "sam_masks": sam_per_key.get(k, 0.0)}
                                for k in keys}}, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main(parser.parse_args())
