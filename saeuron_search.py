# SAeUron's own hyperparameter search (arXiv:2501.18052; SAeUron/scripts/sweep_cls_distr.py +
# find_best_params_cls_sweep.py) run in this pipeline (sdxl-turbo, our SAEs, the UnlearnCanvas classifiers),
# instead of reusing the tau / gamma their Table 5 found for SD-1.5. Objects only, as their class sweep.
#
# Per object x block, every (percentile, multiplier) of their grid:
#   features   SAeUron's importance score (saeuron_scores: share of mean activation on the object's
#              discovery images minus the share on the other objects'), and the features whose score is above
#              torch.quantile(scores, percentile / 100) over EVERY latent (get_percentile_threshold). On our
#              5120-latent SAEs, 99.99 / 99.995 / 99.999 all keep the top-1 feature; settings that keep the
#              same features are generated once.
#   edit       each kept latent -> latent x multiplier x its mean activation on the object, only where it is above
#              its mean over all objects (their SAEMaskedUnlearningHook = our --remove_mode saeuron
#              --saeuron_mask), at the block
#   prompts    their sweep prompts, not our answer set: each object's UnlearnCanvas anchor prompt i +
#              " in {style} style." for the first --limit_themes styles (Seed_Images skipped) -> 20 x 49 = 980
#   score      the UnlearnCanvas object classifier: UA = the object's images not classified as it, IRA = mean
#              per-object accuracy over the other objects (their accuracy_unlearncanvas_cls_sweep_fast.py)
# -> {out_dir}/saeuron_search/{block}.json: {object: [{"percentiles", "tau", "latents", "gamma", "UA", "IRA",
# "score" = (UA + IRA) / 2}]}. evaluate_unlearncanvas.py --saeuron_search_dir picks, per object, the best score
# over every block and setting (find_best_params_cls_sweep.py, with the block searched too) and runs it on the
# benchmark answer set.
#
# Images: {out_dir}/saeuron_search/{block}/t{tau}_g{gamma}/{object}/{prompt object}_{style}.jpg, seed
# --search_seed + prompt index (their sweep draws every prompt from one generator seeded 42). One block per
# job (--block_list) so the blocks run in parallel; every image and score is cached, so a job can be rerun.
# Needs the discovery codes in --cache_dir (run with the same discovery flags as the other runs).
# Run: scripts/uc_saeuron_search.sh
import os
import time

import numpy as np

from attribution import DEFAULT_BLOCK_LIST
from evaluate_sae_features import load_block_codes, load_json, save_json, run_remove_generate, ensure_text_scores, safe
from evaluate_unlearncanvas import (
    parser, CLASSES, THEMES, STYLES, UCModels, discover_entries, saeuron_scores, image_mean_codes,
    attach_latent_means, edit_scale, uc_score_id, load_uc,
)
from experiment_helpers.gpu_details import print_details
from experiment_helpers.init_helpers import repo_api_init

parser.add_argument("--sweep_gammas", nargs="*", type=float, default=[-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0],
                    help="SAeUron's multiplier grid (sweep_cls_distr.py)")
parser.add_argument("--sweep_percentiles", nargs="*", type=float, default=[99.99, 99.995, 99.999],
                    help="SAeUron's feature-percentile grid (sweep_cls_distr.py)")
parser.add_argument("--anchor_prompt_dir", type=str,
                    default="SAeUron/UnlearnCanvas_resources/anchor_prompts/finetune_prompts")
parser.add_argument("--limit_themes", type=int, default=50, help="sweep_cls_distr.py's limit_themes")
parser.add_argument("--search_seed", type=int, default=42)


def search_prompts(args) -> list:
    '''sweep_cls_distr.py's prompts: object's anchor prompt i + " in {style} style." for theme i < limit_themes.'''
    out = []
    for obj in CLASSES:
        with open(os.path.join(args.anchor_prompt_dir, f"sd_prompt_{obj}.txt")) as f:
            lines = f.readlines()  # indexed by theme position, as theirs - not read_lines (it drops blank lines)
        for i, theme in enumerate(THEMES):
            if i >= args.limit_themes:
                break
            if theme == "Seed_Images":
                continue
            p = lines[i].strip()
            p = p[:-1] if p.endswith(".") else p
            out.append({"object": obj, "style": theme, "prompt": f"{p} in {theme.replace('_', ' ')} style."})
    for n, a in enumerate(out):
        a["seed"] = args.search_seed + n
    return out


def results_path(args, block: str) -> str:
    return os.path.join(args.out_dir, "saeuron_search", f"{safe(block)}.json")


def ua_ira(paths: list, prompts: list, concept: str, args):
    '''UA and IRA from the object classifier, or (None, None) if any image isn't scored yet.'''
    preds = []
    for path in paths:
        uc = load_uc(path)
        if uc is None or uc.get("model") != uc_score_id(args):
            return None, None
        preds.append(uc["class_pred"])
    hit = {}
    for a, pred in zip(prompts, preds):
        hit.setdefault(a["object"], []).append(pred == a["object"])
    ua = 1.0 - float(np.mean(hit[concept]))
    ira = float(np.mean([np.mean(h) for obj, h in hit.items() if obj != concept]))
    return ua, ira


def search_block(args, models, entries, targets, block, prompts):
    idx_all, val_all, owner, _, n_dirs = load_block_codes(entries, block)
    means = image_mean_codes(idx_all, val_all, owner, len(entries), n_dirs)
    del idx_all, val_all

    edits = []  # one per object x distinct feature set x gamma
    for concept in targets:
        _, scores = saeuron_scores(args, entries, means, "object", concept)
        sets = {}
        for p in args.sweep_percentiles:
            thr = np.quantile(scores, p / 100.0)  # torch.quantile's default (linear) interpolation
            latents = tuple(int(j) for j in np.argsort(-scores) if scores[j] > thr)
            sets.setdefault(latents, []).append(p)
        for latents, percentiles in sets.items():
            for g in args.sweep_gammas:
                edits.append({"subject": concept, "concept_type": "object", "method": "sam", "block": block,
                              "kind": "saeuron", "latents": list(latents), "gamma": g, "percentiles": percentiles})
        print(f"  {concept} @ {block}: " + "; ".join(f"percentiles {ps} -> {len(ls)} latents {list(ls)[:8]}"
                                                    for ls, ps in sets.items()))
    attach_latent_means(args, entries, edits, [block])  # latent_means (gamma's scale) + thresholds (the mask)

    jobs, paths = [], {}
    for e in edits:
        folder = os.path.join(args.out_dir, "saeuron_search", safe(block), f"t{len(e['latents'])}_g{e['gamma']:g}",
                              safe(e["subject"]))
        part = {"latents": e["latents"], "scale": edit_scale(args, e, e["gamma"]), "thresholds": e["thresholds"]}
        paths[id(e)] = []
        for a in prompts:
            image = os.path.join(folder, f"{a['object']}_{a['style']}.jpg")
            paths[id(e)].append(image)
            jobs.append({"block": block, "parts": {block: part}, "prompt": a["prompt"], "seed": a["seed"],
                         "image": image})
    print(f"{block}: {len(edits)} settings x {len(prompts)} prompts = {len(jobs)} images")
    run_remove_generate(args, models, jobs)
    ensure_text_scores(models.get_uc, "uc", [(j["image"], "image") for j in jobs], args.score_batch_size,
                       model=uc_score_id(args))

    results = load_json(results_path(args, block), {})
    for concept in targets:
        rows = []
        for e in (e for e in edits if e["subject"] == concept):
            ua, ira = ua_ira(paths[id(e)], prompts, concept, args)
            rows.append({"percentiles": e["percentiles"], "tau": len(e["latents"]), "latents": e["latents"],
                         "gamma": e["gamma"], "UA": ua, "IRA": ira,
                         "score": None if ua is None else (ua + ira) / 2})
        results[concept] = rows
        best = max((r for r in rows if r["score"] is not None), key=lambda r: r["score"], default=None)
        if best:
            print(f"  {concept} @ {block}: best tau {best['tau']} gamma {best['gamma']:g} - UA {best['UA']:.3f} "
                  f"IRA {best['IRA']:.3f} | " + " ".join(f"g{r['gamma']:g}:{r['score']:.3f}" for r in rows))
    os.makedirs(os.path.dirname(results_path(args, block)), exist_ok=True)
    save_json(results_path(args, block), results)


def main(args):
    api, accelerator, device = repo_api_init(args)
    args.cache_dir = args.cache_dir or args.out_dir
    args.remove_mode, args.saeuron_mask = "saeuron", True  # SAeUron's edit (see the header)
    args.joint_offsets = None
    block_list = args.block_list or list(DEFAULT_BLOCK_LIST)
    objects = args.object_list or list(CLASSES)
    targets = objects if args.target_objects is None else args.target_objects
    for c in targets:
        assert c in CLASSES, f"'{c}' is not an UnlearnCanvas object"
    entries = discover_entries(args, objects, args.style_list or list(STYLES), style_targets=False)
    missing = [e["sparse"] for e in entries if not os.path.exists(e["sparse"])]
    assert not missing, f"{len(missing)} discovery codes missing (e.g. {missing[0]}) - run evaluate_unlearncanvas.py first"
    prompts = search_prompts(args)
    models = UCModels(args, device)
    for block in block_list:
        search_block(args, models, entries, targets, block, prompts)


if __name__ == "__main__":
    print_details()
    start = time.time()
    args = parser.parse_args()
    print(args)
    main(args)
    print(f"all done! {(time.time() - start) / 3600:.2f} hours")
