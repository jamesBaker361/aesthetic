# CASL's own way of choosing CASL-Steer's alpha (arXiv:2601.15441, App. 7.10, Table 6): k is fixed at 1 (the top-1
# latent of the CASL map, Table 1) and alpha is picked PER CONCEPT BY VISUAL INSPECTION on a validation set - "a
# balance between perceptible attribute change and preservation of image realism"; they do not use EPR for it.
# The edit is at the U-Net bottleneck only (App. 7.5) - here --block_list, default mid_block.attentions.0.
#
# This renders the grid to pick from, per object:
#   prompts   --casl_val_prompts UnlearnCanvas anchor prompts of the object from line 50 on (SAeUron's search uses
#             lines < 50; CASL trains on --object_discover_prompt_file; the benchmark uses its own template), each
#             " in {style} style." with styles spread over the 50, seed 1000 + i
#   columns   unedited, then CASL-Steer at every --casl_alpha_grid value (Table 6's range, 16 ... 160)
#   captions  the UnlearnCanvas object classifier's prediction (scored and cached here)
# -> {out_dir}/casl_alpha_preview/{casl key hash}/{block}/{object}/a{alpha}/{i}.jpg + manifest.json, which
# casl_alpha_pick.ipynb shows; the alphas picked there go to a JSON for evaluate_unlearncanvas.py --casl_alpha_file.
# Trains the CASL maps first if this setting's aren't cached yet (needs runpygpu_chip_40.sh then).
# Run: scripts/uc_compare_methods.sh (step 3), or with the same --casl_* flags as the CASL runs.
import hashlib
import os
import time

from evaluate_sae_features import save_json, ensure_text_scores, safe
from evaluate_unlearncanvas import (
    parser, CLASSES, STYLES, UCModels, discover_entries, load_casl, casl_key, casl_npz_path, run_casl,
    run_steer_generate, uc_score_id,
)
from experiment_helpers.gpu_details import print_details
from experiment_helpers.init_helpers import repo_api_init

parser.add_argument("--anchor_prompt_dir", type=str,
                    default="SAeUron/UnlearnCanvas_resources/anchor_prompts/finetune_prompts")
parser.add_argument("--anchor_offset", type=int, default=50, help="first anchor prompt line used (after SAeUron's 50)")

BOTTLENECK = "mid_block.attentions.0"


def val_prompts(args, obj: str) -> list:
    with open(os.path.join(args.anchor_prompt_dir, f"sd_prompt_{obj}.txt")) as f:
        lines = [l.strip().rstrip(".") for l in f.readlines()[args.anchor_offset:] if l.strip()]
    n = min(args.casl_val_prompts, len(lines))
    styles = [STYLES[(i * len(STYLES)) // n] for i in range(n)]  # spread over the 50 styles
    return [{"prompt": f"{lines[i]} in {styles[i].replace('_', ' ')} style.", "style": styles[i], "seed": 1000 + i}
            for i in range(n)]


def main(args):
    api, accelerator, device = repo_api_init(args)
    args.cache_dir = args.cache_dir or args.out_dir
    args.joint_offsets = None
    block = (args.block_list or [BOTTLENECK])[0]
    objects = args.object_list or list(CLASSES)
    targets = objects if args.target_objects is None else args.target_objects
    models = UCModels(args, device)

    missing = [c for c in targets if block not in load_casl(args, c)]
    if missing:  # train this setting's maps at the block (shared cache, so the steer run reuses them)
        print(f"CASL maps missing for {missing} @ {block} - training them")
        entries = discover_entries(args, objects, args.style_list or list(STYLES), style_targets=False)
        run_casl(args, models, entries, [("object", c) for c in missing], [block])

    key_hash = hashlib.sha1(casl_key(args).encode()).hexdigest()[:12]
    root = os.path.join(args.out_dir, "casl_alpha_preview", key_hash)
    alphas = [0.0] + [float(a) for a in args.casl_alpha_grid]  # 0 = the unedited image
    jobs, manifest = [], {"casl_key": casl_key(args), "block": block, "alphas": alphas, "objects": {}}
    for concept in targets:
        ranking = load_casl(args, concept).get(block)
        if not ranking or not ranking["order"]:
            print(f"  ! no CASL ranking for '{concept}' @ {block} - skipped")
            continue
        latents = ranking["order"][:args.casl_steer_k]  # k = 1 by default, as the paper
        prompts = val_prompts(args, concept)
        images = {}
        for a in alphas:
            paths = []
            for i, p in enumerate(prompts):
                path = os.path.join(root, safe(block), safe(concept), f"a{a:g}", f"{i}.jpg")
                parts = {} if a == 0 else {block: {"steer": casl_npz_path(args, concept, block), "latents": latents,
                                                   "alpha": a}}
                jobs.append({"block": block, "parts": parts, "prompt": p["prompt"], "seed": p["seed"], "image": path})
                paths.append(path)
            images[f"{a:g}"] = paths
        manifest["objects"][concept] = {"latents": latents, "prompts": prompts, "images": images}
    run_steer_generate(args, models, jobs)
    ensure_text_scores(models.get_uc, "uc", [(j["image"], "image") for j in jobs], args.score_batch_size,
                       model=uc_score_id(args))
    os.makedirs(root, exist_ok=True)
    save_json(os.path.join(root, "manifest.json"), manifest)
    print(f"{len(jobs)} preview images; pick alphas in casl_alpha_pick.ipynb from {root}/manifest.json")


if __name__ == "__main__":
    print_details()
    start = time.time()
    args = parser.parse_args()
    print(args)
    main(args)
    print(f"all done! {(time.time() - start) / 3600:.2f} hours")
