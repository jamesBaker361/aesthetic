#!/bin/bash
# CASL-Steer (Adam lr 1e-2, 5k steps) with the strengths chosen as the matched ESSI / SAeUron runs choose theirs, at
# down_blocks.2.attentions.1 and mid_block.attentions.0 (the blocks CASL was trained at). gamma = removal strength,
# alpha = insertion strength (gamma = -alpha on CASL's scale). Settings as scripts/uc_priority.sh (no CLIP, insertion
# on the UnlearnCanvas subjects).
#   auto gamma      the priority runs' ..._autoa out_dirs again (their gamma, answers and removal rows are cached), now
#                   with insertion: at alpha = -auto gamma (k_metric top1) and at the auto strength (--inject_auto_strength,
#                   alpha in [0, --casl_alpha_max]; top1_sauto{TARGET}) -> evaluation/outputs_sdxlcls_ucbase
#   searched gamma  SAeUron's search for alpha (saeuron_search.py --feature_rule casl_steer: every alpha of $ALPHAS on its
#                   sweep prompts, best (UA + IRA) / 2, pretrained classifier as SAeUron's), one job per block, then
#                   removal at gamma = -that alpha and insertion at alpha (--casl_alpha_file) -> ..._asearch
# Run from the repo root on the cluster: bash scripts/uc_casl_matched.sh
#   DRY=1      only print the commands
#   TARGET=... share of the injected patches the probe must call positive (default 0.95)
#   ALPHAS=... the search grid (default the priority sweep's "1 32 64 96 128")
#   RUNNER=... the slurm wrapper (default runpygpu_chip.sh)
#   ONLY=autog only the auto-gamma (_autoa) jobs, again (e.g. after a change to the gamma search)

RUNNER=${RUNNER:-runpygpu_chip.sh}
TARGET=${TARGET:-0.95}
ALPHAS=${ALPHAS:-"1 32 64 96 128"}
STEPS=5000
CLS_DIR=UnlearnCanvas/ckpts/cls_model_sdxl
OUTPUTS=evaluation/outputs_sdxlcls_ucbase
ROOT=inject_uc
RUN=evaluation/uc_auto
CACHE=evaluation/uc/cache
BLOCKS="down_blocks.2.attentions.1 mid_block.attentions.0"
SEARCH=$RUN/casl_steer_paper_adam_lr1e-2_s${STEPS}_alpha_search

# as scripts/uc_priority.sh
DISCOVER="--cache_dir $CACHE --object_discover_prompt_file prompt_dir/dream_prompts.txt --discover_seeds 0"
COMMON="$DISCOVER --target_styles --object_mask_methods sam --style_mask_methods --eval_scope all --eval_seeds 188 \
  --n_random_controls 0 --inject_value checkpoint_mean --disable_clip"
CKPTS="--style_ckpt $CLS_DIR/style_sdxl.pth --class_ckpt $CLS_DIR/class_sdxl.pth --tag sdxlcls"
TEST="$CKPTS --outputs_dir $OUTPUTS --strength_list \
  --base_prompt_file prompt_dir/base_prompts_uc.txt --base_subject_file prompt_dir/base_subjects_uc.txt \
  --base_cache_dir evaluation/uc/cache_ucbase --inject_root $ROOT"
CASL="--casl_steer_k 1 --casl_direction paper --casl_optimizer adam --casl_lr 1e-2 --casl_steps $STEPS"

if [ -z "$DRY" ]; then
  for f in $CLS_DIR/style_sdxl.pth $CLS_DIR/class_sdxl.pth; do
    [ -f "$f" ] || { echo "!! $f missing - run train_uc_classifier_sdxl.py first"; exit 1; }
  done
  [ -d evaluation/uc/cache_ucbase ] || { echo "!! evaluation/uc/cache_ucbase missing - no base images yet"; exit 1; }
fi

mkdir -p slurm_chip/casl_matched
ID=
submit() {  # submit NAME DEPENDENCIES SCRIPT ARGS... (DEPENDENCIES: "id:id", or "" for none) -> $ID
  local name=$1 deps=$2; shift 2
  echo "$name: $*"
  ID=dry_$name
  [ -n "$DRY" ] && return
  ID=$(sbatch --parsable -J casl_matched ${deps:+--dependency=afterok:$deps} \
    --out=slurm_chip/casl_matched/$name.out --err=slurm_chip/casl_matched/$name.err $RUNNER "$@")
}

for block in $BLOCKS; do
  name=casl_steer_all_paper_adam_lr1e-2_s$STEPS
  [ "$block" = mid_block.attentions.0 ] || name=${name}_${block//./_}

  # auto gamma: removal cached; insertion at alpha = -auto gamma and at the auto strength
  submit ${name}_autoa "" evaluate_unlearncanvas.py --out_dir $RUN/${name}_autoa $COMMON $TEST --rules casl_steer $CASL \
    --block_list $block --auto_gamma --casl_alpha_max 192 --inject_auto_strength --inject_strength_target $TARGET
  [ "$ONLY" = autog ] && continue

  # searched gamma: SAeUron's search (no --style_ckpt / --class_ckpt: the pretrained classifiers, as uc_saeuron_search.sh)
  submit casl_s${STEPS}_search_${block//./_} "" saeuron_search.py --out_dir $SEARCH $DISCOVER --block_list $block \
    --feature_rule casl_steer --rules casl_steer $CASL --sweep_casl_alphas $ALPHAS
  submit ${name}_asearch "$ID" evaluate_unlearncanvas.py --out_dir $RUN/${name}_asearch $COMMON $TEST \
    --rules casl_steer $CASL --block_list $block --casl_alpha_file $SEARCH/saeuron_search/${block}_casl_alpha.json
done
