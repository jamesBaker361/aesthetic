#!/bin/bash
# The cheap methods vs the searched baseline on 5 eval seeds, for mean +- std over seeds: ESSI (saeuron_masked), OSP,
# adaptive (auto-k on accuracy, auto gamma, all four blocks) and SAeUron searched (one run per block), with the flags of
# scripts/uc_priority.sh except
#   --eval_seeds 188 288 588 688 888   removal on UnlearnCanvas's five answer-set seeds (was 188 only)
#   --base_seeds 5                     insertion on five copies of each base image (copy 0 = the original 20)
#   --per_seed                         tables keyed by seed (removal) / seed_rep (insertion), FID per seed
# Evaluation only - nothing is searched again: auto-k sets, auto gamma, auto strength (searched on copy 0 alone) and
# SAeUron's tau / gamma (scripts/uc_saeuron_search.sh, on its sweep prompts) are all cached in the same out_dirs
# and don't depend on the eval seeds. New work is the answer / base images at the new seeds and their edits.
# Tables -> $OUTPUTS (not the priority runs' outputs_sdxlcls_ucbase, whose seed-188 rows stay); the out_dirs'
# own uc_summary / inject_summary files are rewritten in the per-seed form.
#   1. ESSI @ down.2.1 alone first: it makes the unedited answer images and base copies every other job shares
#   2. then (afterany) ESSI / OSP / adaptive one job per block, and SAeUron one job per block
#   3. then (afterany, per rule) ESSI / OSP / adaptive at all four blocks: everything cached, writes the tables with
#      every block's rows (rows are replaced per subject x method, so a per-block job's table would drop the others)
# Run from the repo root on the cluster: bash scripts/uc_seeds.sh
#   DRY=1      only print the commands
#   TARGET=... share of the injected patches the probe must call positive (default 0.95, as uc_priority.sh)
#   RUNNER=... the slurm wrapper (default runpygpu_chip.sh)

RUNNER=${RUNNER:-runpygpu_chip.sh}
TARGET=${TARGET:-0.95}
SEEDS="188 288 588 688 888"
CLS_DIR=UnlearnCanvas/ckpts/cls_model_sdxl
OUTPUTS=evaluation/outputs_sdxlcls_ucbase_seeds5
ROOT=inject_uc
RUN=evaluation/uc_auto
CACHE=evaluation/uc/cache
BLOCKS="down_blocks.2.attentions.1 mid_block.attentions.0 up_blocks.0.attentions.0 up_blocks.0.attentions.1"

# as scripts/uc_priority.sh, with the seeds
COMMON="--cache_dir $CACHE --object_discover_prompt_file prompt_dir/dream_prompts.txt --discover_seeds 0 \
  --target_styles --object_mask_methods sam --style_mask_methods --eval_scope all --eval_seeds $SEEDS \
  --n_random_controls 0 --inject_value checkpoint_mean --disable_clip --per_seed"
CKPTS="--style_ckpt $CLS_DIR/style_sdxl.pth --class_ckpt $CLS_DIR/class_sdxl.pth --tag sdxlcls"
TEST="$CKPTS --outputs_dir $OUTPUTS --strength_list \
  --base_prompt_file prompt_dir/base_prompts_uc.txt --base_subject_file prompt_dir/base_subjects_uc.txt \
  --base_cache_dir evaluation/uc/cache_ucbase --inject_root $ROOT --base_seeds 5"
AUTO="--inject_auto_strength --inject_strength_target $TARGET --inject_neg_gamma"
SPARSE="--osp_filter 50 --osp_max_k 8 --osp_time_limit 60 --adaptive_k_max 64"   # uc_sparse_probing.sh's

if [ -z "$DRY" ]; then
  for f in $CLS_DIR/style_sdxl.pth $CLS_DIR/class_sdxl.pth; do
    [ -f "$f" ] || { echo "!! $f missing - run train_uc_classifier_sdxl.py first"; exit 1; }
  done
  for rule in saeuron_masked osp adaptive; do
    ls $RUN/${rule}_all_autok_accuracy_autog/features/*.json >/dev/null 2>&1 || \
      { echo "!! no features in $RUN/${rule}_all_autok_accuracy_autog - run scripts/uc_priority.sh first"; exit 1; }
  done
  for block in $BLOCKS; do
    [ -f "$RUN/saeuron_search/saeuron_search/$block.json" ] || \
      { echo "!! $RUN/saeuron_search/saeuron_search/$block.json missing - run scripts/uc_saeuron_search.sh first"; exit 1; }
  done
fi

mkdir -p slurm_chip/seeds
ID=
submit() {  # submit NAME DEPENDENCIES ARGS... (DEPENDENCIES: "id:id", or "" for none) -> $ID
  local name=$1 deps=$2; shift 2
  echo "$name: $*"
  ID=dry_$name
  [ -n "$DRY" ] && return
  ID=$(sbatch --parsable -J seeds ${deps:+--dependency=afterany:$deps} \
    --out=slurm_chip/seeds/$name.out --err=slurm_chip/seeds/$name.err \
    $RUNNER evaluate_unlearncanvas.py "$@")
}

autok() {  # autok RULE NAME DEPENDENCIES BLOCKS...
  local rule=$1 name=$2 deps=$3; shift 3
  extra=; [ "$rule" = saeuron_masked ] || extra=$SPARSE
  submit $name "$deps" --out_dir $RUN/${rule}_all_autok_accuracy_autog $COMMON $TEST $AUTO --rules $rule $extra \
    --auto_k --auto_k_metric accuracy --auto_gamma --remove_mode saeuron --block_list "$@"
}

# ---------------- 1. the shared unedited images
first=${BLOCKS%% *}
autok saeuron_masked saeuron_masked__${first//./_} "" $first
PREP=$ID

# ---------------- 2. per block, then 3. every block's tables (after that rule's block jobs)
for rule in saeuron_masked osp adaptive; do
  ids=$PREP
  for block in $BLOCKS; do
    [ "$rule" = saeuron_masked ] && [ "$block" = "$first" ] && continue
    autok $rule ${rule}__${block//./_} $PREP $block
    ids=$ids:$ID
  done
  autok $rule ${rule}__all $ids $BLOCKS
done
for block in $BLOCKS; do
  name=saeuron_all_searched_toptau_${block//./_}
  submit $name $PREP --out_dir $RUN/$name $COMMON $TEST $AUTO --rules saeuron \
    --saeuron_search_dir $RUN/saeuron_search --saeuron_block best_score --remove_mode saeuron --saeuron_mask \
    --block_list $block
done
