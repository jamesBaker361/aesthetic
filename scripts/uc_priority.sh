#!/bin/bash
# The priority runs, in two block batches: first down_blocks.2.attentions.1 + mid_block.attentions.0, then (after every
# first-batch job has ended, afterany) up_blocks.0.attentions.0 + up_blocks.0.attentions.1. No CLIP (--disable_clip),
# no VQA (off by default). Insertion on the UnlearnCanvas subjects as scripts/uc_inject_uc_subjects.sh's test jobs, but
# only the searched strengths: --strength_list is empty, so no fixed strength-10 row, and no validation / grid jobs.
#   ESSI (saeuron_masked), OSP, adaptive   auto-k on accuracy, auto gamma (removal); insertion at auto strength
#                                          (k_metric accuracy_sauto{TARGET}) and at minus the auto gamma
#                                          (--inject_neg_gamma; accuracy_snegg). Their out_dirs hold all four blocks, so the second
#                                          batch's job runs all four: the first two are cached, and the merged tables
#                                          (replaced per subject x method) keep every block's rows.
#   SAeUron searched, one run per block    its searched gamma (removal); auto strength and minus its gamma (insertion,
#                                          --inject_neg_gamma; k_metric ..._snegg). No best-score-block run.
#   CASL-Steer (Adam lr 1e-2, 10k / 5k)    first batch only (its two blocks). Per alpha in {1, 32, 64, 96, 128}:
#                                          insertion at alpha, removal at gamma = -alpha. Plus removal only with gamma
#                                          searched (--auto_gamma, the dense probe as ESSI's, |gamma| <= 192; own
#                                          ..._autoa out_dirs), no insertion.
# The base images (evaluation/uc/cache_ucbase) must already exist - scripts/uc_inject_uc_subjects.sh's first test job
# made them.
# Run from the repo root on the cluster: bash scripts/uc_priority.sh
#   DRY=1      only print the commands
#   TARGET=... share of the injected patches the probe must call positive (default 0.95)
#   RUNNER=... the slurm wrapper (default runpygpu_chip.sh)
#   ONLY=autog only the auto-gamma jobs, again (e.g. after a change to the gamma search): ESSI / OSP / adaptive at all
#              four blocks in one job each, and CASL 10k's _autoa (5k's: ONLY=autog bash scripts/uc_casl_matched.sh)

RUNNER=${RUNNER:-runpygpu_chip.sh}
TARGET=${TARGET:-0.95}
CLS_DIR=UnlearnCanvas/ckpts/cls_model_sdxl
OUTPUTS=evaluation/outputs_sdxlcls_ucbase
ROOT=inject_uc
RUN=evaluation/uc_auto
CACHE=evaluation/uc/cache
FIRST="down_blocks.2.attentions.1 mid_block.attentions.0"
SECOND="up_blocks.0.attentions.0 up_blocks.0.attentions.1"

# as scripts/uc_inject_uc_subjects.sh, plus --disable_clip
COMMON="--cache_dir $CACHE --object_discover_prompt_file prompt_dir/dream_prompts.txt --discover_seeds 0 \
  --target_styles --object_mask_methods sam --style_mask_methods --eval_scope all --eval_seeds 188 \
  --n_random_controls 0 --inject_value checkpoint_mean --disable_clip"
CKPTS="--style_ckpt $CLS_DIR/style_sdxl.pth --class_ckpt $CLS_DIR/class_sdxl.pth --tag sdxlcls"
TEST="$CKPTS --outputs_dir $OUTPUTS --strength_list \
  --base_prompt_file prompt_dir/base_prompts_uc.txt --base_subject_file prompt_dir/base_subjects_uc.txt \
  --base_cache_dir evaluation/uc/cache_ucbase --inject_root $ROOT"
AUTO="--inject_auto_strength --inject_strength_target $TARGET"
SPARSE="--osp_filter 50 --osp_max_k 8 --osp_time_limit 60 --adaptive_k_max 64"   # uc_sparse_probing.sh's

if [ -z "$DRY" ]; then
  for f in $CLS_DIR/style_sdxl.pth $CLS_DIR/class_sdxl.pth; do
    [ -f "$f" ] || { echo "!! $f missing - run train_uc_classifier_sdxl.py first"; exit 1; }
  done
  [ -d evaluation/uc/cache_ucbase ] || { echo "!! evaluation/uc/cache_ucbase missing - no base images yet"; exit 1; }
fi

mkdir -p slurm_chip/priority
ID=
BATCH1=   # every first-batch job id: the second batch waits for all of them
submit() {  # submit NAME DEPENDENCIES ARGS... (DEPENDENCIES: "id:id", or "" for none) -> $ID
  local name=$1 deps=$2; shift 2
  echo "$name: $*"
  ID=dry_$name
  [ -n "$DRY" ] && return
  ID=$(sbatch --parsable -J priority ${deps:+--dependency=afterany:$deps} \
    --out=slurm_chip/priority/$name.out --err=slurm_chip/priority/$name.err \
    $RUNNER evaluate_unlearncanvas.py "$@")
}

autok() {  # autok NAME DEPENDENCIES BLOCKS...: ESSI, OSP and adaptive at these blocks
  local tag=$1 deps=$2; shift 2
  for rule in saeuron_masked osp adaptive; do
    extra=; [ "$rule" = saeuron_masked ] || extra=$SPARSE
    submit ${rule}_all_autok_accuracy_autog__$tag "$deps" --out_dir $RUN/${rule}_all_autok_accuracy_autog \
      $COMMON $TEST $AUTO --inject_neg_gamma --rules $rule $extra --auto_k --auto_k_metric accuracy --auto_gamma \
      --remove_mode saeuron --block_list "$@"
    [ "$tag" = first ] && BATCH1=${BATCH1:+$BATCH1:}$ID
  done
}

saeuron() {  # saeuron DEPENDENCIES BLOCKS...: the searched SAeUron, one run per block
  local deps=$1; shift
  for block in "$@"; do
    name=saeuron_all_searched_toptau_${block//./_}
    submit $name "$deps" --out_dir $RUN/$name $COMMON $TEST $AUTO --inject_neg_gamma --rules saeuron \
      --saeuron_search_dir $RUN/saeuron_search --saeuron_block best_score --remove_mode saeuron --saeuron_mask \
      --block_list $block
    [ -z "$deps" ] && BATCH1=${BATCH1:+$BATCH1:}$ID
  done
}

if [ "$ONLY" = autog ]; then
  autok all "" $FIRST $SECOND
  for block in $FIRST; do
    name=casl_steer_all_paper_adam_lr1e-2_s10000
    [ "$block" = mid_block.attentions.0 ] || name=${name}_${block//./_}
    submit ${name}_autoa "" --out_dir $RUN/${name}_autoa $COMMON $CKPTS --outputs_dir evaluation/outputs_sdxlcls \
      --strength_list 10 --disable_inject --rules casl_steer --casl_steer_k 1 --casl_direction paper \
      --casl_optimizer adam --casl_lr 1e-2 --casl_steps 10000 --block_list $block --auto_gamma --casl_alpha_max 192
  done
  exit 0
fi

# ---------------- first batch: down.2.1 + mid.0
autok first "" $FIRST
saeuron "" $FIRST
for steps in 10000 5000; do
  for block in $FIRST; do
    name=casl_steer_all_paper_adam_lr1e-2_s$steps
    [ "$block" = mid_block.attentions.0 ] || name=${name}_${block//./_}
    CASL="--rules casl_steer --casl_steer_k 1 --casl_direction paper --casl_optimizer adam --casl_lr 1e-2 \
      --casl_steps $steps --block_list $block"
    # one setting per alpha: insertion at alpha, removal at gamma = -alpha (one job: the settings share an out_dir)
    submit $name "" --out_dir $RUN/$name $COMMON $TEST $CASL --casl_alpha 1 32 64 96 128
    BATCH1=${BATCH1:+$BATCH1:}$ID
    # gamma searched for removal (--auto_gamma, the dense probe as ESSI's), once per object x block; no insertion
    submit ${name}_autoa "" --out_dir $RUN/${name}_autoa $COMMON $CKPTS --outputs_dir evaluation/outputs_sdxlcls \
      --strength_list 10 --disable_inject $CASL --auto_gamma --casl_alpha_max 192
    BATCH1=${BATCH1:+$BATCH1:}$ID
  done
done

# ---------------- second batch: up.0.0 + up.0.1, after all of the first
autok second "$BATCH1" $FIRST $SECOND
saeuron "$BATCH1" $SECOND
