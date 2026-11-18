#!/bin/bash
# Matched tuning: ESSI / OSP / adaptive vs SAeUron with the removal gamma chosen the same way for both, so the
# comparison isolates the feature choice (auto-k set vs SAeUron's searched top-tau). Settings as scripts/uc_priority.sh
# (no CLIP, insertion on the UnlearnCanvas subjects); all four blocks. Insertion only at minus the new gamma
# (--inject_neg_gamma): the auto strength doesn't depend on gamma, so the priority runs' auto-strength rows hold.
#   searched gamma   ESSI / OSP / adaptive with gamma from the same image search SAeUron's was picked with: their
#                    auto-k set at every gamma of SAeUron's grid on its sweep prompts, best (UA + IRA) / 2. Those
#                    images exist (scripts/uc_gamma_search.sh), but were scored with the sdxl classifier, while
#                    SAeUron's search used the pretrained one (scripts/uc_saeuron_search.sh): step 1 rescores them
#                    with the pretrained classifier (no generation - the block folders are symlinked into a new
#                    out_dir so the sdxl-scored jsons stay), step 2 evaluates at each object x block's best gamma
#                    -> $RUN/{rule}_all_autok_accuracy_gsearchpre
#   auto gamma       SAeUron with its searched tau (per block) but gamma from the probe bisection, as the auto-k
#                    rules' priority runs -> $RUN/saeuron_all_searched_toptau_autog_{block}
# The counterparts are already in uc_priority.sh: ESSI / OSP / adaptive auto gamma, SAeUron searched gamma.
# Run from the repo root on the cluster: bash scripts/uc_matched.sh
#   DRY=1      only print the commands
#   RUNNER=... the slurm wrapper (default runpygpu_chip.sh)
#   ONLY=autog only the SAeUron auto-gamma runs, again (e.g. after a change to the gamma search)

RUNNER=${RUNNER:-runpygpu_chip.sh}
GAMMAS="-1 -5 -10 -15 -20 -25 -30"   # SAeUron's grid, as scripts/uc_gamma_search.sh
CLS_DIR=UnlearnCanvas/ckpts/cls_model_sdxl
OUTPUTS=evaluation/outputs_sdxlcls_ucbase
ROOT=inject_uc
RUN=evaluation/uc_auto
CACHE=evaluation/uc/cache
BLOCKS="down_blocks.2.attentions.1 mid_block.attentions.0 up_blocks.0.attentions.0 up_blocks.0.attentions.1"

# as scripts/uc_priority.sh
COMMON="--cache_dir $CACHE --object_discover_prompt_file prompt_dir/dream_prompts.txt --discover_seeds 0 \
  --target_styles --object_mask_methods sam --style_mask_methods --eval_scope all --eval_seeds 188 \
  --n_random_controls 0 --inject_value checkpoint_mean --disable_clip"
CKPTS="--style_ckpt $CLS_DIR/style_sdxl.pth --class_ckpt $CLS_DIR/class_sdxl.pth --tag sdxlcls"
TEST="$CKPTS --outputs_dir $OUTPUTS --strength_list \
  --base_prompt_file prompt_dir/base_prompts_uc.txt --base_subject_file prompt_dir/base_subjects_uc.txt \
  --base_cache_dir evaluation/uc/cache_ucbase --inject_root $ROOT"
AUTO="--inject_neg_gamma"
SPARSE="--osp_filter 50 --osp_max_k 8 --osp_time_limit 60 --adaptive_k_max 64"   # uc_sparse_probing.sh's

if [ -z "$DRY" ]; then
  for f in $CLS_DIR/style_sdxl.pth $CLS_DIR/class_sdxl.pth; do
    [ -f "$f" ] || { echo "!! $f missing - run train_uc_classifier_sdxl.py first"; exit 1; }
  done
  [ -d evaluation/uc/cache_ucbase ] || { echo "!! evaluation/uc/cache_ucbase missing - no base images yet"; exit 1; }
fi

mkdir -p slurm_chip/matched
ID=
submit() {  # submit NAME DEPENDENCIES SCRIPT ARGS... (DEPENDENCIES: "id:id", or "" for none) -> $ID
  local name=$1 deps=$2; shift 2
  echo "$name: $*"
  ID=dry_$name
  [ -n "$DRY" ] && return
  ID=$(sbatch --parsable -J matched ${deps:+--dependency=afterok:$deps} \
    --out=slurm_chip/matched/$name.out --err=slurm_chip/matched/$name.err $RUNNER "$@")
}

# ---------------- ESSI / OSP / adaptive, searched gamma (pretrained-classifier search, as SAeUron's)
for rule in $([ "$ONLY" = autog ] || echo saeuron_masked osp adaptive); do
  src=$RUN/${rule}_all_autok_accuracy_autog   # the auto-k sets
  old=$RUN/${rule}_gamma_search               # the sdxl-scored search (its images)
  search=$RUN/${rule}_gamma_search_clspre
  name=${rule}_all_autok_accuracy_gsearchpre
  extra=; [ "$rule" = saeuron_masked ] || extra=$SPARSE
  if [ -z "$DRY" ]; then
    ls $src/features/*.json >/dev/null 2>&1 || { echo "!! no features in $src - skipped $rule"; continue; }
    mkdir -p $RUN/$name/features $search/saeuron_search && cp -n $src/features/*.json $RUN/$name/features/
  fi
  ids=
  for block in $BLOCKS; do
    if [ -z "$DRY" ]; then
      [ -d $old/saeuron_search/$block ] || { echo "!! $old/saeuron_search/$block missing - run uc_gamma_search.sh"; exit 1; }
      ln -sfn "$(cd $old/saeuron_search/$block && pwd)" $search/saeuron_search/$block
    fi
    # no --style_ckpt / --class_ckpt: the pretrained classifiers, as uc_saeuron_search.sh
    submit ${rule}_rescore_${block//./_} "" saeuron_search.py --out_dir $search --cache_dir $CACHE --block_list $block \
      --object_discover_prompt_file prompt_dir/dream_prompts.txt \
      --feature_rule auto_k --auto_k_dir $src --auto_k_rule $rule --auto_k_metric accuracy --sweep_gammas $GAMMAS
    ids=${ids:+$ids:}$ID
  done
  submit $name "$ids" evaluate_unlearncanvas.py --out_dir $RUN/$name $COMMON $TEST $AUTO --rules $rule $extra \
    --auto_k --auto_k_metric accuracy --remove_mode saeuron --gamma_search_dir $search --block_list $BLOCKS
done

# ---------------- SAeUron, auto gamma (its searched tau at each block)
for block in $BLOCKS; do
  name=saeuron_all_searched_toptau_autog_${block//./_}
  submit $name "" evaluate_unlearncanvas.py --out_dir $RUN/$name $COMMON $TEST $AUTO --rules saeuron \
    --saeuron_search_dir $RUN/saeuron_search --saeuron_block best_score --remove_mode saeuron --saeuron_mask \
    --auto_gamma --block_list $block
done
