#!/bin/bash
# CASL on CelebA-HQ with an unconditional SDXL (casl_faces.py; its header has the method), as a two-stage sweep.
#   STAGE=1 (default)  24 configs = pipeline {turbo_noise, turbo_invert, base_invert} x block {down.2.1, mid.0} x
#                      Adam lr {1e-3, 1e-2} x lambda_clip : lambda_recon {3:1 (the paper's faces), 10:1}, each on
#                      Smiling + Blond_Hair; W snapshots at 1,000 / 3,000 / 5,000 steps, all evaluated; k = 1, the
#                      paper's alpha grid (Fig. 4). One prepare job per pipeline fills its trajectory cache first.
#   STAGE=2            each "{run} {snapshot}" of $ROOT/best_configs.txt (python casl_faces_summary.py) on all 5
#                      attributes (Table 1's), k = 1, 2, 4 at that snapshot - the same out_dirs, so stage 1's maps
#                      and rows are reused
#   pipelines          turbo_noise   sdxl-turbo, forward noise to t = 749, 3 steps (749, 499, 250), edit t >= 400
#                      turbo_invert  sdxl-turbo, DDIM inversion to t = 999, 4 steps (999, 749, 500, 250), edit t >= 500
#                      base_invert   SDXL base, DDIM inversion to t = 999, 50 steps, edit t >= 500 (the paper's
#                                    App. 7.4, with the Turbo SAEs off their training model)
# Before: python prepare_celebahq.py (data/celebahq), then sbatch runpygpu_chip.sh train_celeba_classifier.py.
# Results -> evaluation/casl_faces/{run}/results.csv; python casl_faces_summary.py tabulates them.
# Run from the repo root on the cluster: bash scripts/casl_faces_sweep.sh   (DRY=1 to only print the commands)
#   RUNNER=...  the slurm wrapper (default runpygpu_chip_40.sh)

RUNNER=${RUNNER:-runpygpu_chip_40.sh}
STAGE=${STAGE:-1}
ROOT=evaluation/casl_faces
CACHE=$ROOT/cache
STAGE1_ATTRS="Smiling Blond_Hair"
ALL_ATTRS="Smiling Young Beards Big_Nose Blond_Hair"

declare -A PIPES=(
  [turbo_noise]="--model turbo --start noise --t_start 749 --n_steps 3 --t_edit 400"
  [turbo_invert]="--model turbo --start invert --t_start 999 --n_steps 4 --t_edit 500"
  [base_invert]="--model base --start invert --t_start 999 --n_steps 50 --t_edit 500"
)
declare -A BLOCKS=([down]=down_blocks.2.attentions.1 [mid]=mid_block.attentions.0)
declare -A PIPE_OF=() ARGS_OF=()   # run name -> its pipeline, its casl_faces.py settings
NAMES=()
for pipe in turbo_noise turbo_invert base_invert; do for blk in down mid; do for lr in 1e-3 1e-2; do
  for lc in 3 10; do
    name=${pipe}_${blk}_lr${lr}_lc${lc}
    NAMES+=($name)
    PIPE_OF[$name]=$pipe
    ARGS_OF[$name]="${PIPES[$pipe]} --block ${BLOCKS[$blk]} --optimizer adam --lr $lr --lambda_clip $lc --lambda_recon 1"
  done
done; done; done

if [ -z "$DRY" ]; then
  [ -f data/celebahq/attrs.csv ] || { echo "!! data/celebahq/attrs.csv missing - run prepare_celebahq.py first"; exit 1; }
  [ -f data/celebahq/attr_r50.pth ] || { echo "!! data/celebahq/attr_r50.pth missing - run train_celeba_classifier.py first"; exit 1; }
fi

mkdir -p slurm_chip/casl_faces
ID=
submit() {  # submit NAME DEPENDENCIES SCRIPT ARGS... (DEPENDENCIES: "id:id", or "" for none) -> $ID
  local name=$1 deps=$2; shift 2
  echo "$name: $*"
  ID=dry_$name
  [ -n "$DRY" ] && return
  ID=$(sbatch --parsable -J casl_faces ${deps:+--dependency=afterok:$deps} \
    --out=slurm_chip/casl_faces/$name.out --err=slurm_chip/casl_faces/$name.err $RUNNER "$@")
}

if [ "$STAGE" = 1 ]; then
  declare -A PREP=()
  for pipe in "${!PIPES[@]}"; do
    submit prepare_$pipe "" casl_faces.py --stage prepare --cache_dir $CACHE ${PIPES[$pipe]}
    PREP[$pipe]=$ID
  done
  for name in "${NAMES[@]}"; do
    submit $name ${PREP[${PIPE_OF[$name]}]} casl_faces.py --out_dir $ROOT/$name --cache_dir $CACHE \
      ${ARGS_OF[$name]} --attributes $STAGE1_ATTRS --k_list 1
  done
elif [ "$STAGE" = 2 ]; then
  [ -f $ROOT/best_configs.txt ] || { echo "!! $ROOT/best_configs.txt missing - run casl_faces_summary.py first"; exit 1; }
  declare -A STEPS=()   # run -> its listed snapshots: one job per run, as its jobs would share its results.csv
  RUNS=()
  while read -r name step; do
    [ -n "$name" ] || continue
    [ -n "${ARGS_OF[$name]}" ] || { echo "!! $name is not a stage-1 run"; exit 1; }
    [ -n "${STEPS[$name]}" ] || RUNS+=($name)
    STEPS[$name]="${STEPS[$name]} $step"
  done < $ROOT/best_configs.txt
  for name in "${RUNS[@]}"; do
    submit ${name}_stage2 "" casl_faces.py --out_dir $ROOT/$name --cache_dir $CACHE ${ARGS_OF[$name]} \
      --attributes $ALL_ATTRS --k_list 1 2 4 --eval_steps ${STEPS[$name]}
  done
fi
