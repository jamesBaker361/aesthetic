#!/bin/bash

#SBATCH --partition=main       # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1               # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)

#SBATCH --mem=128000                # Real memory (RAM) required (MB)

#SBATCH --time=3-00:00:00           # Total run time limit (D-HH:MM:SS)

#SBATCH --output=slurm/generic/%j.out  # STDOUT output file

#SBATCH --error=slurm/generic/%j.err   # STDERR output file (optional)

#SBATCH --exclude=halk[0001-0106]

day=$(date +'%m/%d/%Y %R')
echo "main" ${day} $SLURM_JOBID "node_list" $SLURM_NODELIST $@  "\n" >> jobs.txt
module purge
export MODULEPATH=$MODULEPATH:/projects/community/modulefiles
module load intel/17.0.4
#module load cudnn/7.0.3
module load gcc/10.3.0-pgarias
module load boost/1.71.0-gc563
module load openjdk/1.8.0_362
eval "$(conda shell.bash hook)"
conda activate deepdoom
export CXX=/projects/community/gcc/10.3.0/pgarias/bin/g++  
export CC=/projects/community/gcc/10.3.0/pgarias/bin/gcc 
export TRANSFORMERS_CACHE="/scratch/jlb638/trans_cache"
export HF_HOME="/scratch/jlb638/trans_cache"
export HF_HUB_CACHE="/scratch/jlb638/trans_cache"
export TORCH_CACHE="/scratch/jlb638/torch_hub_cache"
export WANDB_DIR="/scratch/jlb638/wandb"
export WANDB_CACHE_DIR="/scratch/jlb638/wandb_cache"
export HPS_ROOT="/scratch/jlb638/hps-cache"
export IMAGE_REWARD_PATH="/scratch/jlb638/reward-blob"
export IMAGE_REWARD_CONFIG="/scratch/jlb638/ImageReward/med_config.json"
export EMBEDDING_DIR="/scratch/jlb638/embedding_src"
export TORCH_LOCAL_DIR="/scratch/jlb638/local_torch"
export KAGGLEHUB_CACHE="/scratch/jlb638/kaggle_cache"
export KAGGLE_CONFIG_DIR="/scratch/jlb638/kaggle_config"
export TMPDIR="/umbc/rs/pi_donengel/users/jbaker15/tmp"
srun python3 $@
conda deactivate