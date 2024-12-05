#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=8
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=brainbert
#SBATCH --time=169:59:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=test

export WANDB_CACHE_DIR=$DATA/wandb_cache
export HF_HOME=$DATA/hf_cache

srun python $@