#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=1000G
#SBATCH --qos=system
#SBATCH --clusters=htc
#SBATCH --job-name=megalodon
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=test

export WANDB_CACHE_DIR=$DATA/wandb_cache

srun python $@
