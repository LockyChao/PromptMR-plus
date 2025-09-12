#!/bin/bash
#SBATCH --job-name=CMRscratch
#SBATCH --output=logs/%j.log
#SBATCH --error=errors/%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:4


source activate cmr_cs


CMRROOT=$PWD
SAVE_DIR=$PWD/exp

echo "Detailed GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


# wandb agent your-username/promptmr-sweep/abc123def
wandb agent $1/promptmr-sweep/$2