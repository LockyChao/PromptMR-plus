#!/bin/bash
#SBATCH --job-name=CMRscratch
#SBATCH --output=logs/%j.log
#SBATCH --error=errors/%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:l40s:1

source activate cmrxrecon

CMRROOT=/common/lidxxlab/Yifan/PromptMR-plus
SAVE_DIR=/common/lidxxlab/Yifan/PromptMR-plus/experiments/cmr25_dds

echo "Detailed GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

wandb login 0787b9ad523f06b9679a87cba284976707e03c86

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training script with the correct LightningCLI format
python $CMRROOT/main.py fit \
       -c $CMRROOT/configs/base.yaml \
       -c $CMRROOT/configs/train/pmr-plus/cmr25-cardiac-dds.yaml \
       -c $CMRROOT/configs/model/pmr-dds.yaml \
       --trainer.devices=auto \
       --trainer.logger.init_args.save_dir=$SAVE_DIR/promptmr-dds/CMR2025