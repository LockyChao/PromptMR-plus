#!/bin/bash
#SBATCH --job-name=CMRdirectinference
#SBATCH --output=logs/%j.log
#SBATCH --error=errors/%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:l40s:2
#SBATCH --mail-user=chaowei.wu@cshs.org
#SBATCH --mail-type=END,FAIL

source activate cmr

CMRROOT=/common/lidxxlab/cmrchallenge/code/chaowei
SAVE_DIR=/common/lidxxlab/cmrchallenge/data/CMR2025

echo "Detailed GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

wandb login b893a6d433af71e3cdb71002caa7dc90d57d4587

# Run training script with the correct LightningCLI format
python main.py predict \
       -c $CMRROOT/configs/base.yaml \
       -c $CMRROOT/configs/inference/pmr-plus/cmr25-cardiac-training.yaml \
       -c $CMRROOT/configs/model/pmr-plus.yaml \
       --trainer.devices=auto \
       --model.init_args.pretrain=True \
       --model.init_args.pretrain_weights_path=$CMRROOT/experiments/cmr25/promptmr-plus/CMR2025/deep_recon/uec2kxvx/checkpoints/last.ckpt