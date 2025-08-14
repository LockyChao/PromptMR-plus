#!/bin/bash
#SBATCH --job-name=CMRscratch
#SBATCH --output=logs/%j.log
#SBATCH --error=errors/%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --mail-user=junzhou.chen@cshs.org
#SBATCH --mail-type=END,FAIL

source activate cmr_cs


CMRROOT=$PWD
SAVE_DIR=$PWD/exp

echo "Detailed GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"



# Run training script with the correct LightningCLI format

python main.py fit \
    --config $CMRROOT/configs/base.yaml \
    --config $CMRROOT/configs/train/pmr-plus/CMR-37-duplicate-adjacent-timepoints.yaml \
    --config $CMRROOT/configs/model/pmr-plus-xl.yaml \
    --model.init_args.pretrain=True \
    --model.init_args.pretrain_weights_path=/common/lidxxlab/cmrchallenge/task3/PromptMR-plus-Task3/logs/cmr2025_task3/43qr6fxh/checkpoints/best-epochepoch=13-valvalidation_loss=0.0206.ckpt \
    --trainer.logger.init_args.save_dir=$SAVE_DIR/promptmr-plus/CMR-37-duplicate-adjacent-slices