#!/bin/bash
#SBATCH --job-name=CMRscratch
#SBATCH --output=logs/%j.log
#SBATCH --error=errors/%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=168:00:00
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
    --config $CMRROOT/configs/train/pmr-plus/CMR-38-mixed-24-25.yaml \
    --config $CMRROOT/configs/model/pmr-plus-xl.yaml \
    --trainer.logger.init_args.save_dir=$SAVE_DIR/promptmr-plus/CMR2025\
    --model.init_args.pretrain=False