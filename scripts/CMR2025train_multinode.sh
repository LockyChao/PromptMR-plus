#!/bin/bash
#SBATCH --job-name=CMRscratch
#SBATCH --output=logs/%j.log
#SBATCH --error=errors/%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=168:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:h100:4

source activate cmr

CMRROOT=$PWD
SAVE_DIR=$PWD/exp

echo "Detailed GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


# Run training script with the correct LightningCLI format
srun python main.py fit \
       -c $CMRROOT/configs/base.yaml \
       -c $CMRROOT/configs/train/pmr-plus/cmr25-cardiac-upd.yaml \
       -c $CMRROOT/configs/model/pmr-plus.yaml \
       --trainer.devices=4 \
       --trainer.num_nodes=2 \
       --trainer.logger.init_args.save_dir=$SAVE_DIR/promptmr-plus/CMR2025 \
       --model.init_args.pretrain=True \
       --model.init_args.pretrain_weights_path=/common/lidxxlab/cmrchallenge/code/PromptMR-plus/weights/cmr24-cardiac/promptmr-plus-epoch=11-step=337764.ckpt \