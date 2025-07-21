#!/bin/bash
#SBATCH --job-name=cmr_baseline\
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=/common/lidxxlab/cmrchallenge/task3/logs/cmr_baseline-15to30_db_%j.out
#SBATCH --error=/common/lidxxlab/cmrchallenge/task3/logs/cmr_baseline-15to30_db_%j.err

# Working directory
WORKDIR=/common/lidxxlab/cmrchallenge/task3/PromptMR-plus-Task3
cd $WORKDIR

# # save model checkpoints
# SAVE_DIR=/common/lidxxlab/cmrchallenge/task3/Experiments

# Create a log directory (make sure it exists)
mkdir -p /common/lidxxlab/cmrchallenge/task3/logs

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh
# conda activate promptmr
conda activate cmrrecon-task3 # Lisha's environment esplhpccompbio-lv03

# Debug: Print environment information
echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"

# # 设置环境变量。
# export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$WORKDIR:$PYTHONPATH

# Set up the wandb API key
export WANDB_API_KEY="670de942557209d00dd5c21b9dc54f7a28d218ea" # Lisha.Zeng@cshs.org
wandb login $WANDB_API_KEY

# Debug: Print wandb status
echo "Wandb status: $(wandb status)"

# Training parameters - increase epoch and batch size
CONFIG_FILE="configs/train/pmr-plus/cmr25-cardiac-task3_30to15.yaml"
MAX_EPOCHS=50  # Increase the number of epochs to complete the complete training
BATCH_SIZE=1  # Use a single sample for training, because the sample size is inconsistent.
# Note: Each GPU will process a sample, so the effective batch size is 4
echo "Running on node: $(hostname)"
echo "Detailed GPU Information:"
nvidia-smi -L
nvidia-smi

# Set the wandb debugging option
export WANDB_DEBUG=true
export WANDB_LOG_MODEL=true
export WANDB_WATCH=all

# Start training with debug logging
echo "Starting baseline training with PromptMR+..."
# fully configure mode   
srun python main.py fit \
    --config $CONFIG_FILE \
    --trainer.max_epochs=$MAX_EPOCHS \
    --data.init_args.batch_size=$BATCH_SIZE \
    --trainer.devices=4 \
    --trainer.logger.init_args.project="cmr2025_task3" \
    --trainer.logger.init_args.tags="[baseline,promptmr_plus,comparison]" \
    --trainer.logger.init_args.name="pmr_plus_baseline_comparison-15to30-db" \
    --trainer.logger.init_args.log_model="all" \
    --trainer.logger.init_args.offline=false

echo "Baseline training complete!"
