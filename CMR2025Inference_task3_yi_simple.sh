#!/bin/bash
#SBATCH --job-name=cmr_inference_simple
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=512:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=esplhpc-cp080
#SBATCH --output=/common/lidxxlab/cmrchallenge/task3/logs/cmr_inference_task3_yi_simple_%j.out
#SBATCH --error=/common/lidxxlab/cmrchallenge/task3/logs/cmr_inference_task3_yi_simple_%j.err

Working directory
WORKDIR=/common/lidxxlab/cmrchallenge/task3/PromptMR-plus-Task3_clean
cd $WORKDIR

# Create a log directory (make sure it exists)
mkdir -p /common/lidxxlab/cmrchallenge/task3/logs

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cmr

# Debug: Print environment information
echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"

Set Python path
export PYTHONPATH=$WORKDIR:$PYTHONPATH
export CMR_SAVE_AS_MAT=1

# Inference parameters
CONFIG_FILE="$WORKDIR/yi_configs/inference/pmr-plus/cmr-task4-val-no-wandb.yaml"
BATCH_SIZE=1   # Keep batch size 1

# Note: Each GPU will process a sample, so the effective batch size is 2 (1*2)
echo "Running on node: $(hostname)"
echo "Detailed GPU Information:"
nvidia-smi -L
nvidia-smi

# Print inference info
echo "=== PROMPTMR PLUS INFERENCE ==="
echo "Batch size per GPU: $BATCH_SIZE (effective: $(($BATCH_SIZE * 2)))"
echo "Model: PromptMR Plus (standard reconstruction)"
echo "Config: $CONFIG_FILE"
echo "=============================="

# Start inference with PromptMR Plus (no wandb, no logger)
echo "Starting PromptMR Plus inference..."
srun python $WORKDIR/main.py predict \
    --config $CONFIG_FILE \


echo "PromptMR Plus baseline inference complete!"
