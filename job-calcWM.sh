#!/bin/bash
#SBATCH --job-name=dl_train      # 作业名称
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00              # 作业的运行时间上限 (格式: hh:mm:ss)
#SBATCH --mem=16G                     # 请求的内存量
#SBATCH --output=./log/output_%j.log
#SBATCH --err=./log/error_%j.log

# Activate your Conda environment
# 加载所需模块或激活虚拟环境 (可选)
conda init
source ~/.bashrc
conda activate cmrxrecon
module load openblas
module load intel/mkl

# 打印一些调试信息
echo "Running on $(hostname)"
echo "Allocated GPUs: $SLURM_GPUS_PER_TASK"

param="$1"

echo "Worker script received parameter: $param"

# Run your script
python /common/lidxxlab/Yifan/PromptMR-plus/temp_debug.py $param