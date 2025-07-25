#!/bin/bash
#SBATCH --job-name=radio      # 作业名称
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=80:00:00              # 作业的运行时间上限 (格式: hh:mm:ss)
#SBATCH --mem=64G                     # 请求的内存量
#SBATCH --output=output_%j.log
#SBATCH --error=output_err%j.log

#md5sum ChallengeDataTrain.tar.gz > trainmd5.txt || bash ~/mailsender.sh
#mkdir TestGroundtruth
#tar -xvzf ChallengeDataTest.tar.gz -C TestGroundtruth


#2024 train fix symbolic issue
# python prepare_h5_dataset_cmrxrecon_modified.py \
#         --input_matlab_folder /common/lidxxlab/cmrchallenge/data/CMR2024/ChallengeData/MultiCoil \
#         --output_h5_folder /common/lidxxlab/cmrchallenge/data/CMR2024/Processed/MultiCoil \
#         --split_json configs/data_split/cmr24-cardiac.json \
#         --year 2024 \
#         --train_valid_test train \
#         --symbol_only 1

# # 2025 train fix t1w t2w black blood dim
python prepare_h5_dataset_cmrxrecon_addmeta.py \
        --input_matlab_folder /common/lidxxlab/cmrchallenge/data/CMR2025/ChallengeData/MultiCoil \
        --output_h5_folder /common/lidxxlab/Yifan/PromptMR-plus/CMR2025/Processed_addmeta/MultiCoil \
        --split_json configs/data_split/cmr25-cardiac.json \
        --year 2025 \
        --train_valid_test train 
        
# bash ~/mailsender.sh
