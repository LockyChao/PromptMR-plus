#md5sum ChallengeDataTrain.tar.gz > trainmd5.txt || bash ~/mailsender.sh
#mkdir TestGroundtruth
#tar -xvzf ChallengeDataTest.tar.gz -C TestGroundtruth
cd /common/lidxxlab/cmrchallenge/code/PromptMR-plus
python prepare_h5_dataset_cmrxrecon_after.py \
        --input_matlab_folder /common/lidxxlab/cmrchallenge/data/CMR2024_After/groundtruth_mat/MultiCoil \
        --output_h5_folder /common/lidxxlab/cmrchallenge/data/CMR2024_After/Processed \
        --split_json configs/data_split/cmr24-cardiac.json \
        --year 2024 \
        --train_valid_test valid 
python prepare_h5_dataset_cmrxrecon_after.py \
        --input_matlab_folder /common/lidxxlab/cmrchallenge/data/CMR2024_After/groundtruth_mat/MultiCoil \
        --output_h5_folder /common/lidxxlab/cmrchallenge/data/CMR2024_After/Processed \
        --split_json configs/data_split/cmr24-cardiac.json \
        --year 2024 \
        --train_valid_test test 

bash ~/mailsender.sh
