# start from nvidia pytorch image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

## Set workdir in Docker Container
# set default workdir(app) in your docker container
# In other words your scripts will run from this directory
RUN mkdir /app
WORKDIR /app

## Copy files into Docker Container
# Copy everything first, then remove unwanted directories
COPY . /app
COPY /common/lidxxlab/cmrchallenge/code/chaowei/experiments/cmr25/promptmr-plus/CMR2025/deep_recon/uec2kxvx/checkpoints/last.ckpt /app/last.ckpt

## Install python in Docker image
RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip

## Install requirements
RUN pip3 install -r /app/requirements.txt

## Entrypoint
ENTRYPOINT ["python", "main.py", "predict", "--config", "configs/inference/pmr-plus/cmr25-task2-docker.yaml"]