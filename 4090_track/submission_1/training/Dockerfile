FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Git and Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install


RUN apt-get install -y python3 python3-pip

RUN python3 --version 

WORKDIR /train
COPY . /train

RUN pip install -r requirements.txt
