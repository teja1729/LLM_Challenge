FROM huggingface/transformers-pytorch-gpu
# Install Git and Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install

WORKDIR /train
COPY . /train

RUN pip install -r requirements.txt


RUN echo "Docker build and training can be started!!"
