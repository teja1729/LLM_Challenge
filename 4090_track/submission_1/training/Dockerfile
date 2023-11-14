FROM huggingface/transformers-pytorch-gpu
# Install Git and Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install

WORKDIR /train
COPY . /train

RUN pip install -r requirements.txt

RUN pip install git+https://github.com/huggingface/transformers.git
RUN pip install git+https://github.com/huggingface/peft.git
RUN pip install git+https://github.com/huggingface/accelerate.git


RUN echo "training_starts!!"

RUN python3 main.py
