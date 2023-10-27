FROM huggingface/transformers-pytorch-gpu

# Install Git and Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install


# Set working directory and copy files
WORKDIR /submission
COPY . /submission
#RUN pip install helm-crfm[all]
RUN pip install -r requirements.txt
# Install other dependencies

RUN pip install git+https://github.com/huggingface/transformers.git
RUN pip install git+https://github.com/huggingface/peft.git
RUN pip install git+https://github.com/huggingface/accelerate.git

# Run cache script and start FastAPI
RUN ls
CMD ["python3","-m","uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
