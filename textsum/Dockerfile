FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

COPY requirements.txt ./
# torch 2.10.0 wheels on cu128 index (matches CUDA 12.8.1 base)
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-32B-AWQ')"

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_USE_V1=0

COPY model /model
WORKDIR /model
CMD python3 run.py
