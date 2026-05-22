FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

COPY requirements.txt ./
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-m3'); \
snapshot_download('Qwen/Qwen3-32B-AWQ'); \
snapshot_download('BAAI/bge-reranker-v2-m3')"

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY model /model
WORKDIR /model
CMD python3 run.py
