FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Ubuntu 22.04's default python3.11 is 3.11.0~rc1 (a 2022 release candidate
# with different CPython internal struct layouts than 3.11 stable). Triton
# 3.6.0's pybind11 bindings built against 3.11-stable ABI SIGSEGV on rc1
# inside IRBuilder's first setattr (verified via gdb py-bt on the Apptainer
# build, see textsum.def comments). The deadsnakes PPA ships current 3.11.x.
RUN apt-get update && apt-get install -y software-properties-common gnupg ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-distutils python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

COPY requirements.txt ./
# torch 2.10.0 wheels on cu128 index (matches CUDA 12.8.1 base)
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# v16: swap LLM to Qwen3-30B-A3B-Instruct-2507-FP8 (exp51 port, leak-free
# 0.7110, +0.012 over v15-K). 30 B MoE with 3 B active per token, FP8-e4m3
# weights ~30 GB → fits 40 GB A100 with KV headroom.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-30B-A3B-Instruct-2507-FP8')"

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_USE_V1=0

COPY model /model
WORKDIR /model
CMD python3 run.py
