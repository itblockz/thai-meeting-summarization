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

# v16.2 needs NO nvcc: run.py sets VLLM_USE_DEEP_GEMM=0 and keeps
# kv_cache_dtype="auto", forcing the precompiled Marlin/Triton path on both
# A100 and H100 (no runtime JIT). The v21 hybrid's cuda-minimal-build/nvcc bake
# + CUDA_HOME env are dropped here — keeps this single-model image lean.

COPY requirements.txt ./
# torch 2.10.0 wheels on cu128 index (matches CUDA 12.8.1 base)
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# v16.2: single-model exp73 port (leak-free 0.7140, +0.0030 over v16.1's
# 0.7110). One model — RedHatAI/gemma-4-31B-it-NVFP4 (4-bit weights, ~22 GB) —
# with the same V10_factual prompt + exp38 shots as v16.1; only the LLM
# changes. Single-model fallback to the v21 two-stage hybrid (~50 GB) on
# master (~22 GB image — smaller than v16.1's A3B-FP8). Its own cached layer
# so a network drop / future rebuild never re-pulls a completed model. With
# BuildKit, editing model/run.py below invalidates only the COPY layer — the
# ~22 GB weight layer stays cached. hf_transfer = faster, resumable
# multi-part pulls for the large shards.
RUN pip3 install --no-cache-dir hf_transfer
RUN HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c \
    "from huggingface_hub import snapshot_download; snapshot_download('RedHatAI/gemma-4-31B-it-NVFP4')"

RUN mkdir -p /scratch && chmod 1777 /scratch

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TEXTSUM_SCRATCH_DIR=/scratch \
    VLLM_USE_V1=0

COPY model /model
WORKDIR /model
CMD python3 run.py
