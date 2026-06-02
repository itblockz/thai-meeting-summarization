# syntax=docker/dockerfile:1
#
# v22 — two-stage hybrid (exp56 port). Stage A picks refs with
# gemma-4-31B-it-NVFP4 (4-bit NVFP4, ~22 GB — the exp73 model, swapped in
# for v21's Qwen3.6-27B-FP8); Stage B writes the answer with Qwen3-32B-AWQ
# (~18 GB INT4). Both stages share one 40 GB GPU sequentially (combined
# weights ~40 GB → cannot coexist; `del + empty_cache` between).
#
# MODEL WEIGHTS ARE NOT DOWNLOADED HERE. cloudbuild.yaml seeds them from a
# persistent GCS bucket (a HuggingFace cache that accumulates every model we
# have ever pulled), so a rebuild never re-fetches from HF — even after
# switching the Stage-A model away and back. The build context therefore
# carries `hf-cache/` (synced from GCS by the earlier Cloud Build steps), and
# we just COPY it in. Order matters: COPY hf-cache BEFORE COPY model so a
# run.py edit invalidates only the tiny final layer, never the ~40 GB weights.

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

# v21: the base image is CUDA *runtime* (no nvcc). On the H100 benchmark backend
# vLLM's Hopper FP8 fast paths (DeepGEMM + FlashInfer fp8 KV) JIT-compile CUDA
# kernels with nvcc+ninja at first use — v19 died here with `nvcc: not found`.
# Install the matching CUDA 12.8 compiler toolkit (nvcc + headers) + ninja so
# that JIT succeeds. cuda-minimal-build pulls nvcc/cudart-dev without the full
# ~6GB toolkit. run.py routes all JIT caches (TORCH_EXTENSIONS_DIR,
# CUDA_CACHE_PATH, FLASHINFER_WORKSPACE_BASE, HOME) to writable /scratch.
RUN apt-get update && apt-get install -y cuda-minimal-build-12-8 ninja-build && \
    rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH

COPY requirements.txt ./
# torch 2.10.0 wheels on cu128 index (matches CUDA 12.8.1 base)
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# Bake both models' HF cache (gemma-4-31B-NVFP4 + Qwen3-32B-AWQ), seeded from
# GCS by cloudbuild.yaml. This is the ~40 GB layer; it only changes when the
# model set changes, so run.py edits leave it cache-hit.
COPY hf-cache /root/.cache/huggingface

RUN mkdir -p /scratch && chmod 1777 /scratch

# vllm 0.19.1 is V1-ONLY (V0 removed; `VLLM_USE_V1` is no longer a recognised env
# var — setting it just warns). The two-stage handoff RELIES on V1 multiprocessing:
# each stage's EngineCore runs in a child process, so `del engine` tears that
# process down and the OS reclaims ALL its GPU memory before the next stage loads.
# In-process mode (VLLM_ENABLE_V1_MULTIPROCESSING=0) leaves Stage A's ~22 GB
# resident after del+empty_cache → Stage B OOMs on init (verified job 5824350:
# "Free memory 3.27/39.5 GiB"). Pin it ON so a future default flip can't break us.
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TEXTSUM_SCRATCH_DIR=/scratch \
    VLLM_ENABLE_V1_MULTIPROCESSING=1

COPY model /model
WORKDIR /model
CMD python3 run.py
