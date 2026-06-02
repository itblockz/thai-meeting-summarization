# syntax=docker/dockerfile:1
#
# v22 — two-stage hybrid (exp56 port). Stage A picks refs with
# gemma-4-31B-it-NVFP4 (4-bit NVFP4, ~22 GB — the exp73 model, swapped in
# for v21's Qwen3.6-27B-FP8); Stage B writes the answer with Qwen3-32B-AWQ
# (~18 GB INT4). Both stages share one 40 GB GPU sequentially (combined
# weights ~40 GB → cannot coexist; `del + empty_cache` between).
#
# MULTI-STAGE FOR PER-MODEL BUILD CACHE: each model is downloaded in its OWN
# leaf stage (`dl-gemma`, `dl-awq`) that derives from `base`. Changing one
# model's snapshot_download line invalidates ONLY that stage — the other
# model stays a cache hit (no cascade, unlike a single linear Dockerfile
# where a later RUN re-downloads whenever an earlier one changes). On GCB
# (ephemeral workers, no local layer cache) this only pays off with a
# registry cache backend — cloudbuild.yaml uses
# `buildx --cache-to/--cache-from type=registry,...,mode=max`, which (unlike
# inline cache) exports EVERY intermediate stage so a fresh VM restores the
# already-downloaded model stages instead of re-pulling ~40 GB from HF.

# ---- base: python + CUDA toolkit + pip deps (shared by every stage) ----------
FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS base

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
# hf_transfer = faster, resumable multi-part pulls for the large FP8/NVFP4/AWQ
# shards. In `base` so both download stages share the single cached install.
RUN pip3 install --no-cache-dir hf_transfer

# ---- dl-gemma: Stage A weights ONLY (own stage → isolated cache) -------------
FROM base AS dl-gemma
RUN HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c \
    "from huggingface_hub import snapshot_download; snapshot_download('RedHatAI/gemma-4-31B-it-NVFP4')"

# ---- dl-awq: Stage B weights ONLY (own stage → isolated cache) ---------------
FROM base AS dl-awq
RUN HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c \
    "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-32B-AWQ')"

# ---- final: assemble both model caches + the app onto base -------------------
# Each COPY --from merges that stage's HF hub tree (different models → disjoint
# subdirs, no overwrite). A run.py edit invalidates ONLY the `COPY model` layer;
# the two weight COPY layers ride on the cached dl-* stages.
FROM base AS final
COPY --from=dl-gemma /root/.cache/huggingface /root/.cache/huggingface
COPY --from=dl-awq   /root/.cache/huggingface /root/.cache/huggingface

RUN mkdir -p /scratch && chmod 1777 /scratch

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TEXTSUM_SCRATCH_DIR=/scratch \
    VLLM_USE_V1=0

COPY model /model
WORKDIR /model
CMD python3 run.py
