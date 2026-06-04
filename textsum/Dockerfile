# syntax=docker/dockerfile:1
#
# MODEL WEIGHTS ARE NOT DOWNLOADED HERE. cloudbuild.yaml seeds them from a
# persistent GCS tar (a HuggingFace cache), so a rebuild never re-fetches from
# HF. The build context carries `hf-cache/` (restored from GCS by the earlier
# Cloud Build steps) and we just COPY it in. Order matters: COPY hf-cache
# BEFORE COPY model so a run.py edit invalidates only the tiny final layer,
# never the ~47 GB weights.
#
# v17.3 = independent column-merge (NOT v17.1's coupled hint pipe). Two single-
# model images run independently; the final CSV takes the ANSWER from v15.2 and
# the REFS from v16.4 (exp86's recipe). v17.3 = v17.2 with Stage 2 swapped from
# v16.1's A3B to v15.2's Qwen3-32B-AWQ. The hf-cache holds BOTH models:
# nvidia/Gemma-4-26B-A4B-NVFP4 (refs, Stage 1, ~18 GB) + Qwen/Qwen3-32B-AWQ
# (answer, Stage 2, ~18 GB); cloudbuild downloads both into the cache tar.
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

# v17.3 needs NO nvcc: run.py sets VLLM_USE_DEEP_GEMM=0 and keeps
# kv_cache_dtype="auto" (Stage 1's NVFP4 gemma has its baked fp8-KV directive
# stripped at runtime by build_kv_neutralized_model so "auto" → bf16), forcing
# the precompiled CUTLASS/Marlin/NVFP4 path on both A100 and H100 (no runtime
# JIT). The v21 hybrid's cuda-minimal-build/nvcc bake + CUDA_HOME env are
# dropped here — keeps the image lean.

COPY requirements.txt ./
# torch 2.10.0 wheels on cu128 index (matches CUDA 12.8.1 base)
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# v17.3: independent column-merge (32B-AWQ answer + gemma-NVFP4 refs = exp86's
# recipe, NEW BEST 0.7235). Stage 1 = nvidia/Gemma-4-26B-A4B-NVFP4 (cold V10 →
# REFS only), Stage 2 = Qwen/Qwen3-32B-AWQ (E5 prompt → ANSWER only). NO hint
# between them — each runs as its single-model image (v16.4 / v15.2). run.py
# runs each stage in its OWN subprocess (the orchestrator never touches CUDA);
# the worker process exits between stages so the GPU is genuinely fresh for the
# second model → each stage CSV is byte-identical to v16.4 / v15.2 standalone,
# then the orchestrator merges columns.
#
# Bake both models' HF cache, seeded from the GCS tar by cloudbuild.yaml (NOT
# downloaded here — see the header). This is the ~36 GB layer; it only changes
# when the model set changes, so a run.py edit leaves it cache-hit (COPY model
# below is the only busted layer).
COPY hf-cache /root/.cache/huggingface

RUN mkdir -p /scratch && chmod 1777 /scratch

# GPU teardown between stages is handled by the OS: run.py spawns one process
# per stage, and a worker's full exit reclaims ALL its VRAM before the next
# worker spawns (no in-process `del engine` / empty_cache needed). vllm 0.19.1
# is V1-ONLY (`VLLM_USE_V1` is a dead env var). VLLM_ENABLE_V1_MULTIPROCESSING=1
# is kept (harmless; each worker still spawns its EngineCore child) but is no
# longer load-bearing for the handoff — the subprocess boundary is.
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TEXTSUM_SCRATCH_DIR=/scratch \
    VLLM_ENABLE_V1_MULTIPROCESSING=1

COPY model /model
WORKDIR /model
CMD python3 run.py
