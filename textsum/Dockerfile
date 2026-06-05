# syntax=docker/dockerfile:1
#
# MODEL WEIGHTS ARE NOT DOWNLOADED HERE. cloudbuild.yaml seeds them from a
# persistent GCS tar (a HuggingFace cache), so a rebuild never re-fetches from
# HF. The build context carries `hf-cache/` (restored from GCS by the earlier
# Cloud Build steps) and we just COPY it in. Order matters: COPY hf-cache
# BEFORE COPY model so a run.py edit invalidates only the tiny final layer,
# never the ~47 GB weights.
#
# v17.4 = v17.3 + COUPLED ref-index hint (the TRUE exp86 recipe, leak-free
# 0.7235). Stage 1 (gemma) refs HINT Stage 2 (32B-AWQ); the final CSV takes the
# ANSWER from Stage 2 (now generated UNDER the hint) and the REFS from Stage 1
# (exp86 ansB_refA). v17.3 had run the two models independently and column-
# merged (the invalid ~0.7243 ceiling, answer never saw the hint). Same two
# models as v17.3; the hf-cache holds BOTH:
# nvidia/Gemma-4-26B-A4B-NVFP4 (refs+hint, Stage 1, ~18 GB) + Qwen/Qwen3-32B-AWQ
# (hinted answer, Stage 2, ~18 GB); cloudbuild downloads both into the cache tar.
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

# v17.4 needs NO nvcc: run.py sets VLLM_USE_DEEP_GEMM=0 and keeps
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

# v17.4: coupled ref-hint (32B-AWQ hinted answer + gemma-NVFP4 refs = exp86's
# ansB_refA, NEW BEST 0.7235). Stage 1 = nvidia/Gemma-4-26B-A4B-NVFP4 (cold V10
# → REFS + a per-query ref-index hint sidecar), Stage 2 = Qwen/Qwen3-32B-AWQ
# (E5 prompt + Stage 1's hint line → ANSWER). run.py runs each stage in its OWN
# subprocess (the orchestrator never touches CUDA); the worker exits between
# stages so the GPU is genuinely fresh for the second model. Stage 1's CSV is
# still byte-identical to v16.4 standalone (cold); Stage 2 is hinted (NOT
# v15.2), and the handoff is the hint JSON Stage 1 writes before it exits.
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
