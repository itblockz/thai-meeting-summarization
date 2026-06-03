# syntax=docker/dockerfile:1
#
# MODEL WEIGHT IS NOT DOWNLOADED HERE. cloudbuild.yaml seeds it from a
# persistent GCS tar (a HuggingFace cache), so a rebuild never re-fetches from
# HF. The build context carries `hf-cache/` (restored from GCS by the earlier
# Cloud Build steps) and we just COPY it in. Order matters: COPY hf-cache
# BEFORE COPY model so a run.py edit invalidates only the tiny final layer,
# never the ~18 GB weights.
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

# v16.4 needs NO nvcc: run.py sets VLLM_USE_DEEP_GEMM=0 and forces bf16 KV
# (build_kv_neutralized_model strips the NVFP4 checkpoint's fp8-KV directive),
# so neither the DeepGEMM block-FP8 GEMM nor the fp8-KV FlashInfer path JIT-
# compiles on H100 — the precompiled NVFP4/Marlin path runs on both A100 and
# H100. The v21 hybrid's cuda-minimal-build/nvcc bake + CUDA_HOME env are
# dropped here — keeps this single-model image lean.

COPY requirements.txt ./
# torch 2.10.0 wheels on cu128 index (matches CUDA 12.8.1 base)
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# v16.4: single-model exp77 port. One model —
# nvidia/Gemma-4-26B-A4B-NVFP4 (same 26B MoE, ~4B active, NVFP4 experts
# ~18 GB load — attention stays bf16) — with the same V10_factual prompt +
# exp38 shots as v16.2/v16.3; only the quant publisher/format changes (v16.3's
# RedHatAI FP8 → nvidia/ModelOpt NVFP4). ⚠️ KNOWN-NEGATIVE on train (exp77
# leak-free 0.6982, flat vs v16.3's 0.6970, −0.0158 vs v16.2's 0.7140): the
# MoE's ~4B-active answer-quality ceiling caps RougeL/SS regardless of quant
# format. Kept as a packaged variant (this is the same checkpoint the exp77-
# Stage-A hybrid uses as its ref-picker), NOT a single-model quality win —
# prefer v16.2 for score. run.py builds a runtime KV override (strips the
# NVFP4 checkpoint's fp8-KV directive) so kv_cache_dtype="auto" → bf16.
#
# Bake the model's HF cache, seeded from the GCS tar by cloudbuild.yaml (NOT
# downloaded here — see the header). This is the ~18 GB-weights layer; it only
# changes when the model set changes, so a run.py edit leaves it cache-hit
# (COPY model below is the only busted layer).
COPY hf-cache /root/.cache/huggingface

RUN mkdir -p /scratch && chmod 1777 /scratch

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TEXTSUM_SCRATCH_DIR=/scratch \
    VLLM_USE_V1=0

COPY model /model
WORKDIR /model
CMD python3 run.py
