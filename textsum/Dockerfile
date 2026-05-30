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

# v17: two-stage hybrid (exp56 port, leak-free 0.7215, +0.0128 over v16's
# 0.7087). Stage A picks refs with Qwen3.6-27B-FP8 (dense, ~30 GB FP8);
# Stage B writes the answer with Qwen3-32B-AWQ (~18 GB INT4) + hint
# pointing at Stage A's refs. Both stages share one 40 GB GPU sequentially
# (combined weights 48 GB > 40 GB → cannot coexist; `del + empty_cache`
# between). Optimised for H100 40 GB (native FP8 + FA3 + FP8 KV cache,
# auto-detected at runtime); falls back gracefully on A100 (~exp56's
# 2h45m). Bundle BOTH model weights in the SIF (~50 GB image).
# Download each model in its OWN cached layer so a network drop (or a future
# rebuild) never re-pulls a model that already completed. With BuildKit, editing
# model/run.py below invalidates only the COPY layer — these ~45 GB weight layers
# stay cached and are NOT re-downloaded. hf_transfer = faster, resumable
# multi-part pulls for the large FP8/AWQ shards.
RUN pip3 install --no-cache-dir hf_transfer
RUN HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c \
    "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-27B-FP8')"
RUN HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c \
    "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-32B-AWQ')"

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_USE_V1=0

COPY model /model
WORKDIR /model
CMD python3 run.py
