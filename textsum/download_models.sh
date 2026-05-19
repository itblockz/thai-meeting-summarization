#!/bin/bash
set -e

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021

source "$PROJECT/venv/bin/activate"

export HF_HOME="$PROJECT/.hf_cache"

echo "Downloading BAAI/bge-m3 (~2.3GB)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3')
print('bge-m3 done')
"

echo "Downloading Qwen/Qwen2.5-7B-Instruct (~15GB)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct')
print('Qwen2.5-7B done')
"

echo "All models downloaded to $HF_HOME"
