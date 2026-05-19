#!/bin/bash
set -e

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7

python3 -m venv "$PROJECT/venv"
source "$PROJECT/venv/bin/activate"

pip install --upgrade pip

# Install PyTorch with CUDA 12.1 (matches A100 support)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements
pip install -r "$PROJECT/textsum/requirements.txt"

echo "Setup complete. Venv at $PROJECT/venv"
