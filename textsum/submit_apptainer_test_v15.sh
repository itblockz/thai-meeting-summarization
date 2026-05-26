#!/bin/bash
#SBATCH --job-name=v15_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v15_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v15_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v15_test_result"
mkdir -p "$RESULT" "$PROJECT/logs"

# v15 = match LANTA venv exactly (vllm 0.19.1, torch 2.10.0, cu128) +
# VLLM_USE_V1=0 to avoid the EngineCore subprocess segfault in Apptainer.
# Goal: kill the container-vs-venv greedy-decode drift so the context-first
# prompt's 0.6944 venv score transfers to the production container.
echo "=== testing v15_local SIF — vllm 0.19.1 + V0 engine + exp37 prompt ==="
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_WORKER_MULTIPROC_METHOD=fork \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    --env VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
    --env CUDA_LAUNCH_BLOCKING=1 \
    --env TORCH_USE_CUDA_DSA=1 \
    --env MAX_MODEL_LEN=32768 \
    "$PROJECT/textsum_v15_local.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
# Expect: exit 0, submission.csv (50 rows). If segfault: try
# --env VLLM_USE_V1=1 fallback and check VLLM_ENGINE_VERSION compat.
