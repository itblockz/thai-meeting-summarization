#!/bin/bash
#SBATCH --job-name=v17_1_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_1_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_1_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v17_1_test_result"
mkdir -p "$RESULT" "$PROJECT/logs"

# v17.1 = exp81 s2ans_s2ref port (two-stage). Stage 1: gemma-4-26B-A4B-it-FP8-
# Dynamic (~26 GB FP8) V10_factual + exp38 shots → ref-index hint. Stage 2:
# Qwen3-30B-A3B-Instruct-2507-FP8 (~29 GB FP8) SAME V10 + the hint → BOTH the
# abstractive answer AND the refs. Leak-free A100 venv composite: 0.7207
# (ties exp56's 0.7215). NO H100-specific config — both stages are FP8 and run
# the precompiled CUTLASS/Marlin path (VLLM_USE_DEEP_GEMM=0 in run.py, no nvcc).
#
# Bind-mounts run.py over the container's /model/run.py for iteration without
# rebuilding the ~55 GB-weights SIF. --containall mirrors what the benchmark
# backend gives at submission time.
echo "=== v17.1 exp81 s2ans_s2ref port (two-stage) container test ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>&1 || true
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/run.py:/model/run.py:ro" \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env MAX_MODEL_LEN=32768 \
    "$PROJECT/textsum_v17_1_local.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
