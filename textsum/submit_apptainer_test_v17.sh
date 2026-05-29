#!/bin/bash
#SBATCH --job-name=v17_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v17_test_result"
mkdir -p "$RESULT" "$PROJECT/logs"

# v17 = exp56 port (two-stage hybrid). Stage A: Qwen3.6-27B-FP8 (~30 GB
# FP8) with V10_factual prompt + exp38 shots → fresh refs. Stage B:
# Qwen3-32B-AWQ (~18 GB INT4) with exp38 E5 prompt + "เน้นย่อหน้า"
# hint pointing at Stage A's refs → fresh abstractive (refs FIXED to
# Stage A). Leak-free A100 venv composite: 0.7215 (+0.0128 vs v16).
#
# Optimised for H100 40 GB single GPU (auto-detected at runtime):
#   - native FP8 GEMM on Stage A (no Marlin software dequant)
#   - FlashAttention-3 attention backend
#   - FP8 KV cache (halves KV memory → ~2× concurrent prompts)
#   - max_num_batched_tokens=16384 for long-context throughput
# Falls back gracefully on A100 (kv_cache_dtype="auto" + MarlinFP8).
#
# Bind-mounts run.py over the container's /model/run.py for iteration
# without rebuilding the ~50 GB SIF. --containall mirrors what the
# benchmark backend gives at submission time.
echo "=== v17 exp56 port (two-stage hybrid, H100 40GB optimised) container test ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>&1 || true
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/run.py:/model/run.py:ro" \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env MAX_MODEL_LEN=32768 \
    --env LLM_MODEL_STAGE_A=Qwen/Qwen3.6-27B-FP8 \
    --env LLM_MODEL_STAGE_B=Qwen/Qwen3-32B-AWQ \
    "$PROJECT/textsum_v17_local.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
