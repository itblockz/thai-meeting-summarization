#!/bin/bash
#SBATCH --job-name=v17_2_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_2_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_2_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v17_2_test_result"
# Under --containall the image rootfs is read-only and /tmp is a tiny tmpfs, so
# vLLM/Triton kernel JIT compile hits ENOSPC (the v22 local-test trap). Bind an
# ample Lustre dir over /scratch and point the compile caches at it. (run.py
# already falls back to RESULT_DIR for its own scratch, but vLLM's internal
# Triton cache is NOT under run.py's control → must be a writable env.)
SCRATCHDIR="$PROJECT/textsum_v17_2_scratch"
mkdir -p "$RESULT" "$PROJECT/logs" "$SCRATCHDIR/triton" "$SCRATCHDIR/xdg"

# v17.2 = independent column-merge (NOT v17.1's coupled hint pipe). Stage 1:
# nvidia/Gemma-4-26B-A4B-NVFP4 (~18 GB) V10_factual + exp38 shots → REFS only.
# Stage 2: Qwen3-30B-A3B-Instruct-2507-FP8 (~29 GB) SAME cold V10 → ANSWER only.
# No hint between them. Final CSV = A3B answer + gemma refs (exp80-85 "best of
# both", ~0.715-0.7205 leak-free). NO H100-specific config — VLLM_USE_DEEP_GEMM
# =0 + the runtime KV-directive strip (build_kv_neutralized_model) force the
# precompiled NVFP4/Marlin + bf16-KV path, no nvcc.
#
# Bind-mounts run.py over the container's /model/run.py for iteration without
# rebuilding the ~47 GB-weights SIF. --containall mirrors what the benchmark
# backend gives at submission time. run.py runs each stage in its OWN
# subprocess: gemma's worker (~18 GB) fully exits before A3B's worker (~29 GB)
# spawns, so the 40 GB card is fresh for the second model.
echo "=== v17.2 (A3B answer + gemma-NVFP4 refs, independent) container test ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>&1 || true
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/run.py:/model/run.py:ro" \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --bind "$SCRATCHDIR:/scratch" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env MAX_MODEL_LEN=32768 \
    --env TEXTSUM_SCRATCH_DIR=/scratch \
    --env TRITON_CACHE_DIR=/scratch/triton \
    --env XDG_CACHE_HOME=/scratch/xdg \
    --env HOME=/scratch \
    "$PROJECT/textsum_v17_2_local.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
