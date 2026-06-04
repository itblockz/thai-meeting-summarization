#!/bin/bash
#SBATCH --job-name=v17_2_pulled_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_2_pulled_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_2_pulled_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v17_2_pulled_test_result"
# /scratch bind + compile-cache envs: ONLY because this LOCAL apptainer
# --containall sim has a read-only rootfs + tiny /tmp tmpfs → vLLM/Triton JIT
# would ENOSPC (the v22 trap). The REAL backend runs the image with a writable
# rootfs, so these are unnecessary there (the baked /scratch + ~/.triton work).
SCRATCHDIR="$PROJECT/textsum_v17_2_pulled_scratch"
mkdir -p "$RESULT" "$PROJECT/logs" "$SCRATCHDIR/triton" "$SCRATCHDIR/xdg"

# Production verification for the GCB-built v17.2 image. Pull as the benchmark
# backend would (apptainer pull docker://...:v17.2), then run with ONLY the
# binds the backend provides — test data, benchmark_lib, result. NO run.py
# bind: this tests the run.py BAKED INTO the image, not a local edit. If this
# passes with submission.csv the production image is good.
#
# v17.2 = independent column-merge. Stage 1: nvidia/Gemma-4-26B-A4B-NVFP4
# (~18 GB) + V10_factual → REFS only. Stage 2: Qwen3-30B-A3B-Instruct-2507-FP8
# (~29 GB) + same cold V10 → ANSWER only. No hint between them; final CSV = A3B
# answer + gemma refs. run.py runs each stage in its OWN subprocess on one
# 40 GB GPU: gemma's worker exits fully (OS reclaims its VRAM) before A3B's
# worker spawns, so the second model loads on a fresh card — no in-process
# teardown, no OOM risk from residual Stage-1 weights.
echo "=== v17.2 pulled image test (GCB build) — production-equivalent two-stage run ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>&1 || true
apptainer exec --nv --containall --pwd /model \
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
    "$PROJECT/textsum_v17_2_pulled.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
echo "--- submission.csv row count (expect 51 incl. header) ---"
wc -l "$RESULT/submission.csv" 2>/dev/null || echo "NO submission.csv produced"
