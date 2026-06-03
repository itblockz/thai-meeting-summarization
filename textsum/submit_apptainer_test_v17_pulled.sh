#!/bin/bash
#SBATCH --job-name=v17_1_pulled_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_1_pulled_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_1_pulled_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v17_1_pulled_test_result"
mkdir -p "$RESULT" "$PROJECT/logs"

# Production verification for the GCB-built v17.1 image. Pull as the benchmark
# backend would (apptainer pull docker://...:v17.1), then run with ONLY the
# binds the backend provides — test data, benchmark_lib, result. NO run.py
# bind: this tests the run.py BAKED INTO the image, not a local edit. If this
# passes with submission.csv the production image is good.
#
# v17.1 = exp81 s2ans_s2ref port (two-stage). Stage 1: gemma-4-26B-A4B-it-FP8-
# Dynamic + V10_factual → ref-index hint. Stage 2: Qwen3-30B-A3B-Instruct-2507-
# FP8 + the hint → BOTH abstractive AND refs. The two FP8 models share one
# 40 GB GPU sequentially — run.py does `del + gc.collect() +
# torch.cuda.empty_cache()` between them (V1 child-process teardown). Watch the
# Stage 1→2 handoff: gemma's ~26 GB must be fully freed before A3B's ~29 GB
# loads, else the second engine OOMs on the 40 GB card.
echo "=== v17.1 pulled image test (GCB build) — production-equivalent two-stage run ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>&1 || true
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env MAX_MODEL_LEN=32768 \
    "$PROJECT/textsum_v17_1_pulled.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
echo "--- submission.csv row count (expect 51 incl. header) ---"
wc -l "$RESULT/submission.csv" 2>/dev/null || echo "NO submission.csv produced"
