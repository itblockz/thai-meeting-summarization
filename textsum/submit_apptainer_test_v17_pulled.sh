#!/bin/bash
#SBATCH --job-name=v17_pulled_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_pulled_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_pulled_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v17_pulled_test_result"
mkdir -p "$RESULT" "$PROJECT/logs"

# Production verification for the GCB-built v17 image. Pull as the benchmark
# backend would (apptainer pull docker://...:v17 — built on Google Cloud Build,
# digest sha256:8df2ef4f6887...), then run with ONLY the binds the backend
# provides — test data, benchmark_lib, result. NO run.py bind: this tests the
# run.py BAKED INTO the image at build time, not a local edit. If this passes
# with submission.csv (50 rows) the production image is good.
#
# v17 = exp56 port (two-stage hybrid). Stage A: Qwen3.6-27B-FP8 + V10_factual
# → refs. Stage B: Qwen3-32B-AWQ + "เน้นย่อหน้า" hint → abstractive (refs
# FIXED to Stage A). The two stages share one 40 GB GPU sequentially — run.py
# does `del + gc.collect() + torch.cuda.empty_cache()` between them. This is
# the FIRST containerised run of the hybrid, so watch the Stage A→B handoff:
# Stage A's ~30 GB FP8 weights must be fully freed before Stage B's ~18 GB AWQ
# loads, else the second LLM() OOMs on the 40 GB card.
echo "=== v17 pulled image test (GCB build) — production-equivalent hybrid run ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>&1 || true
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env MAX_MODEL_LEN=32768 \
    --env LLM_MODEL_STAGE_A=Qwen/Qwen3.6-27B-FP8 \
    --env LLM_MODEL_STAGE_B=Qwen/Qwen3-32B-AWQ \
    "$PROJECT/textsum_v17_pulled.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
echo "--- submission.csv row count (expect 51 incl. header) ---"
wc -l "$RESULT/submission.csv" 2>/dev/null || echo "NO submission.csv produced"
