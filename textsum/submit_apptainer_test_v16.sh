#!/bin/bash
#SBATCH --job-name=v16_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v16_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v16_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v16_test_result"
mkdir -p "$RESULT" "$PROJECT/logs"

# v16 = exp51 port: Qwen3-30B-A3B-Instruct-2507-FP8 in place of v15-K's
# Qwen3-32B-AWQ. Same prompt / shots / retrieval (none) — single-
# variable model swap. Leak-free venv composite: 0.7110 (+0.012 vs v15).
#
# Bind-mounts run.py over the container's /model/run.py for iteration
# without rebuilding the ~30+ GB SIF. --containall mirrors what the
# benchmark backend gives at submission time.
echo "=== v16 exp51 port (Qwen3-30B-A3B-FP8) container test ==="
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/run.py:/model/run.py:ro" \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env MAX_MODEL_LEN=32768 \
    --env LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    "$PROJECT/textsum_v16_local.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
