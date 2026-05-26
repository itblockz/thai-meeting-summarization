#!/bin/bash
#SBATCH --job-name=v12_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v12_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v12_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v12_test_result"
mkdir -p "$RESULT" "$PROJECT/logs"

# Production-like: only the binds the benchmark backend would provide.
# No run.py overlay, no writable HOME — verifies the SIF is self-contained.
echo "=== testing v12_local SIF — NO RETRIEVAL (full doc) + 2-shot few-shot + vLLM + Qwen3-32B-AWQ ==="
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env MAX_MODEL_LEN=32768 \
    "$PROJECT/textsum_v12_local.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
# Expect: exit 0 and a submission.csv (50 rows) in the result dir.
