#!/bin/bash
#SBATCH --job-name=v15_pulled_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v15_pulled_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v15_pulled_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v15_pulled_test_result"
mkdir -p "$RESULT" "$PROJECT/logs"

# Production verification: pull the v15 image as the benchmark backend would
# (apptainer pull docker://...:v15), then run with ONLY the binds the backend
# provides — test data, benchmark_lib, result. NO run.py bind: this tests
# the run.py BAKED INTO the image at Docker build time, not the local edit.
# If this passes with submission.csv (50 rows), the production image is good.
echo "=== v15 pulled image test — production-equivalent run ==="
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env MAX_MODEL_LEN=32768 \
    "$PROJECT/textsum_v15_pulled.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
