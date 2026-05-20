#!/bin/bash
#SBATCH --job-name=v7_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v7_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v7_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v7_test_result"
mkdir -p "$RESULT" "$PROJECT/logs"

echo "=== testing v7_local SIF — tzdata fix included, NO bind mount ==="
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    "$PROJECT/textsum_v7_local.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
