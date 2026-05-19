#!/bin/bash
#SBATCH --job-name=textsum-apptainer
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/logs/apptainer_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/logs/apptainer_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021

module load Apptainer/1.1.6

mkdir -p "$PROJECT/textsum/result" "$PROJECT/logs"

apptainer run --nv \
    --bind "$PROJECT/textsum/model:/model:ro" \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/result:/result:rw" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$PROJECT/.hf_cache:/root/.cache/huggingface:rw" \
    --env TEST_DIR=/model/test \
    --env RESULT_DIR=/result \
    --env PROGRESS_LIB=/benchmark_lib/progress \
    --env HF_HOME=/root/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$PROJECT/textsum/textsum.sif"
