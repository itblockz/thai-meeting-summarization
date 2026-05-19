#!/bin/bash
#SBATCH --job-name=textsum
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/textsum_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/textsum_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7

source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/model/test"
export RESULT_DIR="$PROJECT/textsum/result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$RESULT_DIR" "$PROJECT/logs"

cd "$PROJECT/textsum/model"
python3 run.py
