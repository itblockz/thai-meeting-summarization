#!/bin/bash
#SBATCH --job-name=exp16_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp16_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp16_eval_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/exp16/eval_result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$PROJECT/logs"

echo "=== exp16: multi-doc length-matched (Q0351+Q0626), held-out doc_045+doc_031 ==="
cd "$PROJECT/exp16"
python3 run.py

echo "=== exp16: scoring (excluding doc_045+doc_031 queries) ==="
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"
