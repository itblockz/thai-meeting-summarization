#!/bin/bash
#SBATCH --job-name=qwen3_rr
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/qwen3_rr_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/qwen3_rr_%j.err

# Prereq (one-time, on the LOGIN node — compute partitions are offline):
#   python eval_retrieval/download_qwen3_reranker.py
# Prereq (one-time):  sbatch eval_retrieval/submit_rerank.sh   # builds the bge pool we reuse

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Override knobs (defaults match the script):
#   QWEN_RERANKER=Qwen/Qwen3-Reranker-4B   # cheaper sanity run
#   BATCH_SIZE=16                          # if VRAM allows
#   MAX_LENGTH=4096                        # for long paragraphs
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"

mkdir -p "$PROJECT/logs" "$PROJECT/eval_retrieval/cache"

cd "$PROJECT"
python3 eval_retrieval/rerank_qwen3_cache.py

echo "=== exit code: $? ==="
echo "compare with bge:"
echo "  python3 eval_retrieval/eval.py --method rerank \\"
echo "      --rerank-cache eval_retrieval/cache/rerank_train.json"
echo "  python3 eval_retrieval/eval.py --method rerank \\"
echo "      --rerank-cache eval_retrieval/cache/rerank_qwen3_train.json"
