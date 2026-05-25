#!/bin/bash
#SBATCH --job-name=exp31_hyde
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp31_hyde_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp31_hyde_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$PROJECT/logs" \
         "$PROJECT/eval_retrieval/cache" \
         "$PROJECT/eval_retrieval/result"

cd "$PROJECT"

echo "=== exp31: HyDE generate (Qwen3-32B-AWQ, 1239 queries) ==="
python3 eval_retrieval/hyde_generate.py

echo ""
echo "=== exp31: HyDE retrieval-only eval (no LLM gen, just embed + metrics) ==="
python3 eval_retrieval/hyde_eval.py
