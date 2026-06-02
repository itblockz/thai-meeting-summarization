#!/bin/bash
#SBATCH --job-name=exp79_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp79_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp79_eval_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/exp79/eval_result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export MAX_MODEL_LEN="32768"
export TP_SIZE="1"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$PROJECT/logs"

# NO override dir (unlike exp77/exp78): unsloth/Qwen3.6-35B-A3B-NVFP4 is a
# compressed-tensors PURE-NVFP4 checkpoint with NO baked kv_cache_quant_algo
# (verified on HF), so kv_cache_dtype="auto" resolves to bf16 naturally and
# the sm80 fp8e4nv trap does not apply. Point straight at the cached model
# (offline) like the exp73/76 RedHatAI runs.
export LLM_MODEL="unsloth/Qwen3.6-35B-A3B-NVFP4"

echo "=== exp79: unsloth/Qwen3.6-35B-A3B-NVFP4 (compressed-tensors pure NVFP4, single A100-40GB) + exp51 prompt/shots ==="
cd "$PROJECT/exp79"
python3 run.py

echo "=== exp79: scoring (full 1239, headline — has doc_050 few-shot leak) ==="
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"

echo "=== exp79: leak-free scoring (hold out doc_050, 1218 queries) ==="
cd "$PROJECT/textsum/eval_train"
python3 score_heldout.py "$RESULT_DIR/submission.csv" doc_050
