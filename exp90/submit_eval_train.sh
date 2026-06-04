#!/bin/bash
#SBATCH --job-name=exp90_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp90_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp90_eval_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/exp90/eval_result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export MAX_MODEL_LEN="32768"
export TP_SIZE="1"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$PROJECT/logs"

# RedHatAI/Qwen3-32B-NVFP4 is compressed-tensors (nvfp4-pack-quantized): NO
# hf_quant_config.json, NO baked FP8-KV directive → kv_cache_dtype="auto"
# resolves to bf16 on its own. So — unlike exp89's nvidia/ModelOpt build — NO
# config-override dir is needed (exp79 confirmed the bare compressed-tensors
# snapshot runs clean on sm80). Point LLM_MODEL straight at the cached snapshot
# so HF_HUB_OFFLINE resolves it without a network hit.
export LLM_MODEL="RedHatAI/Qwen3-32B-NVFP4"

echo "=== exp90: Qwen3-32B-NVFP4 (RedHatAI/compressed-tensors, dense, single A100-40GB) + exp51 prompt/shots ==="
cd "$PROJECT/exp90"
python3 run.py

echo "=== exp90: scoring (full 1239, headline — has doc_050 few-shot leak) ==="
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"

echo "=== exp90: leak-free scoring (hold out doc_050, 1218 queries) ==="
cd "$PROJECT/textsum/eval_train"
python3 score_heldout.py "$RESULT_DIR/submission.csv" doc_050
