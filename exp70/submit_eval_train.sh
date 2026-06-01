#!/bin/bash
#SBATCH --job-name=exp70_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp70_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp70_eval_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/exp70/eval_result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
# 2-shot full-doc (doc_050 x2 ~45.7K) + largest real doc ~27.6K => ~75K.
# 1 GPU only (per requirement). 82K OOM'd (KV 7.5 > 6.75 avail), so
# max_model_len trimmed to 77824 — still above the worst actual prompt
# (~75K; no real doc exceeds doc_006's 27.6K) — and gpu_mem_util raised
# to 0.97 in run.py for the KV headroom.
export MAX_MODEL_LEN="77824"
export TP_SIZE="1"
export LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$PROJECT/logs"

echo "=== exp70: exp51 + FULL-DOC 2-shot (Q0745+Q0746), A3B+V10, 1 GPU 82K ==="
cd "$PROJECT/exp70"
python3 run.py

echo "=== exp70: scoring (full 1239, headline — has doc_050 few-shot leak) ==="
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"

echo "=== exp70: leak-free scoring (hold out doc_050, 1218 queries) ==="
cd "$PROJECT/textsum/eval_train"
python3 score_heldout.py "$RESULT_DIR/submission.csv" doc_050
