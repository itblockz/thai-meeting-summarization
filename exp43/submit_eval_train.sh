#!/bin/bash
#SBATCH --job-name=exp43_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp43_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp43_eval_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
module load cudatoolkit/24.11_12.6
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/exp43/eval_result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export MAX_MODEL_LEN="16384"
export TP_SIZE="1"
export LLM_MODEL="Qwen/Qwen3-32B-FP8"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDAHOSTCXX=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin/nvc++
export NVCC_CCBIN=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin/nvc++
export CXX=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin/nvc++

mkdir -p "$RESULT_DIR" "$PROJECT/logs"

echo "=== exp43: Qwen3.6-27B-FP8 (single A100-40GB) + exp38 prompt/shots ==="
cd "$PROJECT/exp43"
python3 run.py

echo "=== exp43: scoring (full 1239, headline — has doc_050 few-shot leak) ==="
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"

echo "=== exp43: leak-free scoring (hold out doc_050, 1218 queries) ==="
cd "$PROJECT/textsum/eval_train"
python3 score_heldout.py "$RESULT_DIR/submission.csv" doc_050
