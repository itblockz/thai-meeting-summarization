#!/bin/bash
#SBATCH --job-name=v17_1_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_1_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_1_eval_%j.err

# Validate the v17.1 CONTAINER run.py (textsum/model/run.py) on the train set in
# the LANTA venv BEFORE the ~1.5 h GCB image build. The container port is
# functionally identical to exp81's s2ans_s2ref combo (the doc-sort + streaming
# wrapper does not change per-query output), so this should reproduce exp81's
# leak-free 0.7207 (RougeL 0.4879 / SS 0.8607 / IoU 0.8132).

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/textsum/eval_train/result_v17.1"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export MAX_MODEL_LEN="32768"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$PROJECT/logs"

echo "=== v17.1 (exp81 s2ans_s2ref) container run.py — train eval ==="
cd "$PROJECT/textsum/model"
python3 run.py

echo ""
echo "######## scoring (full 1239) ########"
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"
echo "######## leak-free (excl. doc_050) ########"
cd "$PROJECT/textsum/eval_train"
python3 score_heldout.py "$RESULT_DIR/submission.csv" doc_050
