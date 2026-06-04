#!/bin/bash
#SBATCH --job-name=v17_3_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_3_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_3_eval_%j.err

# Validate the v17.3 CONTAINER run.py (textsum/model/run.py) on the train set in
# the LANTA venv BEFORE the ~1.5 h GCB image build. v17.3 = independent column-
# merge: ANSWER from v15.2 (Qwen3-32B-AWQ) + REFS from v16.4 (gemma NVFP4), each
# model run as its single-model image with NO hint between them. This is exp86's
# recipe — expectation = 32B-AWQ's RougeL/SS (exp37/exp38 line) + gemma's IoU
# (exp77 0.8155) = NEW BEST ~0.7235 leak-free. NOTE: run.py runs each stage in
# its OWN subprocess on ONE A100-40GB — gemma NVFP4 (~18 GB) exits fully before
# 32B-AWQ (~18 GB) loads, so the GPU is fresh for the second model (no shared
# CUDA state; each stage CSV == v16.4 / v15.2 standalone). Side outputs:
# $TEXTSUM_SCRATCH_DIR/v17_3_stage{1,2}.csv (scoreable vs v16.4 / v15.2).

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/textsum/eval_train/result_v17.3"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export MAX_MODEL_LEN="32768"
# gemma NVFP4 bakes an fp8-KV directive that run.py strips into a scratch
# override dir (build_kv_neutralized_model); give it a writable Lustre path.
export TEXTSUM_SCRATCH_DIR="$PROJECT/textsum/scratch_v17.3"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$TEXTSUM_SCRATCH_DIR" "$PROJECT/logs"

echo "=== v17.3 (32B-AWQ answer + gemma-NVFP4 refs, independent) container run.py — train eval ==="
cd "$PROJECT/textsum/model"
python3 run.py

echo ""
echo "######## scoring (full 1239) ########"
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"
echo "######## leak-free (excl. doc_050) ########"
cd "$PROJECT/textsum/eval_train"
python3 score_heldout.py "$RESULT_DIR/submission.csv" doc_050
