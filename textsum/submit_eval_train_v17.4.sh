#!/bin/bash
#SBATCH --job-name=v17_4_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_4_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_4_eval_%j.err

# Validate the v17.4 CONTAINER run.py (textsum/model/run.py) on the train set in
# the LANTA venv BEFORE the ~1.5 h GCB image build. v17.4 = v17.3 + COUPLED ref-
# index hint (the TRUE exp86 recipe): Stage 1 (gemma NVFP4) refs HINT Stage 2
# (Qwen3-32B-AWQ); final = Stage 2's hinted ANSWER + Stage 1's REFS (exp86
# ansB_refA). Expectation = exp86 0.7235 leak-free (RougeL 0.4913 / SS 0.8631 /
# IoU 0.8155); slightly under v17.3's column-merge ~0.7243 — the answer is now
# generated UNDER the hint, which is what exp86 actually scored. NOTE: run.py
# runs each stage in its OWN subprocess on ONE A100-40GB — gemma NVFP4 (~18 GB)
# exits fully (writing the hint sidecar) before 32B-AWQ (~18 GB) loads and reads
# it. Side outputs: $RESULT_DIR/_v17_4_stage{1,2}.csv (stage1 == v16.4 cold;
# stage2 = 32B-AWQ hinted) + _v17_4_hint.json (the ref-index handoff).

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
# RESULT_DIR overridable (e.g. an order-diagnostic run into a separate dir):
#   sbatch --export=ALL,RESULT_DIR=...,TEXTSUM_SUBMIT_ORDER=original submit_eval_train_v17.4.sh
export RESULT_DIR="${RESULT_DIR:-$PROJECT/textsum/eval_train/result_v17.4}"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export MAX_MODEL_LEN="32768"
# gemma NVFP4 bakes an fp8-KV directive that run.py strips into a scratch
# override dir (build_kv_neutralized_model); give it a writable Lustre path.
export TEXTSUM_SCRATCH_DIR="$PROJECT/textsum/scratch_v17.4"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$TEXTSUM_SCRATCH_DIR" "$PROJECT/logs"

echo "=== v17.4 (32B-AWQ hinted answer + gemma-NVFP4 refs, coupled = exp86 ansB_refA) container run.py — train eval ==="
cd "$PROJECT/textsum/model"
python3 run.py

echo ""
echo "######## scoring (full 1239) ########"
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"
echo "######## leak-free (excl. doc_050) ########"
cd "$PROJECT/textsum/eval_train"
python3 score_heldout.py "$RESULT_DIR/submission.csv" doc_050
