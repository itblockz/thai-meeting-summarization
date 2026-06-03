#!/bin/bash
#SBATCH --job-name=exp80_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp80_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp80_eval_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/exp80/eval_result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export MAX_MODEL_LEN="32768"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$PROJECT/logs"

echo "=== exp80: S1=A3B -> S2=gemma (hint=ref only), emit 4 combos ==="
cd "$PROJECT/exp80"
python3 run.py

for combo in s1ans_s1ref s1ans_s2ref s2ans_s1ref s2ans_s2ref; do
  echo ""
  echo "######## combo $combo : scoring (full 1239) ########"
  python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/$combo/submission.csv"
  echo "######## combo $combo : leak-free (excl. doc_050) ########"
  cd "$PROJECT/textsum/eval_train"
  python3 score_heldout.py "$RESULT_DIR/$combo/submission.csv" doc_050
  cd "$PROJECT/exp80"
done
