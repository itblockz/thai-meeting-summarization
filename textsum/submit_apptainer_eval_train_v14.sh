#!/bin/bash
#SBATCH --job-name=v14_train_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v14_train_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v14_train_eval_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v14_train_eval_result"
mkdir -p "$RESULT" "$PROJECT/logs"

# Run v14 container against train set (1239 queries) — measures the
# vllm-0.9.2 container's actual score, since exp37's 0.6944 was on the
# vllm-0.19.1 venv and v14 container drifted 33/50 vs venv on test set.
echo "=== v14 container — train eval (1239 queries via Apptainer) ==="
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/eval_train:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env MAX_MODEL_LEN=32768 \
    "$PROJECT/textsum_v14_local.sif" python3 /model/run.py

echo "=== container exit: $? ==="
ls -la "$RESULT"

# Score with venv (score.py needs pythainlp + bge-m3, container only has vllm).
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"
export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Score full 1239 (has doc_050 few-shot leak) ==="
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT/submission.csv"

echo "=== Score leak-free (hold out doc_050, 1218 queries — apples-to-apples vs exp37 0.6944) ==="
cd "$PROJECT/textsum/eval_train"
python3 score_heldout.py "$RESULT/submission.csv" doc_050
