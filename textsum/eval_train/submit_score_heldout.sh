#!/bin/bash
#SBATCH --job-name=score_heldout
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/score_heldout_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/score_heldout_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# score_heldout.py imports score.py from its own directory.
cd "$PROJECT/textsum/eval_train"

echo "=== leak-free re-score (hold out doc_050) ==="
for e in exp03 exp17 exp18 exp19 exp20 exp21; do
    sub="$PROJECT/$e/eval_result/submission.csv"
    echo "----- $e -----"
    if [ -f "$sub" ]; then
        python3 score_heldout.py "$sub" doc_050
    else
        echo "MISSING: $sub"
    fi
    echo
done
