#!/bin/bash
#SBATCH --job-name=exp51_cap_sweep
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp51_cap_sweep_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp51_cap_sweep_%j.err

# Post-hoc length-cap sweep on exp51's stored answers — NO LLM re-run.
# Only needs bge-m3 (SS-score) on one GPU; truncation + RougeL + IoU are CPU.
PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT/exp51"
# arg1: submission (default exp51/eval_result/submission.csv); arg2: heldout doc (default doc_050)
python3 cap_sweep.py "$PROJECT/exp51/eval_result/submission.csv" doc_050
