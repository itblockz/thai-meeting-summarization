#!/bin/bash
#SBATCH --job-name=prompt_lab_r3cthk
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/prompt_lab_r3cthk_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/prompt_lab_r3cthk_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export HF_HOME="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LLM_MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
export RESULT_SUFFIX="r3_cite_only"
export MAX_TOKENS=64
export RANK_BY=IoU
export VARIANTS="R3C11_factual_one_sentence,R3C12_factual_minimal,R3C13_factual_extract,R3C14_named_entities,R3C15_no_redundant,R3C16_complete"

cd "$PROJECT/prompt_lab"
python3 runner.py
