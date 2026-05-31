#!/bin/bash
#SBATCH --job-name=prompt_lab_cite
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/prompt_lab_cite_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/prompt_lab_cite_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export HF_HOME="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
export RESULT_SUFFIX="cite_only"
export MAX_TOKENS=64
export RANK_BY=IoU
export VARIANTS="C1_cite_minimal,C2_cite_direct_all,C3_cite_with_heading,C4_cite_no_heading,C5_cite_key_phrases,C6_cite_silent_answer,C7_cite_balanced_3shot,C8_cite_conservative_3shot"

cd "$PROJECT/prompt_lab"
python3 runner.py
