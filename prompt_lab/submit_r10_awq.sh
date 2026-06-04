#!/bin/bash
#SBATCH --job-name=prompt_lab_r10_awq
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/prompt_lab_r10_awq_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/prompt_lab_r10_awq_%j.err

# Round 10 — answer-quality / length calibration (rank by 0.35·RougeL + 0.45·SS).
# Now computes the REAL SS-score via bge-m3 in a second phase (del llm first).
PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export HF_HOME="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LLM_MODEL="Qwen/Qwen3-32B-AWQ"
export RESULT_SUFFIX="r10"
export RANK_BY="answer_sub"
export VARIANTS="B_v10_factual,B_e5_baseline,N1_precise,N2_no_padding,N3_complete_concise,N4_length_band"

cd "$PROJECT/prompt_lab"
python3 runner.py
