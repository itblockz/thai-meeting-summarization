#!/bin/bash
#SBATCH --job-name=plab_gemma_iou
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/plab_gemma_iou_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/plab_gemma_iou_%j.err

# Prompt-lab Round 8: exp77 IoU failure-mode fixes on the actual ref-picker model
# (nvidia/Gemma-4-26B-A4B-NVFP4). Baseline V10_factual = exp77's deployed recipe.
PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export HF_HOME="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Build a local override of the nvidia/ModelOpt checkpoint: symlink the snapshot
# but strip the FP8 KV directive from the configs so vLLM's "auto" kv_cache_dtype
# resolves to bf16 (sm80 has no fp8e4nv reshape_and_cache kernel; the TRITON_ATTN
# MoE path also rejects an explicit "bfloat16"). Same trap/fix as exp77. The
# override basename = real model name so runner's MODEL_TAG stays clean.
SNAP=$(ls -d "$SHARED"/.hf_cache/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/*/ | head -1)
OVR="$PROJECT/prompt_lab/override/Gemma-4-26B-A4B-NVFP4"
rm -rf "$OVR"; mkdir -p "$OVR"
for f in "$SNAP"*; do ln -s "$f" "$OVR/$(basename "$f")"; done
python3 - "$OVR" <<'PY'
import json, os, sys
d = sys.argv[1]
p = os.path.join(d, "hf_quant_config.json")
j = json.load(open(p, encoding="utf-8"))
j.get("quantization", {}).pop("kv_cache_quant_algo", None)
os.remove(p)
json.dump(j, open(p, "w", encoding="utf-8"), indent=2)
p = os.path.join(d, "config.json")
j = json.load(open(p, encoding="utf-8"))
qc = j.get("quantization_config", {})
qc.pop("kv_cache_scheme", None)
qc.pop("kv_cache_quant_algo", None)
os.remove(p)
json.dump(j, open(p, "w", encoding="utf-8"), indent=2)
print("override built at", d)
PY

export LLM_MODEL="$OVR"
export GPU_MEM_UTIL="0.90"     # 0.95 OOMs the sampling buffer on the ~18 GiB gemma MoE
export RESULT_SUFFIX="iou"
export RANK_BY="iou"
# baseline (= exp77 recipe) + Round 8 fixes
export VARIANTS="V10_factual,C2_cite_direct_all,C4_cite_no_heading,G1_factual_block,G2_factual_outcome,G3_factual_block_outcome,G4_cite_block,G5_cite_outcome,G6_cite_block_outcome"

echo "=== prompt_lab Round 8: gemma NVFP4 IoU fixes (util=$GPU_MEM_UTIL) ==="
cd "$PROJECT/prompt_lab"
python3 runner.py
