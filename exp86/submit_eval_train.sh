#!/bin/bash
#SBATCH --job-name=exp86_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp86_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp86_eval_%j.err

# exp86 — best-of-both two-stage grid:
#   Stage A = nvidia/Gemma-4-26B-A4B-NVFP4 (exp77 config-override) -> refs/hint
#   Stage B = Qwen3-32B-AWQ (exp38 E5 + hint)                      -> answer
# Emits the 4-combo grid (ansA_refA, ansA_refB, ansB_refA, ansB_refB) and
# scores each (full 1239 + leak-free 1218). Target cell = ansB_refA (~0.725).

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export PROJECT
export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/exp86/eval_result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export MAX_MODEL_LEN="32768"
export LLM_MODEL_STAGE_B="Qwen/Qwen3-32B-AWQ"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$PROJECT/logs"

# --- Build Stage-A override (strip FP8-KV directive; see exp77/run.py) -------
SNAP=$(ls -d "$SHARED"/.hf_cache/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/*/ | head -1)
OVR="$PROJECT/exp86/model_override"
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
export LLM_MODEL_STAGE_A="$OVR"

echo "=== exp86: two-stage grid (NVFP4 gemma refs -> 32B-AWQ answer) ==="
cd "$PROJECT/exp86"
python3 run.py

# --- Score every combo (full 1239 + leak-free, doc_050 held out) ------------
for c in ansA_refA ansA_refB ansB_refA ansB_refB; do
    echo ""
    echo "######## combo $c : scoring (full 1239) ########"
    python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/$c/submission.csv"
    echo "######## combo $c : leak-free (excl. doc_050) ########"
    (cd "$PROJECT/textsum/eval_train" && python3 score_heldout.py "$RESULT_DIR/$c/submission.csv" doc_050)
done
