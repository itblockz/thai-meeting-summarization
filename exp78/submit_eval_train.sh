#!/bin/bash
#SBATCH --job-name=exp78_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp78_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp78_eval_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/exp78/eval_result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export MAX_MODEL_LEN="32768"
export TP_SIZE="1"

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$PROJECT/logs"

# Build a local override of the nvidia/ModelOpt checkpoint: symlink the
# snapshot but strip the FP8 KV directive from the configs so vLLM's "auto"
# kv_cache_dtype resolves to bf16 (sm80 has no fp8e4nv reshape_and_cache
# kernel — the exp71/72/75/77 wall). Backend-agnostic; a no-op if the Qwen
# build carries no kv directive. See run.py's KV comment. Rebuilt fresh each
# run. The python step pops the directive only if present (.pop default None).
SNAP=$(ls -d "$SHARED"/.hf_cache/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/*/ | head -1)
OVR="$PROJECT/exp78/model_override"
rm -rf "$OVR"; mkdir -p "$OVR"
for f in "$SNAP"*; do ln -s "$f" "$OVR/$(basename "$f")"; done
python3 - "$OVR" <<'PY'
import json, os, sys
d = sys.argv[1]
changed = []
# hf_quant_config.json: neutralize the FP8 KV directive (if present)
p = os.path.join(d, "hf_quant_config.json")
if os.path.exists(p):
    j = json.load(open(p, encoding="utf-8"))
    if j.get("quantization", {}).pop("kv_cache_quant_algo", None) is not None:
        changed.append("hf_quant_config.kv_cache_quant_algo")
    os.remove(p)  # drop the symlink before writing a real file (keep blob intact)
    json.dump(j, open(p, "w", encoding="utf-8"), indent=2)
# config.json: drop any kv_cache_scheme / kv_cache_quant_algo in quantization_config
p = os.path.join(d, "config.json")
j = json.load(open(p, encoding="utf-8"))
qc = j.get("quantization_config", {})
if qc.pop("kv_cache_scheme", None) is not None:
    changed.append("config.kv_cache_scheme")
if qc.pop("kv_cache_quant_algo", None) is not None:
    changed.append("config.kv_cache_quant_algo")
os.remove(p)
json.dump(j, open(p, "w", encoding="utf-8"), indent=2)
print("override built at", d, "| stripped:", changed or "nothing (no kv directive baked in)")
PY
export LLM_MODEL="$OVR"

echo "=== exp78: Qwen3.6-35B-A3B-NVFP4 (nvidia/ModelOpt, single A100-40GB) + exp51 prompt/shots ==="
cd "$PROJECT/exp78"
python3 run.py

echo "=== exp78: scoring (full 1239, headline — has doc_050 few-shot leak) ==="
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"

echo "=== exp78: leak-free scoring (hold out doc_050, 1218 queries) ==="
cd "$PROJECT/textsum/eval_train"
python3 score_heldout.py "$RESULT_DIR/submission.csv" doc_050
