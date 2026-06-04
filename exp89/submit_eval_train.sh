#!/bin/bash
#SBATCH --job-name=exp89_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp89_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp89_eval_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/exp89/eval_result"
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
# kernel). See run.py's KV comment for the trap (exp77 lineage). The strip is
# a no-op if the checkpoint carries no kv directive. Rebuilt fresh each run.
SNAP=$(ls -d "$SHARED"/.hf_cache/hub/models--nvidia--Qwen3-32B-NVFP4/snapshots/*/ | head -1)
OVR="$PROJECT/exp89/model_override"
rm -rf "$OVR"; mkdir -p "$OVR"
for f in "$SNAP"*; do ln -s "$f" "$OVR/$(basename "$f")"; done
python3 - "$OVR" <<'PY'
import json, os, sys
d = sys.argv[1]
# hf_quant_config.json: neutralize the FP8 KV directive (if present)
p = os.path.join(d, "hf_quant_config.json")
if os.path.islink(p) or os.path.exists(p):
    j = json.load(open(p, encoding="utf-8"))
    j.get("quantization", {}).pop("kv_cache_quant_algo", None)
    os.remove(p)  # drop the symlink before writing a real file (keep blob intact)
    json.dump(j, open(p, "w", encoding="utf-8"), indent=2)
# config.json: drop any kv_cache_scheme / kv_cache_quant_algo in quantization_config
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

echo "=== exp89: Qwen3-32B-NVFP4 (nvidia/ModelOpt, dense, single A100-40GB) + exp51 prompt/shots ==="
cd "$PROJECT/exp89"
python3 run.py

echo "=== exp89: scoring (full 1239, headline — has doc_050 few-shot leak) ==="
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"

echo "=== exp89: leak-free scoring (hold out doc_050, 1218 queries) ==="
cd "$PROJECT/textsum/eval_train"
python3 score_heldout.py "$RESULT_DIR/submission.csv" doc_050
