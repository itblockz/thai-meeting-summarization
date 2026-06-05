#!/bin/bash
#SBATCH --job-name=v17_4_s1mnbt
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_4_s1mnbt_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v17_4_s1mnbt_%j.err

# PROBE: re-run ONLY Stage 1 (gemma NVFP4 refs) with a larger max_num_batched_
# tokens to test whether removing chunked prefill recovers the IoU drift vs
# exp77 (v17.4 stage1 leak-free IoU 0.8103 vs exp77 0.8165, 64/1239 refs differ).
# The full v17.4 run capped STAGE1_MNBT=8192 (v16.4 container value) → the ~14K-
# token doc prompt is split into 2 prefill chunks, changing the FP reduction
# order → a few low-confidence citation tokens flip. mnbt ≥ longest prompt = no
# chunking. Writes _v17_4_stage1.csv into a SEPARATE RESULT_DIR so the original
# run is untouched; score IoU offline (stdlib) afterwards.
#
# Override STAGE1_MNBT on submit:  sbatch --export=ALL,S1MNBT=16384 submit_stage1_mnbt_probe.sh

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7
source "$SHARED/venv/bin/activate"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/textsum/eval_train/result_v17.4_s1mnbt"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export MAX_MODEL_LEN="32768"
export TEXTSUM_SCRATCH_DIR="$PROJECT/textsum/scratch_v17.4"
export STAGE1_MNBT="${S1MNBT:-32768}"   # default: no chunked prefill at all

export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$TEXTSUM_SCRATCH_DIR" "$PROJECT/logs"

echo "=== v17.4 Stage-1-only probe — STAGE1_MNBT=$STAGE1_MNBT ==="
cd "$PROJECT/textsum/model"
python3 run.py --stage 1

echo ""
echo "######## stage1 refs IoU (stdlib, full + leak-free) ########"
cd "$PROJECT/textsum/eval_train"
python3 - "$RESULT_DIR/_v17_4_stage1.csv" <<'PY'
import json, csv, sys
gt={}
for q in json.load(open('test.json',encoding='utf-8'))['queries']:
    r=q.get('refs',[]); r=r if isinstance(r,list) else [r]
    gt[q['ID']]=(q['doc_id'], set(str(x).strip() for x in r if str(x).strip()))
def load(p):
    d={}
    for row in csv.DictReader(open(p,encoding='utf-8')):
        x=row['refs']; d[row['ID']]=set(i.strip() for i in str(x).split(',') if i.strip()) if x and x.strip() else set()
    return d
def iou(p,s): return 0.0 if not s else len(s&p)/len(s|p)
pred=load(sys.argv[1])
full=[iou(pred[i],gt[i][1]) for i in gt]
lf=[iou(pred[i],gt[i][1]) for i in gt if gt[i][0]!='doc_050']
print('full IoU=%.4f (n=%d)  leak-free IoU=%.4f (n=%d)'%(sum(full)/len(full),len(full),sum(lf)/len(lf),len(lf)))
print('avg refs/q = %.3f'%(sum(len(pred[i]) for i in gt)/len(gt)))
PY
