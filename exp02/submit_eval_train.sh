#!/bin/bash
#SBATCH --job-name=exp02_eval
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --exclude=lanta-g-024,lanta-g-097
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp02_eval_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/exp02_eval_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
SHARED=/lustrefs/disk/project/zz991000-zdeva/zz991021

module purge
module load cray-python/3.11.7

source "$SHARED/venv/bin/activate"

_VENV_SITE="$SHARED/venv/lib/python3.11/site-packages"
_NVIDIA_LIBS=$(find "$_VENV_SITE/nvidia" -maxdepth 2 -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${_VENV_SITE}/torch/lib:${_NVIDIA_LIBS}${SHARED}/.cuda_stub${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export TEST_DIR="$PROJECT/textsum/eval_train"
export RESULT_DIR="$PROJECT/exp02/eval_result"
export PROGRESS_LIB="$PROJECT/textsum/benchmark_lib/progress"
export HF_HOME="$SHARED/.hf_cache"
export TRANSFORMERS_CACHE="$SHARED/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$PROJECT/exp02/.vllm_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$RESULT_DIR" "$PROJECT/logs" "$VLLM_CACHE_ROOT"

python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('CUDA not available', flush=True); sys.exit(1)
try:
    torch.zeros(1).cuda()
    print(f'CUDA OK: {torch.cuda.get_device_name(0)}', flush=True)
except RuntimeError as e:
    print(f'CUDA BROKEN on this node: {e}', flush=True); sys.exit(1)
" || { echo "Aborting: bad GPU node ($(hostname)). Re-submit to get a different node."; exit 1; }

echo "=== exp02 eval on train set (1,239 queries) ==="
cd "$PROJECT/exp02"
python3 run.py

echo "=== Computing evaluation metrics ==="
python3 "$PROJECT/textsum/eval_train/score.py" "$RESULT_DIR/submission.csv"
