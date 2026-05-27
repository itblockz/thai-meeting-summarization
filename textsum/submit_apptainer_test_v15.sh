#!/bin/bash
#SBATCH --job-name=v15_test
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v15_test_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v15_test_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v15_test_result"
mkdir -p "$RESULT" "$PROJECT/logs"

# v15-G = fix. v15-F's py-bt identified the bug: vllm 0.19.1's V1 sampler
# top-k/top-p Triton kernel JIT-compiles during _dummy_sampler_run and
# crashes inside Triton's C++ IRBuilder (pybind11/CPython ABI mismatch in
# this container — `self.builder.options = options` hits
# _PyDictKeys_StringLookup with NULL dk). run.py now sets
# sys.modules["triton"]=None before importing vllm so HAS_TRITON=False
# and the sampler falls back to pure-PyTorch (Qwen3 dense AWQ doesn't
# touch any other Triton path).
#
# Bind-mounts run.py over the container's /model/run.py so we can iterate
# without rebuilding the SIF.
echo "=== v15-H surgical sampler-only Triton bypass test ==="
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/run.py:/model/run.py:ro" \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env MAX_MODEL_LEN=32768 \
    "$PROJECT/textsum_v15_local.sif" python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
