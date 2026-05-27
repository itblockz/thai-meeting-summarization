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

# v15-F = python3-dbg diagnostic. v15-D/E confirmed the SIGSEGV is at the
# same deterministic address 0x5266a0 inside the stripped /usr/bin/python3
# binary (isolation ruled out). Rebuild adds python3.11-dbg so gdb can
# (a) resolve 0x5266a0 to a real CPython symbol, and (b) use py-bt to print
# the actual Python frame at the crash — this is the definitive trace.
ulimit -c unlimited
echo "=== v15-F py-bt diagnostic — --containall + python3.11-dbg ==="
apptainer exec --nv --containall --pwd /model \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    --env MAX_MODEL_LEN=32768 \
    --env PYTHONFAULTHANDLER=1 \
    "$PROJECT/textsum_v15_local.sif" \
    gdb -batch \
        -ex 'set pagination off' \
        -ex 'set print thread-events off' \
        -ex 'handle SIGSEGV stop print nopass' \
        -ex 'run' \
        -ex 'echo \n=== INFO SYMBOL @ crash address ===\n' \
        -ex 'info symbol 0x5266a0' \
        -ex 'echo \n=== DISASSEMBLY around crash ===\n' \
        -ex 'x/16i $pc' \
        -ex 'echo \n=== REGISTERS ===\n' \
        -ex 'info registers' \
        -ex 'echo \n=== BACKTRACE (C, crashing thread) ===\n' \
        -ex 'bt full' \
        -ex 'echo \n=== PY-BT (Python frame at crash) ===\n' \
        -ex 'py-bt' \
        -ex 'echo \n=== PY-LIST (source around current Python frame) ===\n' \
        -ex 'py-list' \
        -ex 'echo \n=== PY-LOCALS ===\n' \
        -ex 'py-locals' \
        -ex 'echo \n=== BACKTRACE (C, all threads) ===\n' \
        -ex 'thread apply all bt' \
        --args python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
