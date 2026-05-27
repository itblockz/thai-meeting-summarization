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

# v15-E = isolation hypothesis test. v15-D's gdb trace showed segfault at
# 0x5266a0 inside the stripped python3 binary (Thread 1), with 60+ threads
# alive incl. NCCL Watchdog/HeartbeatMonitor and Gloo TCP loops. Pattern =
# heap corruption (likely ABI/lib mismatch). Swap `--containall` → `--contain`
# so /tmp, /dev/shm, /etc come from host instead of the container's minimal
# defaults. If the segfault disappears, host-vs-container lib clash is the
# root cause. Final submission still needs --containall — this is diagnostic.
ulimit -c unlimited
echo "=== v15-E isolation test — --contain instead of --containall ==="
apptainer exec --nv --contain --pwd /model \
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
        -ex 'echo \n=== BACKTRACE (crashing thread) ===\n' \
        -ex 'bt' \
        -ex 'echo \n=== BACKTRACE (all threads) ===\n' \
        -ex 'thread apply all bt' \
        -ex 'echo \n=== INFO SHARED ===\n' \
        -ex 'info sharedlibrary' \
        --args python3 /model/run.py

echo "=== exit code: $? ==="
ls -la "$RESULT"
