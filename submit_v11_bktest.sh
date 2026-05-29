#!/bin/bash
#SBATCH --job-name=v11_bktest
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v11_bktest_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v11_bktest_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
module purge
module load Apptainer/1.1.6
mkdir -p "$PROJECT/logs"

for BK in FLASHINFER TRITON_ATTN FLEX_ATTENTION; do
    echo ""
    echo "############ VLLM_ATTENTION_BACKEND=$BK ############"
    RESULT="$PROJECT/textsum_v11_bk_$BK"
    mkdir -p "$RESULT"
    apptainer exec --nv --containall --pwd /model \
        --bind "$PROJECT/textsum/model/test:/model/test:ro" \
        --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
        --bind "$RESULT:/result" \
        --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
        --env VLLM_ATTENTION_BACKEND=$BK \
        "$PROJECT/textsum_v11_local.sif" python3 /model/run.py
    RC=$?
    echo "############ $BK exit code: $RC ############"
    if [ $RC -eq 0 ]; then
        echo "############ $BK WORKS -> rows: $(wc -l < "$RESULT/submission.csv" 2>/dev/null) ############"
    fi
done
echo "=== all backends tried ==="
