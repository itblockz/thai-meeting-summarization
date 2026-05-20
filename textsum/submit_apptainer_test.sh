#!/bin/bash
#SBATCH --job-name=v6_debug
#SBATCH --partition=gpu
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v6_debug_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v6_debug_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

RESULT="$PROJECT/textsum_v6_debug_result"
mkdir -p "$RESULT" "$PROJECT/logs"

echo "=== apptainer version ==="
apptainer --version

echo "=== GPU visible ==="
nvidia-smi | head -10

echo "=== running container (exec python3 /model/run.py) ==="
apptainer exec --nv --containall --pwd /model \
    --bind /usr/share/zoneinfo:/usr/share/zoneinfo:ro \
    --bind "$PROJECT/textsum/model/test:/model/test:ro" \
    --bind "$PROJECT/textsum/benchmark_lib:/benchmark_lib:ro" \
    --bind "$RESULT:/result" \
    "$PROJECT/textsum_v6.sif" python3 /model/run.py

echo "=== exit code: $? ==="
echo "=== result dir contents ==="
ls -la "$RESULT"
