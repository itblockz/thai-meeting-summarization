#!/bin/bash
#SBATCH --job-name=v8_build
#SBATCH --partition=compute
#SBATCH --account=zz991021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v8_build_%j.out
#SBATCH --error=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047/logs/v8_build_%j.err

PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047

module purge
module load Apptainer/1.1.6

# Both must point at Lustre — /tmp on the node is too small to unpack
# the ~30 GB image during build.
export APPTAINER_TMPDIR="$PROJECT/apptainer_tmp"
export APPTAINER_CACHEDIR="$PROJECT/apptainer_tmp/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$PROJECT/logs"

cd "$PROJECT/textsum"
apptainer build --force "$PROJECT/textsum_v8_local.sif" textsum.def
