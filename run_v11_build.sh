#!/bin/bash
PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
export APPTAINER_TMPDIR="$PROJECT/apptainer_tmp"
export APPTAINER_CACHEDIR="$PROJECT/apptainer_tmp/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$PROJECT/logs"
echo "=== v11 SIF build started: $(date) ==="
module load Apptainer/1.1.6
cd "$PROJECT/textsum"
apptainer build --force "$PROJECT/textsum_v11_local.sif" textsum.def
BUILD_RC=$?
echo "=== apptainer build exit code: $BUILD_RC ($(date)) ==="
if [ "$BUILD_RC" -eq 0 ]; then
    echo "=== build OK -> submitting container test (submit_apptainer_test_v11.sh) ==="
    sbatch "$PROJECT/textsum/submit_apptainer_test_v11.sh"
else
    echo "=== build FAILED -> container test NOT submitted ==="
fi
