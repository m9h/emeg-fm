#!/bin/bash
#SBATCH --job-name=pytorch-2605-sif
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/derivatives/containers/build_logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Build pytorch_26.05.sif from the locally-pulled NGC 26.05 docker image, to
# replace 26.04 as the Scope B (REVE-tiny Muon) base. The reve-overlay is
# rebuilt on top of THIS sif afterward (build_reve_sif.sbatch) so its
# protect-constraints pin against 26.05's exact torch, not 26.04's.
#
# Conversion path: docker save -> docker-archive:// -> apptainer build.
# `docker-daemon://` / `docker://` deadlock on large multi-layer NGC images
# (apptainer #1019); the image is already pulled so the save is a local no-op.
#
# DISK: apptainer extracts the full ~35 GB rootfs to APPTAINER_TMPDIR before
# mksquashfs, so the scratch dir must be LOCAL NVMe with ~45 GB headroom
# (NFS staging fails: APPTAINER_TMPDIR cannot live on /data). Root was freed to
# ~54 GB for this; if it ENOSPCs, remove another stale docker image first.

set -euo pipefail

SIF=/data/derivatives/containers/pytorch_26.05.sif
IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
# LOCAL NVMe scratch — /data is NFS and unusable for apptainer staging.
TMPDIR_APP=/home/mhough/.cache/apptainer-tmp
TARBALL="$TMPDIR_APP/pytorch-26.05.tar"

mkdir -p "$(dirname "$SIF")" /data/derivatives/containers/build_logs "$TMPDIR_APP"

if [[ -f "$SIF" ]]; then
    echo "$SIF already exists. Delete it to rebuild."
    ls -la "$SIF"
    exit 0
fi

echo "Free space on local NVMe before build:"
df -h / | tail -1

echo "Clearing apptainer scratch at $TMPDIR_APP …"
rm -rf "$TMPDIR_APP"/* 2>/dev/null || true

echo
echo "Step 1/3: docker save $IMAGE -> $TARBALL"
echo "  (sequential ~35 GB tar write to local NVMe)"
time docker save "$IMAGE" -o "$TARBALL"
ls -lh "$TARBALL"

echo
echo "Step 2/3: apptainer build SIF from docker-archive"
echo "  (extracts rootfs to TMPDIR then mksquashfs)"
export APPTAINER_TMPDIR="$TMPDIR_APP"
time apptainer build "$SIF" "docker-archive://$TARBALL"

# Reclaim the ~35 GB tarball immediately — disk is tight.
rm -f "$TARBALL"
ls -la "$SIF"

echo
echo "Step 3/3: verify torch inside the SIF (no GPU yet; GPU verify is separate)…"
apptainer exec --no-init "$SIF" python -c \
    "import torch; print('torch', torch.__version__); print('cuda built:', torch.version.cuda)"

echo "Done. SIF at $SIF"
