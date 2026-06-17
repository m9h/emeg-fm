#!/usr/bin/env bash
#SBATCH --job-name=oneuro-inspect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# Stage + inspect ds002893 / ds004148 BIDS structure (CPU, no GPU) so the
# contrast loaders can be written correctly before the GPU reproduction run.
# Docker NGC 26.05 + openneuro-py on /mnt/t9 (never /data NFS).
set -euo pipefail
cd "$HOME/dev/emeg-fm"

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
REPRO_LIBS="$T9/repro_libs"
mkdir -p "$REPRO_LIBS" "$T9/openneuro"
HOST_UIDGID="$(id -u):$(id -g)"
PYPATH="$REPRO_LIBS:$T9/moabblibs"

exec docker run --rm \
  -v "$EMEG_FM:/emeg-fm" \
  -v "$T9:$T9" \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONPATH="$PYPATH" \
  -e ONEURO_DIR="$T9/openneuro" \
  -w /emeg-fm \
  "$IMAGE" \
  bash -c "
    set -e
    uv pip install --target '$REPRO_LIBS' openneuro-py >/tmp/on_install.log 2>&1 || \
      pip install --target='$REPRO_LIBS' openneuro-py >>/tmp/on_install.log 2>&1 || \
      echo '[warn] openneuro-py install failed'
    tail -n 2 /tmp/on_install.log || true
    python scripts/inspect_openneuro.py
    chown -R $HOST_UIDGID '$T9/openneuro' 2>/dev/null || true
  "
