#!/usr/bin/env bash
#SBATCH --job-name=nf-stage
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# No --time: gx10-dgx-spark single-user / infinite walltime.
#
# Stage a dataset via Meta neuralfetch inside the Docker NGC 26.05 container.
# neuralfetch + its fetch backends are installed to a --target dir on /mnt/t9
# (NEVER pip --user -- that shadows the SIF/container stack), PYTHONNOUSERSITE=1.
# Output goes to /mnt/t9 (NEVER /data NFS; thousands of small BIDS files wedge it).
# CPU only (download + BIDS conversion; no GPU).
#
# Usage:
#   sbatch scripts/neuralfetch_stage.sbatch --coverage
#   sbatch scripts/neuralfetch_stage.sbatch --study Obeid2016Tueg \
#          --query '{"target":"age"}' --out /mnt/t9/tueg
#
# Gated sources (TUH DUA, etc.) still need credentials present in the container
# env -- neuralfetch wraps the download, it does not bypass auth.
set -euo pipefail
cd "$HOME/dev/emeg-fm"

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
NF_LIBS="$T9/neuralfetch_libs"          # isolated --target install (persists across runs)
HOST_UIDGID="$(id -u):$(id -g)"
mkdir -p "$NF_LIBS" "$T9/neurofetch"

exec docker run --rm \
  -v "$EMEG_FM:/emeg-fm" -v "$T9:$T9" \
  -e PYTHONNOUSERSITE=1 -e PYTHONPATH="$NF_LIBS:/emeg-fm" \
  -e HF_HOME="$T9/hf" -e MNE_DATA="$T9/moabb_data" \
  -w /emeg-fm "$IMAGE" \
  bash -c "
    set -e
    # Idempotent isolated install (skip if neuralfetch already present in target).
    python -c 'import neuralfetch' 2>/dev/null || \
      pip install --no-input --target '$NF_LIBS' 'neuralfetch[quickstart]'
    python scripts/neuralfetch_stage.py $* ;
    rc=\$?; chown -R $HOST_UIDGID '$NF_LIBS' '$T9/neurofetch' 2>/dev/null || true; exit \$rc
  "
