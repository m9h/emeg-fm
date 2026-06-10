#!/bin/bash
#SBATCH --job-name=reve-tiny-loader
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Build REVE's real train/val loaders on the downloaded tiny_open subset and
# pull one batch from each. No GPU, no model — just the data path. Runs inside
# the 26.05 base SIF + reve overlay (has numpy/scipy/hydra/omegaconf/torch).

set -euo pipefail

BASE_SIF=/data/derivatives/containers/pytorch_26.05.sif
OVERLAY=/data/derivatives/containers/reve-overlay-2605.img
REVE=/home/mhough/dev/reve_eeg
ROOT=/data/derivatives/reve_dataset/tiny_open

[[ -f "$BASE_SIF" ]] || { echo "ERROR: $BASE_SIF missing" >&2; exit 1; }
[[ -f "$OVERLAY"  ]] || { echo "ERROR: $OVERLAY missing"  >&2; exit 1; }
[[ -d "$ROOT/recordings" ]] || { echo "ERROR: $ROOT not laid out — run download_reve_tiny_open.sbatch first" >&2; exit 1; }

mkdir -p /data/derivatives/eeg_sae/logs

# PYTHONNOUSERSITE=1: use the overlay stack, not ~/.local. No --nv (CPU only).
apptainer exec --no-init \
    --overlay "$OVERLAY":ro \
    --env PYTHONNOUSERSITE=1 \
    --env REVE_TINY_ROOT="$ROOT" \
    -B /data:/data \
    -B "$REVE":"$REVE" \
    "$BASE_SIF" python "$REVE/verify_tiny_loader.py"

echo "[done] tiny_open loader verification complete."
