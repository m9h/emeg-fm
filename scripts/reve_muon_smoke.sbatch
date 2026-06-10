#!/bin/bash
#SBATCH --job-name=reve-muon-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Scope B GPU smoke: REVE-tiny MAE + MuonHybrid end-to-end inside the REVE env
# (base pytorch SIF + reve-overlay). Validates the hybrid against REVE's real
# param shapes, the trapezoid-scheduler x lr_mult interaction, and a finite
# decreasing loss. No data needed (dummy batch).

set -euo pipefail

BASE_SIF=/data/derivatives/containers/pytorch_26.05.sif
OVERLAY=/data/derivatives/containers/reve-overlay-2605.img
REVE=/home/mhough/dev/reve_eeg

mkdir -p /data/derivatives/eeg_sae/logs
[[ -f "$BASE_SIF" ]] || { echo "ERROR: $BASE_SIF missing" >&2; exit 1; }
[[ -f "$OVERLAY"  ]] || { echo "ERROR: $OVERLAY missing"  >&2; exit 1; }

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Proven GPU/SIF recipe (see reve_lora_bifactor.sbatch): --nv --no-init, bind
# /data (not auto-bound), and bind the host libcuda over the stale compat path.
# PYTHONNOUSERSITE=1: Scope B uses the overlay's transformers/accelerate/peft,
# NOT ~/.local (which has a shadowing transformers 5.8.0 + jax).
LIBCUDA_BIND=()
HOST_LIBCUDA=$(readlink -f /usr/lib/aarch64-linux-gnu/libcuda.so.1 2>/dev/null || true)
if [[ -n "$HOST_LIBCUDA" && -e "$HOST_LIBCUDA" ]]; then
    LIBCUDA_BIND=( -B "$HOST_LIBCUDA":/usr/local/cuda/compat/lib/libcuda.so.1 )
fi

apptainer exec --nv --no-init \
    --overlay "$OVERLAY":ro \
    --env PYTHONNOUSERSITE=1 \
    -B /data:/data \
    -B "$REVE":"$REVE" \
    "${LIBCUDA_BIND[@]}" \
    "$BASE_SIF" python "$REVE/scope_b_gpu_smoke.py"

echo "[done] Scope B GPU smoke complete."
