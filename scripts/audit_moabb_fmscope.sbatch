#!/bin/bash
#SBATCH --job-name=moabb-fmscope
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# FMScope identity-trap audit of a MOABB BCI cohort (BNCI2014_001 left/right
# motor imagery) against frozen REVE. Runs inside the PyTorch 26.04 NGC SIF
# (CUDA + torch + transformers). moabb is NOT bundled in the SIF, so it is
# pip-installed to a persistent --target dir that the audit script appends to
# the END of sys.path — SIF numpy/scipy/torch keep precedence; only moabb (and
# its non-SIF deps: pyriemann/pooch/edfio/coloredlogs) resolve from the target.
# This is the sanctioned alternative to `pip install --user`, which would shadow
# the SIF torch stack.
#
# Usage:
#   sbatch scripts/audit_moabb_fmscope.sbatch
#   LAYER=8 sbatch scripts/audit_moabb_fmscope.sbatch
#
# Pre-req: accept the brain-bzh/reve-base Responsible Use Agreement on HF and
# have HF_TOKEN in ~/.cache/huggingface/token (REVE is gated).

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/emeg-fm
SIF=/data/derivatives/containers/pytorch_26.04.sif
HF_CACHE=/data/derivatives/eeg_sae/hf_cache
HF_TOKEN_FILE=${HF_HOME:-$HOME/.cache/huggingface}/token

# Node-local install + dataset cache. MUST NOT be on /data (NFS): pip unpacks
# ~14k small files for moabb's dep tree, and small-file writes to NFS wedge in
# uninterruptible D-state for hours (worse under concurrent /data load). /tmp is
# node-local — install is seconds, not hours. Re-installs per node, which is
# fine (cheap). Override with MOABB_LIBS/MNE_DATA env if needed.
MOABB_LIBS=${MOABB_LIBS:-/tmp/moabblibs}
MNE_DATA=${MNE_DATA:-/tmp/moabb_data}
LAYER=${LAYER:-6}
MODEL=${MODEL:-brain-bzh/reve-base}

mkdir -p /data/derivatives/eeg_sae/logs "$HF_CACHE" "$MOABB_LIBS" "$MNE_DATA"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: $SIF not found." >&2; exit 1
fi

# REVE is gated (brain-bzh/*) — token required.
HF_TOKEN=""
if [[ "$MODEL" == brain-bzh/* ]]; then
    if [[ ! -s "$HF_TOKEN_FILE" ]]; then
        echo "ERROR: HF token not at $HF_TOKEN_FILE — needed for REVE gated model." >&2
        exit 1
    fi
    HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

echo "Job:    audit_moabb_fmscope.sbatch"
echo "Model:  $MODEL  layer=$LAYER"
echo "Cohort: BNCI2014_001 (LeftRightImagery)"
echo "Out:    $PROJECT_ROOT/results/moabb_fmscope/BNCI2014_001-LR.json"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Host CUDA driver bind (harmless for REVE; mirrors the extract launcher so the
# SIF's stale compat symlink never trips Triton on shared infra).
LIBCUDA_BIND=()
HOST_LIBCUDA=$(readlink -f /usr/lib/aarch64-linux-gnu/libcuda.so.1 2>/dev/null || true)
if [[ -n "$HOST_LIBCUDA" && -e "$HOST_LIBCUDA" ]]; then
    LIBCUDA_BIND=( -B "$HOST_LIBCUDA":/usr/local/cuda/compat/lib/libcuda.so.1 )
fi

# --no-init: NGC SIF startup scripts hang (license/nvidia-cli probe); we pass
# every needed env explicitly. -B /data:/data: apptainer won't auto-bind NFS.
apptainer exec --nv --no-init \
    --env HF_TOKEN="$HF_TOKEN" \
    --env HF_HOME="$HF_CACHE" \
    --env PYTHONPATH="$PROJECT_ROOT" \
    --env MOABB_LIBS="$MOABB_LIBS" \
    --env MNE_DATA="$MNE_DATA" \
    -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
    -B /data:/data \
    -B "$HF_CACHE":"$HF_CACHE" \
    "${LIBCUDA_BIND[@]}" \
    "$SIF" bash -c '
        set -euo pipefail
        cd "$1"; shift
        # One-time moabb install into the --target dir (idempotent: skip if
        # already present). torch is not a moabb dep, so nothing here shadows
        # the SIF GPU stack.
        if [[ ! -d "$MOABB_LIBS/moabb" ]]; then
            echo "[setup] installing moabb -> $MOABB_LIBS"
            python -m pip install --no-cache-dir --target "$MOABB_LIBS" "moabb>=1.0"
        else
            echo "[setup] moabb present at $MOABB_LIBS"
        fi
        echo "[exec] $*"
        exec "$@"
    ' bash "$PROJECT_ROOT" \
        python scripts/audit_moabb_fmscope.py \
            --layer "$LAYER" --model "$MODEL" \
            --out-dir "$PROJECT_ROOT/results/moabb_fmscope"

unset HF_TOKEN
echo "[done] audit finished."
