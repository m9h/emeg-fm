#!/usr/bin/env bash
# Phase 2: REVE-embed natview per-TR windows inside the NGC PyTorch container on /mnt/t9.
# Mirrors reve_brain_age_t9.sh (container torch/transformers + /mnt/t9/moabblibs deps +
# tokfix + emeg-fm REVEAdapter/ReveInputNorm; HF gated token for brain-bzh/reve-base).
# Usage: scripts/natview_reve_embed_t9.sh
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:26.05-py3}"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
PYPATH="$T9/tokfix:/emeg-fm:/emeg-fm/fmscope:$T9/moabblibs"

exec docker run --rm --gpus all \
  -v "$EMEG_FM:/emeg-fm" \
  -v "$T9:$T9" \
  -v "/data:/data:ro" \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONPATH="$PYPATH" \
  -e HF_HOME="$T9/hf" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e REVE_LAYER="${REVE_LAYER:-6}" \
  -w /emeg-fm \
  "$IMAGE" \
  python scripts/natview_reve_embed.py "$@"
