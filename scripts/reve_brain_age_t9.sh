#!/usr/bin/env bash
# REVE foundation-model brain-age on LEMON, same cohort/CV as NEOBA + coffeine.
#
# Runs scripts/reve_brain_age.py inside the Docker NGC PyTorch 26.05 container on
# /mnt/t9 (the local SSD) — never /data NFS or the NFS apptainer SIF. The
# container supplies torch/transformers (GPU); /mnt/t9/moabblibs supplies
# mne/scikit-learn/pandas; /mnt/t9/tokfix pins tokenizers/hub; emeg-fm provides
# REVEAdapter + ReveInputNorm. LEMON epochs are pre-staged at /mnt/t9/lemon_epo.
#
# Usage:
#   scripts/reve_brain_age_t9.sh [extra args for reve_brain_age.py]
# Examples:
#   scripts/reve_brain_age_t9.sh                       # all 120 staged subjects, layer 6
#   scripts/reve_brain_age_t9.sh --layer 6 --max-subjects 20
set -euo pipefail

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"

# HF gated token (REVE) — read from the user's cache, passed only as a runtime
# env var (never written to disk here).
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"

# emeg_fm first so REVEAdapter/ReveInputNorm resolve; moabblibs supplies deps.
PYPATH="$T9/tokfix:/emeg-fm:/emeg-fm/fmscope:$T9/moabblibs"

exec docker run --rm --gpus all \
  -v "$EMEG_FM:/emeg-fm" \
  -v "$T9:$T9" \
  -v "/data:/data:ro" \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONPATH="$PYPATH" \
  -e HF_HOME="$T9/hf" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MNE_DATA="$T9/moabb_data" \
  -w /emeg-fm \
  "$IMAGE" \
  python scripts/reve_brain_age.py "$@"
