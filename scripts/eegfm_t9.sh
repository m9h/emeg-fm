#!/usr/bin/env bash
# General EEG-FM runner on the up-to-date NGC PyTorch 26.06 container — Tier-0 base
# for the NeuroTechX Container Center. Validated 2026-06-30 on the GB10: torch 2.13
# / CUDA 13.3 + transformers 5.12 + braindecode 1.5.2 + torchaudio (GPU fftconvolve)
# + the REVE adapter all load.
#
# Dep layers (PYTHONPATH order matters — tokfix first for the tokenizers pin, the
# numeric/torch stack always resolves from the container, never ~/.local):
#   container               torch 2.13 / CUDA 13.3 (GB10), numpy/scipy/torchvision
#   /mnt/t9/tokfix          tokenizers 0.22.2 pin (transformers<=0.23 compat) + hub
#   /mnt/t9/moabblibs       mne 1.12.1 / moabb / transformers 5.12 / matplotlib
#   /mnt/t9/eegfm_libs_2606 braindecode / timm / weightwatcher / skorch / eegdash /
#                           torchaudio (CUDA-minor-version check patched)
# Build the last layer with scripts/build_eegfm_libs_2606.sh.
#
# Usage:
#   scripts/eegfm_t9.sh python scripts/analyze_eegfm_weightwatcher.py --models cbramod
#   scripts/eegfm_t9.sh python -c "import braindecode; print(braindecode.__version__)"
set -euo pipefail

IMAGE="${EEGFM_IMAGE:-nvcr.io/nvidia/pytorch:26.06-py3}"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"

# HF token (gated REVE/Brant) — runtime env only, never written to disk here.
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"

PYPATH="$T9/tokfix:/emeg-fm:/emeg-fm/fmscope:$T9/moabblibs:$T9/eegfm_libs_2606"

exec docker run --rm --gpus all --ipc=host \
  -v "$EMEG_FM:/emeg-fm" \
  -v "$T9:$T9" \
  -v "/data:/data:ro" \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONPATH="$PYPATH" \
  -e HF_HOME="$T9/hf" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MNE_DATA="$T9/moabb_data" \
  -w /emeg-fm \
  "$IMAGE" "$@"
