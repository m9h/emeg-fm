#!/usr/bin/env bash
# (Re)build /mnt/t9/eegfm_libs_2606 — the braindecode/timm/weightwatcher/torchaudio
# layer for the NGC PyTorch 26.06 EEG-FM container (Tier-0 of the NeuroTechX Container
# Center). The container supplies torch 2.13 / CUDA 13.3 (GB10) + tokenizers +
# safetensors; /mnt/t9/tokfix + /mnt/t9/moabblibs already exist (version-portable from
# 26.05 and reused). Runs inside the container via uv, NOT ~/.local (whose
# torch 2.12+cpu would shadow the GPU torch). Idempotent.
#
# After this, run any model with scripts/eegfm_t9.sh.
set -euo pipefail
IMAGE="${EEGFM_IMAGE:-nvcr.io/nvidia/pytorch:26.06-py3}"
T9="${T9:-/mnt/t9}"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
TARGET="$T9/eegfm_libs_2606"

docker run --rm \
  -v "$T9:$T9" -v "$EMEG_FM:/emeg-fm" -v "$HOME/.local/bin/uv:/usr/local/bin/uv:ro" \
  -e UV_CACHE_DIR="$T9/uv-cache" -e PYTHONNOUSERSITE=1 "$IMAGE" bash -c "
    set -e
    uv pip install --target '$TARGET' braindecode timm weightwatcher eegdash
    uv pip install --target '$TARGET' --no-deps torchaudio
    cd '$TARGET'
    # numeric/torch stack resolves from the container — strip target copies so they
    # cannot shadow the GPU torch (KEEP torchaudio: braindecode imports it, and its
    # CUDA 13.0 build is minor-compatible with the 13.3 runtime).
    rm -rf torch torch-* functorch torchvision torchvision-* torchgen torio \
           nvidia* triton* numpy numpy-* numpy.libs scipy scipy-* scipy.libs
    python /emeg-fm/scripts/_patch_torchaudio_cudacheck.py '$TARGET'
  "
echo "[done] $TARGET built — run models with scripts/eegfm_t9.sh"
