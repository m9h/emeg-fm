#!/usr/bin/env bash
# Time-series foundation-model brain-age, same cohort/CV as coffeine/NEOBA/REVE.
#
# Runs scripts/ts_fm_brain_age.py inside the Docker NGC PyTorch 26.05 container on
# /mnt/t9 (local SSD) — never /data NFS. The container supplies torch (GPU);
# /mnt/t9/moabblibs supplies mne/scikit-learn/pandas/scipy; the TS-FM package
# (momentfm, ...) is pip-installed once into /mnt/t9/tsfmlibs and cached.
#
# Usage:
#   scripts/ts_fm_brain_age_t9.sh --model moment \
#       --epochs-glob '/mnt/t9/brain_age_deriv/TDBRAIN_EEG/**/*proc-autoreject_epo.fif' \
#       --participants /mnt/t9/tdbrain_v3.1_full/TDBRAIN_Dataset_V3_1/participants.tsv \
#       --subject-regex 'sub-([A-Za-z0-9]+)' --seed 42 \
#       --out /mnt/t9/moment_tdbrain_emb.npz
set -euo pipefail

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
PYPATH="/emeg-fm:/emeg-fm/fmscope:$T9/moabblibs"

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
  bash -c '
    set -e
    TSLIBS=/mnt/t9/tsfmlibs
    export PYTHONPATH="$TSLIBS:$PYTHONPATH"
    # Install with --no-deps: the heavy runtime deps (torch/transformers/numpy/
    # huggingface_hub) already live in the NGC container. Without it, momentfm et al.
    # try to BUILD an ancient pinned numpy from source, which fails on py3.12
    # (pkgutil.ImpImporter removed). einops is the one pure-python extra momentfm needs.
    for spec in momentfm:momentfm mantis:mantis-tsfm chronos:chronos-forecasting; do
      imp=${spec%%:*}; pkg=${spec##*:}
      python -c "import $imp" 2>/dev/null || pip install -q --no-deps --target "$TSLIBS" "$pkg"
    done
    python -c "import einops" 2>/dev/null || pip install -q --no-deps --target "$TSLIBS" einops
    exec python scripts/ts_fm_brain_age.py "$@"
  ' _ "$@"
