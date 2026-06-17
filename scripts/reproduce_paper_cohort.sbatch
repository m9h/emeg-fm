#!/usr/bin/env bash
#SBATCH --job-name=paper-repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# From-raw reproduction of an FMScope paper generalization cohort (default
# ds004362 motor imagery). Docker NGC PyTorch 26.05 + uv on /mnt/t9 (never /data
# NFS / NFS SIF). REVE GPU extraction; preprocess_v2 deps (mne_icalabel,
# autoreject, pooch) uv-installed --no-deps into a persistent /mnt/t9 target so
# they borrow mne/sklearn/scipy from moabblibs and torch from the NGC image.
# Both ICLabel and autoreject degrade gracefully (logged) if install fails.
#
# Usage: sbatch scripts/reproduce_paper_cohort.sbatch [--n-subjects 60 ...]
set -euo pipefail
cd "$HOME/dev/emeg-fm"

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
OUT_SUB="${OUT_SUB:-results/paper_repro}"
REPRO_LIBS="$T9/repro_libs"

mkdir -p "$EMEG_FM/$OUT_SUB" "$REPRO_LIBS" "$T9/mne_data" "$T9/openneuro"
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
# tokfix (hub) > our code > repro_libs (icalabel/autoreject/pooch) > moabblibs
# (mne/sklearn/scipy/joblib/numpy) > fooof_libs.
PYPATH="$T9/tokfix:/emeg-fm:/emeg-fm/fmscope:$REPRO_LIBS:$T9/moabblibs:$T9/fooof_libs"
HOST_UIDGID="$(id -u):$(id -g)"

exec docker run --rm --gpus all --ipc=host \
  -v "$EMEG_FM:/emeg-fm" \
  -v "$T9:$T9" \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONPATH="$PYPATH" \
  -e HF_HOME="$T9/hf" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MNE_DATA="$T9/mne_data" \
  -e ONEURO_DIR="$T9/openneuro" \
  -w /emeg-fm \
  "$IMAGE" \
  bash -c "
    set -e
    # Best-effort install of preprocess_v2 deps into the persistent target
    # (--no-deps so they don't drag a conflicting torch/mne over the NGC stack).
    uv pip install --target '$REPRO_LIBS' --no-deps mne-icalabel autoreject pooch h5io \
        >/tmp/repro_install.log 2>&1 || \
      pip install --target='$REPRO_LIBS' --no-deps mne-icalabel autoreject pooch h5io \
        >>/tmp/repro_install.log 2>&1 || \
      echo '[warn] repro-libs install failed; preprocessing will degrade (logged)'
    # openneuro-py needs its deps (httpx/tqdm/...) -> install WITH deps.
    uv pip install --target '$REPRO_LIBS' openneuro-py >>/tmp/repro_install.log 2>&1 || \
      pip install --target='$REPRO_LIBS' openneuro-py >>/tmp/repro_install.log 2>&1 || \
      echo '[warn] openneuro-py install failed (needed for ds002893/ds004148)'
    tail -n 3 /tmp/repro_install.log || true
    python scripts/reproduce_paper_cohort.py --out-dir '/emeg-fm/$OUT_SUB' $*
    rc=\$?
    chown -R $HOST_UIDGID '/emeg-fm/$OUT_SUB'
    exit \$rc
  "
