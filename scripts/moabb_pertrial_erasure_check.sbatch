#!/usr/bin/env bash
#SBATCH --job-name=moabb-pertrial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# Decisive per-trial vs recording-pooled subject-axis erasure on the
# BNCI2014_001 LeftRightImagery PoC, inside the Docker NGC PyTorch 26.05
# container with uv-installed deps on /mnt/t9 (never /data NFS / NFS SIF).
# Mirrors run_fmscope_leaderboard.sh's container env so the cohort builder,
# REVE adapter, and erasure resolve identically to the leaderboard.
set -euo pipefail
cd "$HOME/dev/emeg-fm"

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
MOABB_FORK="${MOABB_FORK:-$HOME/dev/moabb}"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
OUT_SUB="${OUT_SUB:-results/moabb_fmscope}"

mkdir -p "$EMEG_FM/$OUT_SUB"
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
PYPATH="$T9/tokfix:/moabb:/emeg-fm:/emeg-fm/fmscope:$T9/moabblibs:$T9/fooof_libs"
HOST_UIDGID="$(id -u):$(id -g)"

exec docker run --rm --gpus all --ipc=host \
  -v "$MOABB_FORK:/moabb" \
  -v "$EMEG_FM:/emeg-fm" \
  -v "$T9:$T9" \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONPATH="$PYPATH" \
  -e HF_HOME="$T9/hf" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MNE_DATA="$T9/moabb_data" \
  -w /emeg-fm \
  "$IMAGE" \
  bash -c "python scripts/moabb_pertrial_erasure_check.py --out-dir '/emeg-fm/$OUT_SUB' $* ; \
           rc=\$?; chown -R $HOST_UIDGID '/emeg-fm/$OUT_SUB'; exit \$rc"
