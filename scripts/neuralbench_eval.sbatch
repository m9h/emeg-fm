#!/usr/bin/env bash
#SBATCH --job-name=nb-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
# No --time: gx10-dgx-spark single-user / infinite walltime.
#
# STUB — Meta neuralbench eval harness, in the Docker NGC 26.05 container.
# neuralbench's canonical eval (e.g. tasks/eeg/psychopathology/config.yaml IS the
# BrainCapture HBN bifactor eval, R5 held-out) replaces our hand-rolled subject-
# stratified probe for downstream/brain-age work. CLI: `neuralbench eeg <task> -m <model>`.
#
# Installed to a --target dir on /mnt/t9 (NEVER pip --user — shadows the SIF/container
# stack), PYTHONNOUSERSITE=1. GPU for the FM forward passes.
#
# PENDING before this is more than a stub:
#   1. Stage HBN via neuralfetch (scripts/neuralfetch_stage.sbatch --study Shirazi2024Hbn).
#   2. Verify the task name + config (psychopathology / age / motor_imagery / ...).
#   3. REVE is gated -> HF token must be in-container (see run_fmscope_leaderboard.sh).
#
# Usage (once unblocked):
#   sbatch scripts/neuralbench_eval.sbatch eeg psychopathology -m reve
#   sbatch scripts/neuralbench_eval.sbatch eeg age -m reve
set -euo pipefail
cd "$HOME/dev/emeg-fm"

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
T9="${T9:-/mnt/t9}"
NB_LIBS="$T9/neuralbench_libs"          # isolated --target install (persists across runs)
HOST_UIDGID="$(id -u):$(id -g)"
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"   # gated REVE
mkdir -p "$NB_LIBS" "$T9/neuralbench_out"

# Default to the HBN bifactor eval on REVE if no args given.
ARGS=("${@:-eeg psychopathology -m reve}")

exec docker run --rm --gpus all --ipc=host \
  -v "$HOME/dev/emeg-fm:/emeg-fm" -v "$T9:$T9" \
  -e PYTHONNOUSERSITE=1 -e PYTHONPATH="$NB_LIBS:/emeg-fm" \
  -e HF_HOME="$T9/hf" -e HF_TOKEN="$HF_TOKEN" -e MNE_DATA="$T9/moabb_data" \
  -e NEURALBENCH_OUT="$T9/neuralbench_out" \
  -w /emeg-fm "$IMAGE" \
  bash -c "
    set -e
    python -c 'import neuralbench' 2>/dev/null || \
      pip install --no-input --target '$NB_LIBS' 'neuralbench[models]'
    neuralbench ${ARGS[*]} ;
    rc=\$?; chown -R $HOST_UIDGID '$NB_LIBS' '$T9/neuralbench_out' 2>/dev/null || true; exit \$rc
  "
