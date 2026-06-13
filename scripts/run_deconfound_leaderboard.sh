#!/usr/bin/env bash
# Run the identity-free MOABB leaderboard (both columns) on local-SSD infra.
#
# Uses the Docker NGC PyTorch 26.05 container + uv-installed deps on /mnt/t9
# (the local 3.6 TB SSD) — never /data NFS or the NFS apptainer SIF. The moabb
# *fork* (~/dev/moabb, with moabb/deconfound) is placed first on PYTHONPATH so
# its `moabb.deconfound` wins; /mnt/t9/moabblibs supplies mne/pyriemann/sklearn,
# /mnt/t9/tokfix pins tokenizers/hub for transformers, and emeg-fm + fmscope
# provide the REVE adapter for the FM column.
#
# Usage:
#   scripts/run_deconfound_leaderboard.sh [extra args for `python -m moabb.deconfound`]
# Examples:
#   scripts/run_deconfound_leaderboard.sh                      # full LeftRight sweep, both columns
#   scripts/run_deconfound_leaderboard.sh --no-fm              # pipeline column only (no GPU/FM)
#   scripts/run_deconfound_leaderboard.sh --datasets BNCI2014-001 Zhou2016
set -euo pipefail

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
MOABB_FORK="${MOABB_FORK:-$HOME/dev/moabb}"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
# Output dir, expressed RELATIVE to the emeg-fm repo so it resolves both on the
# host (for mkdir) and inside the container (where the repo is mounted at
# /emeg-fm). Passing a host-absolute path into the container would write to the
# container's ephemeral FS and vanish on --rm.
OUT_SUB="${OUT_SUB:-results/moabb_deconfound}"
OUT_DIR="$EMEG_FM/$OUT_SUB"

# HF gated token (REVE) — read from the user's cache, passed only as a runtime
# env var (never written to disk here). Required for the FM column.
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"

mkdir -p "$OUT_DIR"

# PYTHONPATH order matters: tokfix (pinned hub/tokenizers) > fork moabb >
# emeg_fm/fmscope > moabblibs (deps incl. its own moabb, which loses to the fork).
PYPATH="$T9/tokfix:/moabb:/emeg-fm:/emeg-fm/fmscope:$T9/moabblibs"

exec docker run --rm --gpus all \
  -v "$MOABB_FORK:/moabb" \
  -v "$EMEG_FM:/emeg-fm" \
  -v "$T9:$T9" \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONPATH="$PYPATH" \
  -e HF_HOME="$T9/hf" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MNE_DATA="$T9/moabb_data" \
  -w /moabb \
  "$IMAGE" \
  python -m moabb.deconfound --out-dir "/emeg-fm/$OUT_SUB" "$@"
