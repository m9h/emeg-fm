#!/usr/bin/env bash
# Run the FMScope identity-free MOABB leaderboard sweep on local-SSD infra.
#
# Drives scripts/moabb_identity_leaderboard.py inside the Docker NGC PyTorch
# 26.05 container with uv-installed deps on /mnt/t9 (the local 3.6 TB SSD) —
# never /data NFS or the NFS apptainer SIF. Frozen-REVE block-6 features +
# fmscope.verdict.audit_cell; writes results/moabb_fmscope/leaderboard_*.{csv,md}.
#
# The moabb *fork* (~/dev/moabb) is first on PYTHONPATH so its registry wins;
# /mnt/t9/moabblibs supplies mne/pyriemann/sklearn, /mnt/t9/tokfix pins
# tokenizers/hub, emeg-fm + fmscope provide the REVE adapter for the FM column.
#
# Usage:
#   scripts/run_fmscope_leaderboard.sh [extra args for moabb_identity_leaderboard.py]
# Examples:
#   scripts/run_fmscope_leaderboard.sh                                  # full registry (skips done)
#   scripts/run_fmscope_leaderboard.sh --datasets Beetl2021-A Beetl2021-B PhysionetMotorImagery
set -euo pipefail

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
MOABB_FORK="${MOABB_FORK:-$HOME/dev/moabb}"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
OUT_SUB="${OUT_SUB:-results/moabb_fmscope}"

mkdir -p "$EMEG_FM/$OUT_SUB"

# HF gated token (REVE) — read from the user's cache, passed only as a runtime
# env var (never written to disk). Required for the FM column.
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"

# PYTHONPATH order: tokfix (pinned hub/tokenizers) > fork moabb > emeg_fm/fmscope
# > moabblibs (deps incl. its own moabb, which loses to the fork) > fooof_libs
# (FOOOF for the --fooof aperiodic-ablation diagnostic; --no-deps install, so it
# borrows numpy/scipy from moabblibs/container).
PYPATH="$T9/tokfix:/moabb:/emeg-fm:/emeg-fm/fmscope:$T9/moabblibs:$T9/fooof_libs"

# Host uid:gid so docker-written leaderboard files stay host-editable (the
# container runs as root; chown them back after the sweep finishes).
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
  bash -c "python scripts/moabb_identity_leaderboard.py --out-dir '/emeg-fm/$OUT_SUB' $* ; \
           rc=\$?; chown -R $HOST_UIDGID '/emeg-fm/$OUT_SUB'; exit \$rc"
