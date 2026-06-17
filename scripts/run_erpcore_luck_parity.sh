#!/usr/bin/env bash
# Run the ERP CORE Luck-lab parity decoding comparison on local-SSD infra.
#
# Drives scripts/erpcore_luck_parity.py inside the Docker NGC PyTorch 26.05
# container with uv-installed deps on /mnt/t9 (the local 3.6 TB SSD) — never
# /data NFS or the NFS apptainer SIF. Frozen-REVE block-6 embeddings + raw
# scalp voltage decoded under the ERPLAB12 pop_decoding protocol; writes
# results/erpcore_luck_parity/erpcore_luck_parity.{csv,md}.
#
# The moabb *fork* (~/dev/moabb) is first on PYTHONPATH so its ErpCore2021_*
# registry wins; /mnt/t9/moabblibs supplies mne/pyriemann/sklearn, /mnt/t9/tokfix
# pins tokenizers/hub, emeg-fm + fmscope provide the REVE adapter + svm_probe.
#
# Usage:
#   scripts/run_erpcore_luck_parity.sh [extra args for erpcore_luck_parity.py]
# Examples:
#   scripts/run_erpcore_luck_parity.sh                          # all 7 components
#   scripts/run_erpcore_luck_parity.sh --components N170 P3 --max-subjects 10
set -euo pipefail

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
MOABB_FORK="${MOABB_FORK:-$HOME/dev/moabb}"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
OUT_SUB="${OUT_SUB:-results/erpcore_luck_parity}"

mkdir -p "$EMEG_FM/$OUT_SUB"

# HF gated token (REVE) — read from the user's cache, passed only as a runtime
# env var (never written to disk). Required for the REVE column.
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
  bash -c "python scripts/erpcore_luck_parity.py --out-dir '/emeg-fm/$OUT_SUB' $* ; \
           rc=\$?; chown -R $HOST_UIDGID '/emeg-fm/$OUT_SUB'; exit \$rc"
