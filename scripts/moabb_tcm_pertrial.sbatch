#!/usr/bin/env bash
#SBATCH --job-name=tcm-pertrial
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# No --time: gx10-dgx-spark single-user / infinite walltime.
#
# Wang TCM per-trial identity-trap sweep over a MOABB paradigm's datasets, in the
# Docker NGC 26.05 container (moabb fork + emeg-fm + moabblibs on /mnt/t9).
# Pure-CPU (SVD + sklearn erasure); no GPU. Output CSV is paradigm-specific
# (tcm_pertrial.csv for MI; tcm_pertrial_<paradigm>.csv for erp/ssvep). Usage:
#   sbatch scripts/moabb_tcm_pertrial.sbatch [--paradigm erp|ssvep] [--datasets ...]
set -euo pipefail
cd "$HOME/dev/emeg-fm"

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
MOABB_FORK="${MOABB_FORK:-$HOME/dev/moabb}"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
mkdir -p "$EMEG_FM/results/moabb_fmscope"
PYPATH="$T9/tokfix:/moabb:/emeg-fm:/emeg-fm/fmscope:$T9/moabblibs"
HOST_UIDGID="$(id -u):$(id -g)"

exec docker run --rm \
  -v "$MOABB_FORK:/moabb" -v "$EMEG_FM:/emeg-fm" -v "$T9:$T9" \
  -e PYTHONNOUSERSITE=1 -e PYTHONPATH="$PYPATH" -e MNE_DATA="$T9/moabb_data" \
  -w /emeg-fm "$IMAGE" \
  bash -c "python scripts/moabb_tcm_pertrial.py $* ; \
           rc=\$?; chown -R $HOST_UIDGID '/emeg-fm/results/moabb_fmscope'; exit \$rc"
