#!/usr/bin/env bash
#SBATCH --job-name=reve-hbn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# Frozen REVE-base (block 6) brain-age on HBN, on the SAME 10-fold CV splits as
# the classical coffeine filterbank-riemann baseline (MAE 1.69 / R2 0.61):
# same 2537 feature-OK subjects in sorted participant_id order + KFold(10,
# shuffle, random_state=42). EGI GSN-HydroCel labels (E1..E128) are remapped to
# REVE's electrode vocabulary by nearest 3D montage position. Docker NGC 26.05 +
# GPU on /mnt/t9; reads HBN autoreject epochs from /data:ro (mounted by t9.sh).
set -euo pipefail
cd "$HOME/dev/emeg-fm"
exec scripts/reve_brain_age_t9.sh \
  --epochs-glob "/data/derivatives/brain_age/HBN_EEG/sub-*/eeg/*proc-autoreject_epo.fif" \
  --participants "/data/datasets/hbn-eeg/participants.tsv" \
  --egi-remap --seed 42 \
  --out "/mnt/t9/reve_hbn_emb.npz" \
  "$@"
