#!/usr/bin/env bash
# Sequential FastSurfer surface runs (host-safe run_one.sh: 16G RAM cap + swap) for the given subject
# ids. Sequential (one at a time) keeps the Legion's memory safe. Args: SID...
# Idempotent: skips subjects whose surfaces are already on /data.
set -u
RUN=/data/derivatives/volume_conduction/fastsurfer/run_one.sh
for s in "$@"; do
  if [ -f "/data/derivatives/volume_conduction/fastsurfer/$s/surf/lh.pial" ]; then
    echo "[skip $s] surfaces present"; continue
  fi
  t1=$(ls /data/raw/hbn-bids/$s/ses-*/anat/${s}_*acq-HCP*T1w.nii.gz 2>/dev/null | head -1 \
       || ls /data/raw/hbn-bids/$s/ses-*/anat/${s}_*T1w.nii.gz 2>/dev/null | head -1)
  [ -z "${t1:-}" ] && { echo "[no-t1 $s]"; continue; }
  echo "[run $s] $(date +%H:%M:%S)"
  bash "$RUN" "$s" "$t1"
done
echo "BATCH DONE $(date +%H:%M:%S)"
