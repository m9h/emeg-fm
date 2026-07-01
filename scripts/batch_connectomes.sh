#!/usr/bin/env bash
# Parallel 4S structural-connectome batch over the DWI subset, niced and shared-box-friendly.
# Idempotent and pipelines with the download: each pass re-scans for subjects whose DWI has landed
# and whose connectome isn't built yet, runs ~3 concurrently, sleeps, repeats until all are done.
#   RES=456 NSTREAM=1M NTHREADS=5 bash scripts/batch_connectomes.sh
SUBSET=/data/derivatives/volume_conduction/dwi_zeta_subset.txt
SCRIPT=/home/mhough/dev/emeg-fm/scripts/build_connectome.sh
CONN=/data/derivatives/volume_conduction/connectomes
DWI=/data/derivatives/volume_conduction/qsiprep_dwi
RES="${RES:-456}"
export NSTREAM="${NSTREAM:-1M}" NTHREADS="${NTHREADS:-5}"
N=$(wc -l < "$SUBSET")
echo "batch start $(date +%H:%M:%S)  subset=$N  RES=$RES  NSTREAM=$NSTREAM  NTHREADS=$NTHREADS  (3 concurrent, niced)"
for pass in $(seq 1 80); do
  todo=$(while read s; do
    [ -f "$CONN/$s/connectome_${RES}.csv" ] && continue
    [ -f "$CONN/$s/.qcfail_t1" ] && continue          # QC-failed (degenerate T1) -> don't retry
    [ -n "$(find "$DWI/$s" -name '*space-T1w_desc-preproc_dwi.nii.gz' 2>/dev/null)" ] && echo "$s"
  done < "$SUBSET")
  if [ -n "$todo" ]; then
    echo "$todo" | xargs -P 3 -I{} nice -n 12 bash "$SCRIPT" {} "$RES" || true
  fi
  done_n=$(ls "$CONN"/*/connectome_${RES}.csv 2>/dev/null | wc -l)
  landed=$(ls -d "$DWI"/sub-* 2>/dev/null | wc -l)
  echo ">>> pass $pass: $done_n/$N connectomes built ($landed DWI landed)  $(date +%H:%M:%S)"
  [ "$done_n" -ge "$N" ] && { echo "ALL_DONE $(date +%H:%M:%S)"; break; }
  sleep 90
done
