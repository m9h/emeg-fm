#!/usr/bin/env bash
# FastSurfer SEG-ONLY batch on the DGX (aarch64 GB10) -> aparc.DKTatlas+aseg + per-region volume stats.
# GPU + low RAM (~4-8 GB), NO recon-surf / NO FreeSurfer -> safe on the 120 GB box (this is the seg half
# that runs natively on aarch64; surfaces are the x86 track). Idempotent: skips subjects already done.
#
#   LIMIT=20 bash scripts/run_fastsurfer_seg.sh   # pilot first N
#   bash scripts/run_fastsurfer_seg.sh            # whole hbn-bids cohort (~2 min/subj on the GB10)
set -u
BIDS=/data/raw/hbn-bids
OUT=/data/derivatives/volume_conduction/fastsurfer_seg
IMAGE="${IMAGE:-fastsurfer:grace}"
LIMIT="${LIMIT:-0}"
mkdir -p "$OUT"
if [ -n "${SUBJ_FILE:-}" ] && [ -f "${SUBJ_FILE:-}" ]; then   # subject ids (one per line) from a file
  mapfile -t subs < "$SUBJ_FILE"
elif [ "$#" -gt 0 ]; then                   # explicit subject ids as args -> just those
  subs=("$@")
else                                        # else the whole hbn-bids cohort (optionally first LIMIT)
  mapfile -t subs < <(ls -d "$BIDS"/sub-* 2>/dev/null | xargs -n1 basename | sort)
  [ "$LIMIT" -gt 0 ] && subs=("${subs[@]:0:$LIMIT}")
fi
echo "$(date +%H:%M:%S)  ${#subs[@]} subjects -> $OUT (LIMIT=$LIMIT)"
done=0; fail=0; skip=0
for sub in "${subs[@]}"; do
  if [ -f "$OUT/$sub/stats/aseg+DKT.VINN.stats" ]; then skip=$((skip+1)); continue; fi
  t1=$(ls "$BIDS/$sub"/ses-*/anat/"$sub"_*acq-HCP*T1w.nii.gz 2>/dev/null | head -1 \
       || ls "$BIDS/$sub"/ses-*/anat/"$sub"_*T1w.nii.gz 2>/dev/null | head -1)
  [ -z "${t1:-}" ] && { echo "[no-t1 $sub]"; continue; }
  if docker run --gpus all --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
       -v "$(dirname "$t1"):/in:ro" -v "$OUT:/out" "$IMAGE" \
       --t1 "/in/$(basename "$t1")" --sid "$sub" --sd /out --seg_only --threads 8 \
       > "$OUT/$sub.seg.log" 2>&1; then
    done=$((done+1)); echo "[ok $sub] ($done done)"
  else
    fail=$((fail+1)); echo "[FAIL $sub] see $OUT/$sub.seg.log"
  fi
done
echo ">>> $(date +%H:%M:%S)  newly segmented=$done  failed=$fail  already-present=$skip"
