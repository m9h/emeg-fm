#!/usr/bin/env bash
# Run FastSurfer's FULL pipeline (seg + recon-surf SURFACES) on an x86_64 + NVIDIA-GPU host.
#
# WHY x86: FastSurfer's surface half (recon-surf) needs FreeSurfer 7.4.1, which has no Linux-aarch64
# build, so it cannot run on the GB10/DGX-Spark (the seg half does — see fastsurfer:grace there). The
# surfaces produced here feed `charm --fs-dir` back on the aarch64 box, which then skips the broken CAT
# surface step (verified: simnibs charm_main.py:334) and yields a cortical-surface source-space leadfield.
#
# WORKFLOW (two machines):
#   1. (aarch64) rsync the 5 prototype T1s to the x86 host (each ~26 MB):
#        rsync -a /data/raw/hbn-bids/sub-NDARAA948VFH .../sub-NDARAD224CRB  x86:/path/hbn-bids/
#   2. (x86)     bash run_fastsurfer_x86.sh   # ~1 h/subject on a GPU; produces FreeSurfer subject dirs
#   3. (aarch64) rsync the FS subject dirs back to $FS_OUT below, then run the surface leadfield:
#        charm <id> <T1> --fs-dir $FS_OUT/sub-<id>   # grabs surfaces, no CAT; then TDCSLEADFIELD
#                                                     # with interpolation='middle gm' (surface source space)
#
# Requirements on the x86 host: Docker + NVIDIA Container Toolkit, the public FastSurfer image
# (`docker pull deepmi/fastsurfer:latest`), and a FreeSurfer license file.
set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-./hbn-bids}"                       # where the T1s live on the x86 host
FS_OUT="${FS_OUT:-./fastsurfer}"                           # FreeSurfer SUBJECTS_DIR output
FS_LICENSE="${FS_LICENSE:-$HOME/license.txt}"             # FreeSurfer license (register at surfer.nmr.mgh.harvard.edu)
IMAGE="${IMAGE:-deepmi/fastsurfer:latest}"                 # x86 FastSurfer (full pipeline incl. FreeSurfer)
THREADS="${THREADS:-8}"
if [ "$#" -gt 0 ]; then
  SUBJECTS=("$@")
else
  SUBJECTS=(sub-NDARAA948VFH sub-NDARAB458VK9 sub-NDARAC349YUC sub-NDARAC853DTE sub-NDARAD224CRB)
fi

mkdir -p "$FS_OUT"
for sub in "${SUBJECTS[@]}"; do
  # match the acq-HCP T1 (fallback any T1), same selection as tier3_leadfield_prototype.inputs()
  t1=$(ls "$BIDS_ROOT/$sub"/ses-*/anat/"$sub"_*acq-HCP*T1w.nii.gz 2>/dev/null | head -1 \
       || ls "$BIDS_ROOT/$sub"/ses-*/anat/"$sub"_*T1w.nii.gz 2>/dev/null | head -1)
  if [ -z "${t1:-}" ]; then echo "[skip $sub] no T1 under $BIDS_ROOT/$sub"; continue; fi
  if [ -f "$FS_OUT/$sub/surf/lh.pial" ]; then echo "[skip $sub] surfaces present"; continue; fi
  echo "[run $sub] T1=$(basename "$t1")"
  docker run --gpus all --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$(cd "$(dirname "$t1")" && pwd):/in:ro" \
    -v "$(cd "$FS_OUT" && pwd):/out" \
    -v "$(cd "$(dirname "$FS_LICENSE")" && pwd)/$(basename "$FS_LICENSE"):/fs_license.txt:ro" \
    "$IMAGE" \
    --t1 "/in/$(basename "$t1")" --sid "$sub" --sd /out \
    --fs_license /fs_license.txt --3T --parallel --threads "$THREADS"
  echo "[ok $sub] -> $FS_OUT/$sub  (surf/, mri/, label/)"
done
echo ">>> done. rsync $FS_OUT/* back to the aarch64 box's \$FS_OUT, then run charm --fs-dir per subject."
