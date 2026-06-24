#!/usr/bin/env bash
# Run FastSurfer (full pipeline, 1mm) for ONE subject on the x86 fedora box.
# Output goes to LOCAL disk (heavy FreeSurfer I/O off the NAS), then the finished subject dir is
# copied to the shared /data. A sentinel file records the true exit code (the run_fastsurfer | tee
# pattern otherwise masks it). Args: SID  T1_ABS_PATH
set -u
sid="$1"; t1="$2"
work="/home/mhough/fs_work"                       # local scratch (281G free on /home)
out="/data/derivatives/volume_conduction/fastsurfer"   # shared NAS destination
license="/data/derivatives/volume_conduction/fs_license.txt"   # staged here (one level up from $out)
mkdir -p "$work"
rm -rf "$work/$sid" "$out/$sid.done"

# Memory-capped + de-prioritised: this box is a shared interactive machine (31 GB). --parallel + 8
# threads OOM-killed the host once. --memory caps the container's cgroup so any overflow is killed
# INSIDE the container (graceful failure, sentinel rc!=0) instead of taking down the host's processes.
nice -n 15 podman run --rm --memory=16g --memory-swap=30g \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --userns=keep-id --user "$(id -u):$(id -g)" \
  -v "$(dirname "$t1"):/in:ro" \
  -v "$work:/out" \
  -v "$license:/fs_license.txt:ro" \
  deepmi/fastsurfer:latest \
  --t1 "/in/$(basename "$t1")" --sid "$sid" --sd /out \
  --fs_license /fs_license.txt --vox_size 1 --3T --threads 4    # no --parallel: sequential hemispheres, lower peak RAM
rc=$?

if [ "$rc" -eq 0 ] && [ -f "$work/$sid/surf/lh.pial" ] && [ -f "$work/$sid/surf/rh.pial" ]; then
  rsync -a "$work/$sid" "$out/"
  echo "rc=0 surfaces=yes copied=$?" > "$out/$sid.done"
else
  echo "rc=$rc surfaces=$([ -f "$work/$sid/surf/lh.pial" ] && echo yes || echo no)" > "$out/$sid.done"
fi
