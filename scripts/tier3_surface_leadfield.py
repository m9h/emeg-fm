"""Surface-source-space EEG leadfield via FastSurfer surfaces + `charm --fs-dir` (the x86-surface track).

FastSurfer (run on x86) produces FreeSurfer-format surfaces; `charm <id> T1 --fs-dir <fsdir>` then uses
them for the cortex (taking the `if fs_dir:` branch in charm_main.py -> SKIPS the broken aarch64 CAT
surface step) while SAMSEG still does the full head tissue segmentation and mmg meshes it. With the
central GM surface available, TDCSLEADFIELD runs with `interpolation='middle gm'` -> the leadfield on the
**central GM surface** (the standard EEG source space), the proper version of the GM-*volume* fallback in
`tier3_leadfield_prototype.py`.

    python scripts/tier3_surface_leadfield.py sub-NDARAB458VK9 [sub-...]
"""
import argparse
import glob
import os
import subprocess

BIDS = os.environ.get("BIDS", "/data/raw/hbn-bids")
# FreeSurfer subject dirs: FastSurfer (x86 track) by default, or set FSROOT to an FS6 recon-all tree
# (e.g. /data/raw/hbn-freesurfer) to harvest gold-standard surfaces with no recompute.
FSROOT = os.environ.get("FSROOT", "/data/derivatives/volume_conduction/fastsurfer")
OUT = "/data/derivatives/volume_conduction/surface_forward"
SIMNIBS_ENV = "/home/mhough/miniforge3/envs/simnibs_test_v11"
SIMNIBS_PY = f"{SIMNIBS_ENV}/bin/python"
EEG_CAP = (f"{SIMNIBS_ENV}/lib/python3.12/site-packages/simnibs/resources/"
           "ElectrodeCaps_MNI/EEG10-10_UI_Jurak_2007.csv")

# leadfield on the central GM surface (surfaces now exist via --fs-dir); contrast with the GM-volume run
_LEADFIELD = """
import sys
from simnibs import sim_struct, run_simnibs
m2m, pathfem, cap = sys.argv[1], sys.argv[2], sys.argv[3]
lf = sim_struct.TDCSLEADFIELD()
lf.subpath = m2m
lf.pathfem = pathfem
lf.eeg_cap = cap
lf.field = "E"
lf.anisotropy_type = "scalar"
lf.interpolation = "middle gm"     # central GM surface source space
run_simnibs(lf)
"""


def _env() -> dict:
    e = dict(os.environ)
    lib = f"{SIMNIBS_ENV}/lib"
    pre = f"{lib}/libstdc++.so.6"
    e["LD_PRELOAD"] = f"{pre} {e['LD_PRELOAD']}" if e.get("LD_PRELOAD") else pre
    e["LD_LIBRARY_PATH"] = f"{lib}:{e['LD_LIBRARY_PATH']}" if e.get("LD_LIBRARY_PATH") else lib
    return e


def _t1(sub: str):
    return next(iter(glob.glob(f"{BIDS}/{sub}/ses-*/anat/{sub}_*acq-HCP*T1w.nii.gz")
                     or glob.glob(f"{BIDS}/{sub}/ses-*/anat/{sub}_*T1w.nii.gz")), None)


def run_subject(sub: str):
    subID = sub.replace("sub-", "")
    fsdir = f"{FSROOT}/{sub}"
    # prefer the BIDS T1; fall back to the recon-all input (raw.nii.gz) for FS6 trees
    t1 = _t1(sub) or (f"{fsdir}/raw.nii.gz" if os.path.exists(f"{fsdir}/raw.nii.gz") else None)
    if not t1:
        print(f"[skip {sub}] no T1 (BIDS or {fsdir}/raw.nii.gz)"); return
    if not os.path.exists(f"{fsdir}/surf/lh.pial"):
        print(f"[skip {sub}] no surfaces at {fsdir}"); return
    subdir = f"{OUT}/{sub}"
    m2m = f"{subdir}/m2m_{subID}"
    fem = f"{subdir}/leadfield"
    os.makedirs(subdir, exist_ok=True)
    env = _env()
    # 1) charm with FreeSurfer surfaces (skips CAT) -> SAMSEG head seg + mmg mesh
    if os.path.exists(f"{m2m}/{subID}.msh"):
        print(f"[skip charm {sub}] mesh present", flush=True)
    else:
        subprocess.run([SIMNIBS_PY, "-m", "simnibs.cli.charm", subID, t1, "--fs-dir", fsdir, "--forcerun"],
                       cwd=subdir, env=env, check=True)
    # 2) leadfield on the central GM surface
    if glob.glob(f"{fem}/*_leadfield_*.hdf5"):
        print(f"[skip leadfield {sub}] present", flush=True); return
    subprocess.run([SIMNIBS_PY, "-c", _LEADFIELD, m2m, fem, EEG_CAP], env=env, check=True)
    print(f"[ok {sub}] surface-source leadfield -> {fem}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="+")
    a = ap.parse_args()
    print(f"surface leadfield (charm --fs-dir) over {len(a.subjects)} subjects -> {OUT}")
    for s in a.subjects:
        try:
            run_subject(s)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL {s}] {e}", flush=True)


if __name__ == "__main__":
    main()
