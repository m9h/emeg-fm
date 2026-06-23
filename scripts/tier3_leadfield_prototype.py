"""Tier-3 forward-model prototype (5 subjects) — the CAUSAL volume-conduction test.

Per subject: CHARM head segmentation (raw T1) → SimNIBS EEG leadfield. Comparing the lead fields
across ages shows how the conduction operator itself changes with age — the mechanism the tier-1
correlational redundancy can only hint at.

    python scripts/tier3_leadfield_prototype.py --dry-run     # check input discovery first
    python scripts/tier3_leadfield_prototype.py [--limit N]   # ~1-3 h CHARM + 0.5-2 h leadfield / subj

SimNIBS 4.1.0 is installed in the conda env below but its petsc4py needs libstdc++ preloaded (the
shipped libpetsc.so does not list it as NEEDED), and the bin/ console-scripts don't set that up — so
every SimNIBS call here goes through the env's python with LD_PRELOAD/LD_LIBRARY_PATH set
(`_simnibs_env`), invoking CHARM as a module (`-m simnibs.cli.charm`). See docs/simnibs_install_notes.md.

CHARM's CAT cortical-surface reconstruction (run_cat_multiprocessing) crashes on this aarch64 build,
so we run with `--usesettings charm_nosurf.ini` (surf=[]/pial=[] + the *_from_surf flags off). The
central GM surfaces are only used to refine the segmentation and for source-space modelling; the
TES/EEG volume-conduction FEM mesh is built from the tissue *volume* labels alone, which SAMSEG
produces fine. (For the first subject, segmentation had already completed before the CAT crash, so the
mesh was recovered directly with `charm <id> --mesh`.)

Conductivity is **scalar (isotropic)**: `dwi2cond` is not packaged in this aarch64 build, so DWI
anisotropy ('vn') is a documented follow-up. The age-varying head *geometry* from CHARM is the
first-order conduction effect and is fully captured. The EEG cap is SimNIBS's bundled 10-10 montage;
an EGI-128 cap built from emeg_fm/montage.py (GSN-HydroCel-128) is the matching follow-up.
"""
import argparse
import glob
import os
import subprocess

BIDS = "/data/raw/hbn-bids"
QSIPREP = "/data/raw/hbn-qsiprep"
OUT = "/data/derivatives/volume_conduction/forward"

# CHARM settings with CAT cortical-surface reconstruction disabled (surf=[]/pial=[] + the
# *_from_surf flags). The CAT step (run_cat_multiprocessing) crashes on this aarch64 build, and a
# TES/EEG FEM mesh needs only the tissue *volume* labels — not central GM surfaces. Absolute path
# because charm runs with cwd=subject dir. See docs/simnibs_install_notes.md.
CHARM_SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charm_nosurf.ini")

# SimNIBS 4.1.0 install (conda env) + the libstdc++ preload its petsc4py requires.
SIMNIBS_ENV = "/home/mhough/miniforge3/envs/simnibs_test_v11"
SIMNIBS_PY = f"{SIMNIBS_ENV}/bin/python"
# Prototype EEG cap = SimNIBS's bundled 10-10 montage; swap for an EGI-128 cap (emeg_fm/montage.py).
EEG_CAP = (f"{SIMNIBS_ENV}/lib/python3.12/site-packages/simnibs/resources/"
           "ElectrodeCaps_MNI/EEG10-10_UI_Jurak_2007.csv")

# verified EEG∩T1∩DWI∩age (see docs/volume_conduction_plan.md)
SUBJECTS = ["sub-NDARAA948VFH", "sub-NDARAB458VK9", "sub-NDARAC349YUC",
            "sub-NDARAC853DTE", "sub-NDARAD224CRB"]

# Run the leadfield in the env python (TDCSLEADFIELD = EEG forward by reciprocity); argv = m2m,fem,cap.
_LEADFIELD = """
import sys
from simnibs import sim_struct, run_simnibs
m2m, pathfem, cap = sys.argv[1], sys.argv[2], sys.argv[3]
lf = sim_struct.TDCSLEADFIELD()
lf.subpath = m2m
lf.pathfem = pathfem
lf.eeg_cap = cap
lf.field = "E"
lf.anisotropy_type = "scalar"   # 'vn' (DWI-anisotropic) needs dwi2cond, absent in this build
lf.interpolation = None         # no 'middle gm' surface interpolation (CAT surfaces broken on aarch64)
lf.tissues = [2]                # ROI = GM *volume* tets (ElementTags.GM). The stock default is
                                # [1006]=eye-balls (a landmark); cortex normally comes from the
                                # disabled 'middle gm' step, so without this the leadfield is eyes-only.
run_simnibs(lf)
"""


def _simnibs_env() -> dict:
    """os.environ + the LD_PRELOAD/LD_LIBRARY_PATH that make SimNIBS's petsc4py importable."""
    e = dict(os.environ)
    lib = f"{SIMNIBS_ENV}/lib"
    pre = f"{lib}/libstdc++.so.6"
    e["LD_PRELOAD"] = f"{pre} {e['LD_PRELOAD']}" if e.get("LD_PRELOAD") else pre
    e["LD_LIBRARY_PATH"] = f"{lib}:{e['LD_LIBRARY_PATH']}" if e.get("LD_LIBRARY_PATH") else lib
    return e


def inputs(sub: str):
    t1 = next(iter(glob.glob(f"{BIDS}/{sub}/ses-*/anat/{sub}_*acq-HCP*T1w.nii.gz")
                   or glob.glob(f"{BIDS}/{sub}/ses-*/anat/{sub}_*T1w.nii.gz")), None)
    dwi = next(iter(glob.glob(f"{QSIPREP}/{sub}/ses-*/dwi/*acq-64dir*space-T1w*desc-preproc_dwi.nii.gz")), None)
    return t1, dwi


def have_simnibs() -> bool:
    """SimNIBS imports only under the preload env, and from the env's own python — check exactly that."""
    try:
        r = subprocess.run([SIMNIBS_PY, "-c", "import simnibs"],
                           env=_simnibs_env(), capture_output=True, timeout=180)
        return r.returncode == 0
    except Exception:
        return False


def run_subject(sub: str, dry: bool):
    t1, dwi = inputs(sub)
    if not t1:
        print(f"[skip {sub}] no raw T1"); return
    subID = sub.replace("sub-", "")
    subdir = f"{OUT}/{sub}"
    m2m = f"{subdir}/m2m_{subID}"
    fem = f"{subdir}/leadfield"
    if dry:
        print(f"[dry {sub}] T1={os.path.basename(t1)} dwi={'y' if dwi else 'no'} -> {m2m}"); return
    os.makedirs(subdir, exist_ok=True)
    env = _simnibs_env()
    # 1) head segmentation + mesh (CHARM/SAMSEG; cortical surfaces disabled via CHARM_SETTINGS) —
    #    idempotent: skip if the m2m mesh already exists; --forcerun overwrites a partial m2m folder.
    if os.path.exists(f"{m2m}/{subID}.msh"):
        print(f"[skip charm {sub}] {m2m} present", flush=True)
    else:
        subprocess.run([SIMNIBS_PY, "-m", "simnibs.cli.charm", subID, t1,
                        "--usesettings", CHARM_SETTINGS, "--forcerun"],
                       cwd=subdir, env=env, check=True)
    # 2) EEG leadfield (scalar conductivity; see module docstring for the DWI-anisotropy follow-up)
    subprocess.run([SIMNIBS_PY, "-c", _LEADFIELD, m2m, fem, EEG_CAP], env=env, check=True)
    print(f"[ok {sub}] CHARM + EEG leadfield -> {fem}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and not have_simnibs():
        print(f"SimNIBS not importable from {SIMNIBS_PY} under the preload env. See "
              "docs/simnibs_install_notes.md. Use --dry-run to check inputs without SimNIBS.")
        return
    subs = SUBJECTS[: a.limit] if a.limit else SUBJECTS
    print(f"tier-3 forward-model prototype over {len(subs)} subjects -> {OUT}")
    for s in subs:
        try:
            run_subject(s, a.dry_run)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL {s}] {e}", flush=True)


if __name__ == "__main__":
    main()
