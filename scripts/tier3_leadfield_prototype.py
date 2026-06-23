"""Tier-3 forward-model prototype (5 subjects) — the CAUSAL volume-conduction test.

Per subject: CHARM head segmentation (raw T1) → anisotropic conductivity from DWI tensors → SimNIBS
EEG leadfield (EGI-128 electrodes). Comparing the lead fields across ages shows how the conduction
operator itself changes with age — the mechanism the tier-1 correlational redundancy can only hint at.

STAGED — requires SimNIBS installed (it is a source checkout only right now):
    pip install <simnibs-checkout>     # or the official installer; bundles charm/SAMSEG + atlas
    python scripts/tier3_leadfield_prototype.py --dry-run
    python scripts/tier3_leadfield_prototype.py        # ~1-3 h CHARM + 0.5-2 h leadfield per subject

CHARM and dwi2cond are stable SimNIBS 4.x CLIs and are wired here; the leadfield call is sketched
against the v4 Python API and must be confirmed on the installed version (marked CONFIRM below).
"""
import argparse
import glob
import os
import subprocess

BIDS = "/data/raw/hbn-bids"
QSIPREP = "/data/raw/hbn-qsiprep"
OUT = "/data/derivatives/volume_conduction/forward"

# verified EEG∩T1∩DWI∩age (see docs/volume_conduction_plan.md)
SUBJECTS = ["sub-NDARAA948VFH", "sub-NDARAB458VK9", "sub-NDARAC349YUC",
            "sub-NDARAC853DTE", "sub-NDARAD224CRB"]


def inputs(sub: str):
    t1 = next(iter(glob.glob(f"{BIDS}/{sub}/ses-*/anat/{sub}_*acq-HCP*T1w.nii.gz")
                   or glob.glob(f"{BIDS}/{sub}/ses-*/anat/{sub}_*T1w.nii.gz")), None)
    dwi = next(iter(glob.glob(f"{QSIPREP}/{sub}/ses-*/dwi/*acq-64dir*space-T1w*desc-preproc_dwi.nii.gz")), None)
    return t1, dwi


def have_simnibs() -> bool:
    try:
        import simnibs  # noqa: F401
        return True
    except Exception:
        return False


def run_subject(sub: str, dry: bool):
    t1, dwi = inputs(sub)
    if not t1:
        print(f"[skip {sub}] no raw T1"); return
    m2m = f"{OUT}/{sub}/m2m_{sub.replace('sub-', '')}"
    if dry:
        print(f"[dry {sub}] T1={os.path.basename(t1)} dwi={'y' if dwi else 'no'} -> {m2m}"); return
    os.makedirs(os.path.dirname(m2m), exist_ok=True)
    # 1) head segmentation + mesh (SAMSEG/CHARM) — stable CLI
    subprocess.run(["charm", sub.replace("sub-", ""), t1], cwd=f"{OUT}/{sub}", check=True)
    # 2) anisotropic conductivity from DWI tensors — stable CLI (registers DTI to the mesh)
    if dwi:
        tensor = f"{OUT}/{sub}/dti_tensor.nii.gz"
        b = dwi[: -len(".nii.gz")] + ".b"
        subprocess.run(["dwi2tensor", dwi, "-grad", b, tensor, "-force"], check=True)
        subprocess.run(["dwi2cond", f"--aniso", sub.replace("sub-", ""), tensor], cwd=f"{OUT}/{sub}", check=True)
    # 3) EEG leadfield — CONFIRM against the installed SimNIBS version's API
    #    from simnibs import sim_struct, run_simnibs
    #    lf = sim_struct.TDCSLEADFIELD(); lf.subpath = m2m
    #    lf.eeg_cap = <EGI-128 cap csv from emeg_fm/montage.py positions>
    #    lf.pathfem = f"{OUT}/{sub}/leadfield"; lf.anisotropy_type = "vn" if dwi else "scalar"
    #    run_simnibs(lf)
    print(f"[ok {sub}] CHARM (+ dwi2cond) done; wire the leadfield call (see CONFIRM block)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not have_simnibs() and not a.dry_run:
        print("SimNIBS is not installed (source checkout only at ~/dev/simnibs). Install it "
              "(pip install the checkout / official installer), then re-run. Use --dry-run to check inputs.")
        return
    print(f"tier-3 forward-model prototype over {len(SUBJECTS)} subjects -> {OUT}")
    for s in SUBJECTS:
        try:
            run_subject(s, a.dry_run)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL {s}] {e}", flush=True)


if __name__ == "__main__":
    main()
