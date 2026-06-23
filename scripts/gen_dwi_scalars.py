"""Generate per-subject DWI FA/MD from HBN qsiprep (mrtrix 3.x). Cluster job, ~1-3 min/subject.

    python scripts/gen_dwi_scalars.py --dry-run        # check input discovery first
    python scripts/gen_dwi_scalars.py [--limit N]      # then generate

Writes <OUT>/sub-XXX/{fa,md}.nii.gz (idempotent — skips completed subjects). The glob patterns and the
mrtrix `.b` / mask filenames are best-effort against the verified qsiprep layout; confirm with
`--dry-run` on the first pass and adjust if a subject's files differ.
"""
import argparse
import glob
import os
import subprocess

QSIPREP = "/data/raw/hbn-qsiprep"
OUT = "/data/derivatives/volume_conduction/dwi_scalars"


def find_inputs(subdir: str):
    dwi = glob.glob(f"{subdir}/ses-*/dwi/*acq-64dir*space-T1w*desc-preproc_dwi.nii.gz")
    if not dwi:
        return None
    dwi = dwi[0]
    d = os.path.dirname(dwi)
    base = dwi[: -len(".nii.gz")]
    grad = base + ".b" if os.path.exists(base + ".b") else next(iter(glob.glob(f"{d}/*preproc_dwi.b")), None)
    mask = next(iter(glob.glob(f"{d}/*desc-brain_mask.nii.gz") or glob.glob(f"{d}/*mask.nii.gz")), None)
    return dwi, grad, mask


def run(subdir: str, dry: bool) -> str:
    sub = os.path.basename(subdir)
    inp = find_inputs(subdir)
    if not inp or not inp[1]:
        return f"[skip {sub}] no preproc dwi / mrtrix .b grad table"
    dwi, grad, mask = inp
    od = f"{OUT}/{sub}"
    fa, md = f"{od}/fa.nii.gz", f"{od}/md.nii.gz"
    if os.path.exists(fa) and os.path.exists(md):
        return f"[done {sub}]"
    if dry:
        return f"[dry {sub}] dwi={os.path.basename(dwi)} grad={'y' if grad else 'NO'} mask={'y' if mask else 'no'}"
    os.makedirs(od, exist_ok=True)
    tn = f"{od}/tensor.mif"
    cmd = ["dwi2tensor", dwi, "-grad", grad, tn, "-force"] + (["-mask", mask] if mask else [])
    subprocess.run(cmd, check=True)
    subprocess.run(["tensor2metric", tn, "-fa", fa, "-adc", md, "-force"], check=True)
    os.remove(tn)
    return f"[ok {sub}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    subs = sorted(s for s in glob.glob(f"{QSIPREP}/sub-*") if os.path.isdir(s))   # skip .html report files
    if a.limit:
        subs = subs[: a.limit]
    print(f"{len(subs)} qsiprep subjects -> {OUT}")
    for s in subs:
        try:
            print(run(s, a.dry_run), flush=True)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL {os.path.basename(s)}] {e}", flush=True)


if __name__ == "__main__":
    main()
