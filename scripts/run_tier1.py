"""Tier-1 EEG↔anatomy variance partition, end-to-end on HBN. Run AFTER gen_dwi_scalars.py.

    python scripts/run_tier1.py [--grid 8] [--limit 0]

Anatomy A per subject = block-pooled **MNI** GM-probseg (VBM; cross-subject aligned) ⊕ global FA/MD
scalars (DWI microstructure; native-space global/tissue summaries are cross-subject comparable). Runs
`variance_partition` → the headline: % of EEG brain-age that is anatomy-redundant (consistent with
volume conduction) vs EEG-unique. Subject-level CV (rows = subjects). The WM proxy (FA>0.2) is crude —
swap in the qsiprep dseg WM label for a published run.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from structural import reve_embeddings, map_features, assemble   # noqa: E402
from variance_partition import variance_partition               # noqa: E402

QSIPREP = "/data/raw/hbn-qsiprep"
DWI_SCALARS = "/data/derivatives/volume_conduction/dwi_scalars"


def gm_mni(sub: str):
    g = glob.glob(f"{QSIPREP}/{sub}/anat/{sub}_*space-MNI*label-GM_probseg.nii.gz")
    return g[0] if g else None


def dwi_scalars(sub: str):
    import nibabel as nib
    fa, md = f"{DWI_SCALARS}/{sub}/fa.nii.gz", f"{DWI_SCALARS}/{sub}/md.nii.gz"
    if not (os.path.exists(fa) and os.path.exists(md)):
        return None
    FA = np.asarray(nib.load(fa).dataobj, float)
    MD = np.asarray(nib.load(md).dataobj, float)
    brain = FA > 0
    wm = FA > 0.2                                    # crude WM proxy; prefer qsiprep dseg WM label
    return np.array([FA[brain].mean(), FA[brain].std(),
                     FA[wm].mean() if wm.any() else 0.0,
                     MD[brain].mean(), MD[wm].mean() if wm.any() else 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/tier1_variance_partition.json")
    a = ap.parse_args()

    subs, X, ages = reve_embeddings()
    if a.limit:
        subs, X, ages = subs[: a.limit], X[: a.limit], ages[: a.limit]
    grid = (a.grid,) * 3
    anat = {}
    for s in subs:
        g, d = gm_mni(s), dwi_scalars(s)
        if g is None or d is None:
            continue
        anat[s] = np.concatenate([map_features(g, grid), d])
    print(f"anatomy features assembled for {len(anat)}/{len(subs)} EEG subjects", flush=True)

    E, A, y, kept = assemble(subs, X, ages, anat)
    r = variance_partition(E, A, y)
    print(f"\nn={len(kept)}  r2_eeg={r['r2_eeg']:.3f}  r2_anat={r['r2_anat']:.3f}  r2_joint={r['r2_joint']:.3f}")
    print(f"redundant_fraction = {r['redundant_fraction']:.2f}  (EEG age-signal reproducible from anatomy)")
    print(f"eeg_unique_fraction = {r['eeg_unique_fraction']:.2f}  (anatomy cannot reproduce → candidate neural)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({**r, "n": len(kept), "grid": a.grid}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")
    print("\nReminder: this is correlational redundancy, not causal proof of conduction — tier 3 is the causal test.")


if __name__ == "__main__":
    main()
