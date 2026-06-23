"""Tier-2: build the per-subject joint VBM/DWI structural embedding over the EEG cohort and save a
reusable npz, so E1/E2/E4 can consume the structural side. Run AFTER gen_dwi_scalars.py.

    python scripts/build_structural_embedding.py [--grid 8] [--limit 0]

Feature per subject = block-pooled MNI GM-probseg (VBM, cross-subject aligned) ⊕ global FA/MD scalars
(DWI). Saves X (n×d) + ids (subject order) to <OUT>.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from structural import reve_embeddings, subject_structural_features   # noqa: E402

OUT = "/data/derivatives/volume_conduction/structural_emb.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    subs, _, _ = reve_embeddings()                      # EEG cohort = the subject universe
    if a.limit:
        subs = subs[: a.limit]
    grid = (a.grid,) * 3
    ids, feats = [], []
    for i, s in enumerate(subs):
        v = subject_structural_features(s, grid)
        if v is not None:
            ids.append(s); feats.append(v)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(subs)}  kept={len(ids)}", flush=True)
    if not feats:
        print("no structural features built — did you run gen_dwi_scalars.py? are GM-probsegs present?")
        return
    X = np.vstack(feats)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, X=X, ids=np.array(ids))
    print(f"structural embedding: {len(ids)} subjects × {X.shape[1]} features -> {a.out}")


if __name__ == "__main__":
    main()
