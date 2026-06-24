"""Tier-3 leadfield summary pipeline — reduce each SimNIBS GM-volume leadfield to a compact descriptor.

Raw leadfields are ~2 GB each (~3.7 TB over the n≈1534 cohort), so they cannot be stored at scale. This
reads every leadfield HDF5 under the forward dir, computes the per-subject descriptor + per-electrode
gain (emeg_fm.leadfield), and saves them to one small npz the cohort-scale tier-3 analysis can hold.

    python scripts/build_leadfield_descriptors.py [--grid 4] [--discard-raw]

`--discard-raw` deletes each 2 GB HDF5 right after its descriptor is computed — the scaling mode (run
the leadfield, summarize, drop the raw field). Without it the raw leadfields are kept (the 5-subject
prototype). Prints the n=5 leadfield-vs-age trend (illustrative; n=5 is far too small for a real test).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from leadfield import leadfield_descriptor                       # noqa: E402
from structural import _participant_ages                         # noqa: E402

OUT = "/data/derivatives/volume_conduction/forward"
DESC_NPZ = "/data/derivatives/volume_conduction/leadfield_descriptors.npz"
PARTICIPANTS = "/data/datasets/hbn-eeg/participants.tsv"
_SID = re.compile(r"sub-([A-Za-z0-9]+)")


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--discard-raw", action="store_true", help="rm each HDF5 after summarizing (scale mode)")
    ap.add_argument("--out", default=DESC_NPZ)
    a = ap.parse_args()
    grid = (a.grid,) * 3
    ages_by_id = _participant_ages(PARTICIPANTS)

    hdf5s = sorted(glob.glob(f"{OUT}/sub-*/leadfield/*_leadfield_*.hdf5"))
    print(f"{len(hdf5s)} leadfield HDF5(s) -> descriptors (grid={grid}, discard_raw={a.discard_raw})", flush=True)
    ids, X, gains, ages, names = [], [], [], [], None
    for fn in hdf5s:
        sub = _SID.search(fn).group(1)
        r = leadfield_descriptor(fn, grid)
        names = names or r["electrode_names"]
        ids.append(f"sub-{sub}")
        X.append(r["descriptor"]); gains.append(r["gain"])
        ages.append(ages_by_id.get(sub, np.nan))
        print(f"  [{sub}] n_tet={r['n_tet']:>8d}  descriptor={r['descriptor'].size}  "
              f"mean_gain={r['gain'].mean():.3f}  age={ages[-1]}", flush=True)
        if a.discard_raw:
            os.remove(fn); print(f"    discarded raw {os.path.basename(fn)}", flush=True)
    X, gains, ages = np.array(X), np.array(gains), np.array(ages, float)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, X=X, gains=gains, ids=np.array(ids), ages=ages,
             electrode_names=np.array(names), grid=np.array(grid))
    print(f"\nwrote {a.out}  (X {X.shape}, gains {gains.shape})")

    # ---- n-small leadfield-vs-age trend (illustrative) ----
    fin = np.isfinite(ages)
    if fin.sum() >= 3:
        ag, gn, XX = ages[fin], gains[fin], X[fin]
        order = np.argsort(ag)
        print(f"\n=== leadfield vs age (n={fin.sum()}, ILLUSTRATIVE — far too few for a test) ===")
        print("  ages (sorted):", np.round(ag[order], 1))
        print(f"  mean per-electrode gain vs age: r = {_pearson(ag, gn.mean(1)):+.3f}")
        # strongest per-electrode gain–age correlations
        rs = np.array([_pearson(ag, gn[:, e]) for e in range(gn.shape[1])])
        top = np.argsort(-np.abs(np.nan_to_num(rs)))[:5]
        print("  top |r| electrode gains:",
              ", ".join(f"{names[e]}:{rs[e]:+.2f}" for e in top))
        # descriptor PC1 vs age (shape of GM sensitivity)
        Xc = XX - XX.mean(0)
        if Xc.shape[0] > 1:
            pc1 = np.linalg.svd(Xc, full_matrices=False)[0][:, 0]
            print(f"  descriptor PC1 vs age: r = {_pearson(ag, pc1):+.3f}  "
                  f"(sign arbitrary; PC1 explains the dominant cross-subject leadfield-shape axis)")
        print("  NOTE: n is a feasibility prototype, not a powered tier-3 test. Scale the summary-only")
        print("        pipeline (--discard-raw) over the cohort, then variance-partition vs the EEG age signal.")


if __name__ == "__main__":
    main()
