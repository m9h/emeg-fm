"""Compare the surface-source vs GM-volume EEG leadfields per subject (tier-3 sanity / proxy check).

Both representations exist for the prototype subjects: the GM-*volume* leadfield (cheap, computable for
all subjects on aarch64) and the central-GM-*surface* leadfield (the standard EEG source space, but it
needs x86 FastSurfer surfaces via `charm --fs-dir`). This asks: does the volume fallback capture the same
**per-electrode cortical coupling** as the surface gold standard? For each electrode we take the RMS field
magnitude over the ROI (its "gain"), then correlate the 75-electrode gain vectors (Pearson + rank).

    python scripts/compare_surface_volume.py
"""
import glob
import json
import os

import h5py
import numpy as np

SURF = "/data/derivatives/volume_conduction/surface_forward"
VOL = "/data/derivatives/volume_conduction/forward"


def _gain(path: str) -> dict:
    """Per-electrode RMS leadfield magnitude over the ROI -> {electrode_name: gain}."""
    with h5py.File(path, "r") as f:
        d = f["mesh_leadfield/leadfields/tdcs_leadfield"]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in d.attrs["electrode_names"]]
        L = d[:]                                          # (n_elec, n_pts, 3)
    return dict(zip(names, np.sqrt((L ** 2).sum(2).mean(1))))


def _lf(root: str, sub: str):
    g = glob.glob(f"{root}/{sub}/leadfield/*_leadfield_*.hdf5")
    return g[0] if g else None


def main():
    subs = sorted(os.path.basename(d) for d in glob.glob(f"{SURF}/sub-*")
                  if _lf(SURF, os.path.basename(d)) and _lf(VOL, os.path.basename(d)))
    rows, rs = [], []
    print(f"{'subject':18s} {'n_elec':>6s} {'Pearson':>8s} {'Spearman':>9s}")
    for s in subs:
        gs, gv = _gain(_lf(SURF, s)), _gain(_lf(VOL, s))
        common = [e for e in gs if e in gv]
        a = np.array([gs[e] for e in common]); b = np.array([gv[e] for e in common])
        r = float(np.corrcoef(a, b)[0, 1])
        rho = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
        rows.append({"subject": s, "n_elec": len(common), "pearson": r, "spearman": rho}); rs.append(r)
        print(f"{s:18s} {len(common):6d} {r:8.3f} {rho:9.3f}")
    summary = {"n_subjects": len(rows), "mean_pearson": float(np.mean(rs)) if rs else float("nan"),
               "per_subject": rows}
    print(f"\nmean Pearson r = {summary['mean_pearson']:.3f} over {len(rows)} subjects "
          f"=> GM-volume leadfield is a {'strong' if summary['mean_pearson'] > 0.8 else 'partial'} "
          f"per-electrode proxy for the surface-source leadfield")
    os.makedirs("results", exist_ok=True)
    json.dump(summary, open("results/surface_vs_volume.json", "w"), indent=2)
    print("wrote results/surface_vs_volume.json")


if __name__ == "__main__":
    main()
