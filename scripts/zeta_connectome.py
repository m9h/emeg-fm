"""Cross-subject zeta verdict on 4S structural connectomes — heavy-tailed (zeta) vs spiked.

For each subject's connectome (NxN streamline-count matrix) we take the eigenvalue spectrum of the
symmetric weighted adjacency and the edge-weight distribution, fit log-log tail slopes, and measure
the top-mode fraction. Then we aggregate the DISTRIBUTION across subjects and compare to the two
anchors we already have: the morphometry covariance (spiked: top-mode 40-76%, ~2-5% above MP edge,
slope ~-0.7) and the EEG REVE embeddings (zeta: slope ~-2.2, alpha~2). This is the structural object
that — unlike morphometry — has a real shot at zeta.

    python scripts/zeta_connectome.py [RES]   # RES default 456
"""
import glob
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "456"
files = sorted(glob.glob(f"/data/derivatives/volume_conduction/connectomes/sub-*/connectome_{RES}.csv"))
print(f"connectomes found: {len(files)}  (RES={RES})")
if not files:
    sys.exit(0)

eig_slope, w_slope, top_frac, density, wtail = [], [], [], [], []
skipped = []
for f in files:
    sid = f.split("/")[-2]
    C = np.loadtxt(f, delimiter=",")
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.sum() <= 0:
        skipped.append(sid); continue
    C = (C + C.T) / 2.0
    p = C.shape[0]
    nz = np.sort(C[np.triu_indices(p, 1)])[::-1]
    nz = nz[nz > 0]
    if len(nz) < 100:                       # degenerate connectome (failed warp/track) -> skip
        skipped.append(sid); continue
    # eigenvalue spectrum of the weighted adjacency
    w = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]
    N = min(50, p // 2)
    eig_slope.append(np.polyfit(np.log(np.arange(1, N + 1)), np.log(w[:N] + 1e-9), 1)[0])
    top_frac.append(w[0] / w.sum())
    density.append((C > 0).mean())
    # edge-weight distribution (rank-ordered nonzero upper-triangle weights)
    k = min(len(nz), max(10, len(nz) // 3))
    w_slope.append(np.polyfit(np.log(np.arange(1, k + 1)), np.log(nz[:k]), 1)[0])
    wtail.append(nz.max() / np.median(nz))
if skipped:
    print(f"  (skipped {len(skipped)} degenerate/empty connectomes: {', '.join(skipped[:6])}"
          f"{'...' if len(skipped) > 6 else ''})")


def stat(a):
    a = np.asarray(a); return f"{np.median(a):+.2f} [{np.percentile(a,25):+.2f},{np.percentile(a,75):+.2f}]"


print(f"\nstructural connectome ESD across {len(eig_slope)} subjects (median [IQR]):")
print(f"  eigenvalue top-mode fraction : {stat(top_frac)}   (morphometry spiked: 0.40-0.76)")
print(f"  eigenvalue top-50 log-log slope: {stat(eig_slope)}   (EEG REVE zeta: ~-2.2; morphometry: ~-0.7)")
print(f"  edge-weight top-third slope   : {stat(w_slope)}")
print(f"  edge-weight max/median ratio  : {stat(wtail)}   (heavy-tailed weights if >>1)")
print(f"  connectome density            : {stat(density)}")
hv = np.median(top_frac) < 0.15
print(f"\n  → top eigenmode does NOT dominate (median {np.median(top_frac):.2f}): "
      f"{'connectome spectrum is spread/heavy-tailed — qualitatively UNLIKE the spiked morphometry' if hv else 'spiked-like'}")
