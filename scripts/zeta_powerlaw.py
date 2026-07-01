"""Rigorous power-law (zeta) test on the 4S structural connectomes — Clauset/KS, not a slope.

For each subject's connectome we fit a power-law (MLE alpha + KS-optimal xmin) to the two candidate
"data distributions" and, critically, run a likelihood-ratio test vs **lognormal** — because a heavy
tail is not a power law, and connectome weights are classically lognormal. Reports, across subjects,
the alpha distribution and the FRACTION where power-law is actually favored over lognormal (R>0,
p<0.1). This is the verdict the descriptive top-50 slope cannot give.

  eigenvalue spectrum  -> the HT-SR / data-spectrum analog of the EEG embedding singular values
  edge-weight dist     -> is the connectivity itself power-law or (more likely) lognormal

    python scripts/zeta_powerlaw.py [RES]
"""
import glob
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import powerlaw  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else "456"
files = sorted(glob.glob(f"/data/derivatives/volume_conduction/connectomes/sub-*/connectome_{RES}.csv"))


def fit_one(x, discrete):
    f = powerlaw.Fit(x, discrete=discrete, verbose=False)
    R, p = f.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    return f.alpha, f.xmin, R, p


ev_a, ev_fav, ew_a, ew_fav, n = [], 0, [], 0, 0
for fpath in files:
    C = np.loadtxt(fpath, delimiter=",")
    if C.ndim != 2 or C.sum() <= 0:
        continue
    C = (C + C.T) / 2.0
    ew = C[np.triu_indices(C.shape[0], 1)]; ew = ew[ew > 0]
    if len(ew) < 100:
        continue
    w = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]; w = w[w > 1e-9]
    a, _, R, p = fit_one(w, discrete=False)        # eigenvalue spectrum tail
    ev_a.append(a); ev_fav += int(R > 0 and p < 0.1)
    aw, _, Rw, pw = fit_one(ew, discrete=True)     # edge-weight distribution
    ew_a.append(aw); ew_fav += int(Rw > 0 and pw < 0.1)
    n += 1

if n == 0:
    print("no usable connectomes yet"); sys.exit(0)
md = lambda v: f"{np.median(v):.2f} [{np.percentile(v,25):.2f},{np.percentile(v,75):.2f}]"
print(f"rigorous power-law (Clauset) over n={n} connectomes (RES={RES}):\n")
print(f"  EIGENVALUE spectrum : alpha {md(ev_a)}   power-law favored over lognormal: {ev_fav}/{n} "
      f"({100*ev_fav/n:.0f}%)")
print(f"  EDGE-WEIGHT dist    : alpha {md(ew_a)}   power-law favored over lognormal: {ew_fav}/{n} "
      f"({100*ew_fav/n:.0f}%)")
print(f"\n  EEG REVE anchor: singular-value spectrum was ~power-law, slope -2.2 (alpha~2, HT-SR critical).")
verdict = "power-law/zeta" if ev_fav > 0.5 * n else "heavy-tailed but NOT power-law (lognormal-like)"
print(f"  -> structural connectome eigenvalue spectrum is: {verdict}")
