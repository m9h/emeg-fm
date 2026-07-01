"""Phase 3 (host): fair REVE-vs-band-power comparison at predicting occipital BOLD.

The Calhas "two strategies" question -- does a frozen EEG foundation-model embedding beat
hand-crafted band-power? Both feature sets are HRF-convolved, then scored by an IDENTICAL
metric: 5-fold blocked ridge CV predicted-vs-true r, significance via a circular-shift null
of the CV pipeline (removes the shared autocorrelation bias), z-scored, combined across
subjects by nested Stouffer. REVE (512-d) is PCA-reduced to N_PC before ridge. Higher
Stouffer Z = more predictive skill above the autocorrelation baseline. Reports each
strategy's Z and the paired REVE-minus-BP Z. Usage: natview_reve_compare.py
"""
import glob
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from emeg_fm import natview as nv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natview_eeg_to_bold import TR  # noqa: E402

OUT = "/mnt/t9/natview_reve"
N_NULL = 100
N_PC = 30
RNG = np.random.default_rng(0)


def cv_pred_r(X, y):
    pred = np.zeros_like(y)
    for tr_i, te in KFold(5, shuffle=False).split(X):
        pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr_i], y[tr_i]).predict(X[te])
    return float(np.corrcoef(pred, y)[0, 1])


def impute_nan(X):
    """Replace NaN feature rows (e.g. an empty trigger window) with the column mean."""
    X = np.array(X, float)
    if np.isnan(X).any():
        col = np.nanmean(X, axis=0)
        col = np.where(np.isfinite(col), col, 0.0)
        bad = np.where(np.isnan(X))
        X[bad] = np.take(col, bad[1])
    return X


def design_z(X, y, n):
    X = impute_nan(X)
    h = nv.hrf(TR)
    Xc = np.column_stack([np.convolve(X[:, j], h)[:n] for j in range(X.shape[1])])
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
    if Xc.shape[1] > N_PC:
        Xc = PCA(n_components=N_PC, random_state=0).fit_transform(Xc)
        Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
    r = cv_pred_r(Xc, y)
    null = np.array([cv_pred_r(np.roll(Xc, RNG.integers(10, n - 10), axis=0), y)
                     for _ in range(N_NULL)])
    return r, float((r - null.mean()) / (null.std() + 1e-12))


def stouffer(by_sub):
    subj = np.array([np.sum(v) / np.sqrt(len(v)) for v in by_sub.values()])
    return subj.sum() / np.sqrt(len(subj)), subj


def main():
    rows = []
    for pth in sorted(glob.glob(f"{OUT}/prep_*.npz")):
        key = os.path.basename(pth)[len("prep_"):-len(".npz")]
        emb_p = f"{OUT}/emb_{key}.npz"
        if not os.path.exists(emb_p):
            continue
        pz = np.load(pth, allow_pickle=True)
        ez = np.load(emb_p, allow_pickle=True)
        y, bp, emb = pz["y"].astype(float), pz["bp"].astype(float), ez["reve_emb"].astype(float)
        n = min(len(y), len(bp), len(emb))
        if n < 50:
            continue
        y, bp, emb = y[:n], bp[:n], emb[:n]
        rb, zb = design_z(bp, y, n)
        rr, zr = design_z(emb, y, n)
        rows.append((key.split("_")[0], key, zb, zr, rb, rr))
        print(f"  {key}  BP z={zb:+.2f} r={rb:+.3f} | REVE z={zr:+.2f} r={rr:+.3f}", flush=True)

    bp_s, re_s, dz_s = defaultdict(list), defaultdict(list), defaultdict(list)
    for sub, _, zb, zr, _, _ in rows:
        bp_s[sub].append(zb)
        re_s[sub].append(zr)
        dz_s[sub].append(zr - zb)
    Zb, _ = stouffer(bp_s)
    Zr, _ = stouffer(re_s)
    Zd, subjd = stouffer(dz_s)

    print(f"\n=== natview REVE(layer6) vs band-power -> occipital BOLD "
          f"({len(rows)} pairs / {len(bp_s)} subj) ===")
    print("higher Stouffer Z = more predictive skill above the autocorrelation baseline")
    print(f"  band-power   : Stouffer Z = {Zb:+.2f}  p = {stats.norm.sf(abs(Zb)) * 2:.4g}")
    print(f"  REVE  frozen : Stouffer Z = {Zr:+.2f}  p = {stats.norm.sf(abs(Zr)) * 2:.4g}")
    print(f"  REVE - BP    : Stouffer Z = {Zd:+.2f}  p = {stats.norm.sf(abs(Zd)) * 2:.4g} "
          f"| REVE>BP in {int((subjd > 0).sum())}/{len(subjd)} subj")
    np.savez("/mnt/t9/natview_reve_compare.npz",
             keys=np.array([r[1] for r in rows]),
             zb=np.array([r[2] for r in rows]), zr=np.array([r[3] for r in rows]))
    print("[saved] /mnt/t9/natview_reve_compare.npz", flush=True)


if __name__ == "__main__":
    main()
