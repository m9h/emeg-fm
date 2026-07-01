"""Faithful group test of the Valdes-Sosa occipital alpha<->BOLD coupling on natview.

The 5-band ridge coefficient is collinearity-fragile; the specific Valdes-Sosa claim is
UNIVARIATE -- occipital alpha POWER covaries NEGATIVELY with occipital BOLD. Per pair we
correlate HRF-convolved occipital-alpha power with occipital BOLD, build a per-pair
circular-shift null (accounts for EEG/BOLD autocorrelation), z-score, and combine across
SUBJECTS (independent units) with a nested Stouffer's Z (within-subject then across).
Also reports the 5-band ridge held-out r for context. Trigger-aligned (R128).
Usage: natview_group_alpha_bold.py [task] [n_null]
"""
import glob
import os
import sys
import tempfile
from collections import defaultdict

import mne
import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt, hilbert
from sklearn.linear_model import RidgeCV  # noqa: F401  (via base cv_r)

from emeg_fm import natview as nv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natview_eeg_to_bold import (  # noqa: E402
    NV, TR, bandpower_per_tr, clean_bold, cv_r, mcflirt, volume_trigger_samples,
)

mne.set_log_level("ERROR")
TASK = sys.argv[1] if len(sys.argv) > 1 else "inscapes"
N_NULL = int(sys.argv[2]) if len(sys.argv) > 2 else 200
RNG = np.random.default_rng(0)


def occ_alpha_power(raw, occ_ch, trig, n_tr):
    sf = raw.info["sfreq"]
    x = raw.get_data()[occ_ch]
    b, a = butter(4, [8 / (sf / 2), 13 / (sf / 2)], btype="band")
    env = np.abs(hilbert(filtfilt(b, a, x, axis=1), axis=1)) ** 2
    return nv.bin_by_triggers(env.mean(0), trig, n_tr)


def fit_pair(eeg_path, bold_path):
    with tempfile.TemporaryDirectory(dir="/mnt/t9") as wd:
        d, par, affine = mcflirt(bold_path, wd)
    y = clean_bold(d, nv.occipital_mask(nv.brain_mask(d), affine), par)
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    occ_ch = nv.select_occipital_channels(raw.ch_names)
    if not occ_ch:
        return None
    trig = volume_trigger_samples(eeg_path, raw.info["sfreq"])
    a = occ_alpha_power(raw, occ_ch, trig, d.shape[3])
    n = min(len(y), len(a))
    y, a = y[:n], a[:n]
    if np.isnan(a).any():
        return None
    h = nv.hrf(TR)
    ac = np.convolve(a, h)[:n]
    ac = (ac - ac.mean()) / (ac.std() + 1e-8)
    r = float(np.corrcoef(ac, y)[0, 1])
    null = np.array([np.corrcoef(np.roll(ac, RNG.integers(10, n - 10)), y)[0, 1]
                     for _ in range(N_NULL)])
    z = float((r - null.mean()) / (null.std() + 1e-12))
    Xb = bandpower_per_tr(raw, trig, d.shape[3], occ_ch)[:n]
    Xc = np.column_stack([np.convolve(Xb[:, j], h)[:n] for j in range(Xb.shape[1])])
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
    ridge_r, _ = cv_r(Xc, y)
    return {"n": n, "r_alpha": r, "z": z, "ridge_r": float(ridge_r)}


def main():
    pairs = []
    for e in sorted(glob.glob(f"{NV}/preproc_data/sub-*/ses-*/eeg/*task-{TASK}*_eeg.set")):
        p = e.split("/")
        sub = next(x for x in p if x.startswith("sub-"))
        ses = next(x for x in p if x.startswith("ses-"))
        b = sorted(glob.glob(f"{NV}/raw_data/{sub}/{ses}/func/{sub}_{ses}_task-{TASK}*_bold.nii.gz"))
        if b:
            pairs.append((sub, ses, e, b[0]))
    print(f"[group] {len(pairs)} {TASK} pairs, {N_NULL} circular nulls/pair", flush=True)

    rows = []
    for i, (sub, ses, e, b) in enumerate(pairs):
        try:
            res = fit_pair(e, b)
        except Exception as ex:  # noqa: BLE001
            print(f"  [{i + 1}/{len(pairs)}] {sub} {ses} SKIP({type(ex).__name__})", flush=True)
            continue
        if res is None:
            print(f"  [{i + 1}/{len(pairs)}] {sub} {ses} SKIP(no occ/NaN)", flush=True)
            continue
        rows.append((sub, ses, res))
        print(f"  [{i + 1}/{len(pairs)}] {sub} {ses} r_alpha={res['r_alpha']:+.3f} "
              f"z={res['z']:+.2f} ridge_r={res['ridge_r']:+.3f}", flush=True)

    bysub_r, bysub_z = defaultdict(list), defaultdict(list)
    for sub, _, res in rows:
        bysub_r[sub].append(res["r_alpha"])
        bysub_z[sub].append(res["z"])
    subs = sorted(bysub_r)
    sr = np.array([np.mean(bysub_r[s]) for s in subs])
    z_subj = np.array([np.sum(bysub_z[s]) / np.sqrt(len(bysub_z[s])) for s in subs])  # within
    stouffer = z_subj.sum() / np.sqrt(len(z_subj))                                     # across
    p_neg = stats.norm.cdf(stouffer)   # one-sided: NEGATIVE = alpha suppresses BOLD
    t, pt = stats.ttest_1samp(sr, 0.0)
    try:
        _, pw = stats.wilcoxon(sr)
    except ValueError:
        pw = float("nan")

    print(f"\n=== GROUP occipital ALPHA POWER -> occipital BOLD  (natview {TASK}) ===")
    print(f"{len(rows)} pairs / {len(subs)} subjects | Valdes-Sosa predicts NEGATIVE coupling")
    print(f"per-subject r_alpha: mean={sr.mean():+.3f} sd={sr.std():.3f} | "
          f"neg {int((sr < 0).sum())}/{len(sr)} | t={t:+.2f} p2={pt:.4f} | Wilcoxon p={pw:.4f}")
    print(f"nested Stouffer Z = {stouffer:+.2f}  one-sided p(negative) = {p_neg:.4g}")
    print(f"multivariate 5-band ridge held-out r: mean="
          f"{np.mean([res['ridge_r'] for _, _, res in rows]):+.3f}")
    np.savez("/mnt/t9/natview_group_alpha_bold.npz", subs=np.array(subs), sr=sr, z_subj=z_subj,
             pair_r=np.array([res["r_alpha"] for _, _, res in rows]),
             pair_z=np.array([res["z"] for _, _, res in rows]))
    print("[saved] /mnt/t9/natview_group_alpha_bold.npz", flush=True)


if __name__ == "__main__":
    main()
