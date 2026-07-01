"""Group-level natview occipital-alpha -> occipital-BOLD: the statistically proper test.

A single subject (286 TRs) is underpowered for EEG->BOLD. Here we test whether the
trigger-aligned occipital-alpha <-> occipital-BOLD coupling REPLICATES across all natview
pairs, and -- the sharp Valdes-Sosa prediction -- whether its sign is consistently
NEGATIVE (visual alpha suppresses occipital BOLD). Primary statistic: the HRF-convolved
occipital-alpha regression coefficient, tested vs 0 across subjects (independent units)
and pairs. Secondary: held-out CV r (single-subject predictive skill is expected weak).
Usage: natview_group_occipital.py [task]
"""
import glob
import os
import sys
import tempfile
from collections import defaultdict

import mne
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV

from emeg_fm import natview as nv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natview_eeg_to_bold import (  # noqa: E402
    ALPHA_IDX, NV, TR, bandpower_per_tr, clean_bold, cv_r, mcflirt,
    volume_trigger_samples,
)

mne.set_log_level("ERROR")
TASK = sys.argv[1] if len(sys.argv) > 1 else "inscapes"


def all_pairs(task):
    out = []
    for e in sorted(glob.glob(f"{NV}/preproc_data/sub-*/ses-*/eeg/*task-{task}*_eeg.set")):
        p = e.split("/")
        sub = next(x for x in p if x.startswith("sub-"))
        ses = next(x for x in p if x.startswith("ses-"))
        b = sorted(glob.glob(f"{NV}/raw_data/{sub}/{ses}/func/{sub}_{ses}_task-{task}*_bold.nii.gz"))
        if b:
            out.append((sub, ses, e, b[0]))
    return out


def fit_one(eeg_path, bold_path):
    with tempfile.TemporaryDirectory(dir="/mnt/t9") as wd:
        d, par, affine = mcflirt(bold_path, wd)
    occ = nv.occipital_mask(nv.brain_mask(d), affine)
    y = clean_bold(d, occ, par)
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    occ_ch = nv.select_occipital_channels(raw.ch_names)
    if not occ_ch:
        return None
    trig = volume_trigger_samples(eeg_path, raw.info["sfreq"])
    Xb = bandpower_per_tr(raw, trig, d.shape[3], occ_ch)
    n = min(len(y), len(Xb))
    y, Xb = y[:n], Xb[:n]
    if np.isnan(Xb).any():
        return None
    h = nv.hrf(TR)
    Xc = np.column_stack([np.convolve(Xb[:, j], h)[:n] for j in range(Xb.shape[1])])
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
    r, _ = cv_r(Xc, y)
    coef = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xc, y).coef_
    return {"n": n, "r": float(r), "alpha_coef": float(coef[ALPHA_IDX])}


def report(tag, x):
    x = np.asarray(x)
    t, p = stats.ttest_1samp(x, 0.0)
    try:
        _, pw = stats.wilcoxon(x)
    except ValueError:
        pw = float("nan")
    print(f"  {tag}: N={len(x)} mean={x.mean():+.3f} sd={x.std():.3f} | "
          f"neg {int((x < 0).sum())}/{len(x)} | t={t:+.2f} p={p:.4f} | Wilcoxon p={pw:.4f}",
          flush=True)


def main():
    pairs = all_pairs(TASK)
    print(f"[group] {len(pairs)} {TASK} EEG+BOLD pairs", flush=True)
    rows = []
    for i, (sub, ses, e, b) in enumerate(pairs):
        try:
            res = fit_one(e, b)
        except Exception as ex:  # noqa: BLE001 keep the sweep going
            print(f"  [{i + 1}/{len(pairs)}] {sub} {ses}  SKIP ({type(ex).__name__})", flush=True)
            continue
        if res is None:
            print(f"  [{i + 1}/{len(pairs)}] {sub} {ses}  SKIP (no occ ch / NaN)", flush=True)
            continue
        rows.append((sub, ses, res))
        print(f"  [{i + 1}/{len(pairs)}] {sub} {ses}  r={res['r']:+.3f}  "
              f"alpha_coef={res['alpha_coef']:+.3f}", flush=True)

    coefs = np.array([r["alpha_coef"] for _, _, r in rows])
    rs = np.array([r["r"] for _, _, r in rows])
    bysub = defaultdict(list)
    for sub, _, r in rows:
        bysub[sub].append(r["alpha_coef"])
    subj_coefs = np.array([np.mean(v) for v in bysub.values()])

    print(f"\n=== GROUP occipital alpha -> occipital BOLD  (natview {TASK}, trigger-aligned) ===")
    print(f"usable {len(rows)}/{len(pairs)} pairs, {len(bysub)} subjects | "
          f"Valdes-Sosa predicts coef < 0 (visual alpha suppresses occipital BOLD)")
    report("alpha_coef / subject", subj_coefs)   # independent units -> the headline test
    report("alpha_coef / pair   ", coefs)
    report("held-out CV r / pair ", rs)
    np.savez("/mnt/t9/natview_group_occipital.npz", coefs=coefs, rs=rs,
             subj_coefs=subj_coefs, subs=[s for s, _, _ in rows], ses=[s for _, s, _ in rows])
    print("[saved] /mnt/t9/natview_group_occipital.npz", flush=True)


if __name__ == "__main__":
    main()
