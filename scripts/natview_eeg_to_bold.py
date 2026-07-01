"""Within-subject EEG->BOLD prediction on natview (simultaneous EEG-fMRI).

Composes the unit-tested emeg_fm.natview primitives + FSL mcflirt motion correction.
Two tests: (1) all-channel band-power -> motion/drift-cleaned GLOBAL BOLD; (2) the classic
OCCIPITAL-channel band-power -> OCCIPITAL BOLD (where the robust alpha<->visual coupling
lives). Temporal ridge CV + circular-shift null; reports the alpha coefficient sign.
Band-power baseline; the frozen-REVE-embedding FM comparison is next. Usage: [task]
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import mne
import nibabel as nib
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from emeg_fm import natview as nv

mne.set_log_level("ERROR")
FSL = os.environ.get("FSLDIR", "/home/mhough/fsl")
NV = "/data/datasets/eeg_fmri/natview"
TASK = sys.argv[1] if len(sys.argv) > 1 else "inscapes"
TR = 2.1
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
ALPHA_IDX = list(BANDS).index("alpha")
RNG = np.random.default_rng(0)


def find_pair():
    for eeg in sorted(glob.glob(f"{NV}/preproc_data/sub-*/ses-*/eeg/*task-{TASK}*_eeg.set")):
        p = eeg.split("/")
        sub = next(x for x in p if x.startswith("sub-"))
        ses = next(x for x in p if x.startswith("ses-"))
        bolds = glob.glob(f"{NV}/raw_data/{sub}/{ses}/func/{sub}_{ses}_task-{TASK}*_bold.nii.gz")
        if bolds:
            return sub, ses, eeg, sorted(bolds)[0]
    return None


def mcflirt(bold_path, workdir):
    local = os.path.join(workdir, "bold.nii.gz")
    shutil.copy(bold_path, local)
    out = os.path.join(workdir, "bold_mc")
    subprocess.run([f"{FSL}/bin/mcflirt", "-in", local, "-out", out, "-plots"],
                   check=True, env={**os.environ, "FSLDIR": FSL, "FSLOUTPUTTYPE": "NIFTI_GZ"},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    img = nib.load(out + ".nii.gz")
    return img.get_fdata(), np.loadtxt(out + ".par"), img.affine


def bandpower_per_tr(raw, n_tr, ch_idx=None):
    x = raw.get_data()
    if ch_idx is not None:
        x = x[ch_idx]
    sf = raw.info["sfreq"]
    feats = []
    for lo, hi in BANDS.values():
        b, a = butter(4, [lo / (sf / 2), hi / (sf / 2)], btype="band")
        env = np.abs(hilbert(filtfilt(b, a, x, axis=1), axis=1)) ** 2
        feats.append(nv.bin_to_tr(env.mean(0), sf, TR, n_tr))
    m = min(len(f) for f in feats)
    return np.column_stack([f[:m] for f in feats])


def clean_bold(d, mask, par):
    g = nv.roi_timecourse(d, mask)
    t = np.arange(d.shape[3])
    y = nv.regress_confounds(g, np.column_stack([par, t, t ** 2]))
    return (y - y.mean()) / (y.std() + 1e-8)


def cv_r(X, y):
    pred = np.zeros_like(y)
    for tr_i, te in KFold(5, shuffle=False).split(X):
        pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr_i], y[tr_i]).predict(X[te])
    return float(np.corrcoef(pred, y)[0, 1]), r2_score(y, pred)


def evaluate(name, Xb, y):
    n = min(len(y), len(Xb))
    y, Xb = y[:n], Xb[:n]
    h = nv.hrf(TR)
    Xc = np.column_stack([np.convolve(Xb[:, j], h)[:n] for j in range(Xb.shape[1])])
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
    r, r2 = cv_r(Xc, y)
    null = np.array([cv_r(np.roll(Xc, RNG.integers(10, n - 10), axis=0), y)[0] for _ in range(500)])
    p = (np.sum(null >= r) + 1) / (len(null) + 1)
    coef = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xc, y).coef_
    print(f"  [{name}] n={n}  r={r:+.3f} R2={r2:+.3f} | null {null.mean():+.3f}±{null.std():.3f} "
          f"p={p:.4f} | alpha_coef={coef[ALPHA_IDX]:+.3f}", flush=True)


def main():
    pair = find_pair()
    if pair is None:
        sys.exit(f"no EEG+BOLD pair for task={TASK}")
    sub, ses, eeg_path, bold_path = pair
    print(f"[pair] {sub} {ses} task-{TASK}", flush=True)
    with tempfile.TemporaryDirectory(dir="/mnt/t9") as wd:
        print("[fmri] mcflirt ...", flush=True)
        d, par, affine = mcflirt(bold_path, wd)
    brain = nv.brain_mask(d)
    occ = nv.occipital_mask(brain, affine)
    y_glob = clean_bold(d, brain, par)
    y_occ = clean_bold(d, occ, par)
    print(f"[fmri] {d.shape[3]} TRs | brain {int(brain.sum())} vox, occipital {int(occ.sum())} vox",
          flush=True)

    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    occ_ch = nv.select_occipital_channels(raw.ch_names)
    print(f"[eeg] {len(raw.ch_names)}ch @ {raw.info['sfreq']}Hz | occipital ch: "
          f"{[raw.ch_names[i] for i in occ_ch]}", flush=True)
    Xb_all = bandpower_per_tr(raw, d.shape[3])
    print(f"\n=== natview EEG->BOLD  {sub} {ses} task-{TASK} ===")
    evaluate("all-ch -> GLOBAL", Xb_all, y_glob)
    if occ_ch:
        Xb_occ = bandpower_per_tr(raw, d.shape[3], occ_ch)
        evaluate("occ-ch -> OCCIPITAL", Xb_occ, y_occ)   # classic alpha<->visual test
    else:
        print("  (no occipital channels matched -> skipping occipital test)")


if __name__ == "__main__":
    main()
