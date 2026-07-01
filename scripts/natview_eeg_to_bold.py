"""Within-subject EEG->BOLD prediction on natview (simultaneous EEG-fMRI).

Composes the unit-tested emeg_fm.natview primitives with FSL mcflirt motion correction:
raw BOLD -> mcflirt -> global timecourse with motion (6 params) + drift regressed out
-> predict from HRF-convolved EEG band-power, temporal ridge CV + circular-shift null.
This is the VALID band-power baseline (the crude first pass was motion-dominated); the
frozen-REVE-embedding FM comparison is the follow-up. Usage: natview_eeg_to_bold.py [task]
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
    """FSL mcflirt -> (motion-corrected 4D array, motion params [T,6])."""
    local = os.path.join(workdir, "bold.nii.gz")
    shutil.copy(bold_path, local)
    out = os.path.join(workdir, "bold_mc")
    subprocess.run([f"{FSL}/bin/mcflirt", "-in", local, "-out", out, "-plots"],
                   check=True, env={**os.environ, "FSLDIR": FSL, "FSLOUTPUTTYPE": "NIFTI_GZ"},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = nib.load(out + ".nii.gz").get_fdata()
    par = np.loadtxt(out + ".par")            # (T,6) rot xyz + trans xyz
    return d, par


def eeg_bandpower_per_tr(raw, n_tr):
    x = raw.get_data()
    sf = raw.info["sfreq"]
    feats = []
    for lo, hi in BANDS.values():
        b, a = butter(4, [lo / (sf / 2), hi / (sf / 2)], btype="band")
        env = np.abs(hilbert(filtfilt(b, a, x, axis=1), axis=1)) ** 2
        feats.append(nv.bin_to_tr(env.mean(0), sf, TR, n_tr))
    m = min(len(f) for f in feats)
    return np.column_stack([f[:m] for f in feats])


def cv_r(X, y):
    pred = np.zeros_like(y)
    for tr_i, te in KFold(5, shuffle=False).split(X):
        pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr_i], y[tr_i]).predict(X[te])
    return float(np.corrcoef(pred, y)[0, 1]), r2_score(y, pred)


def main():
    pair = find_pair()
    if pair is None:
        sys.exit(f"no EEG+BOLD pair for task={TASK}")
    sub, ses, eeg_path, bold_path = pair
    print(f"[pair] {sub} {ses} task-{TASK}", flush=True)

    with tempfile.TemporaryDirectory(dir="/mnt/t9") as wd:
        print("[fmri] mcflirt motion correction ...", flush=True)
        d, par = mcflirt(bold_path, wd)
    n_tr = d.shape[3]
    # global timecourse, motion (6 params) + quadratic drift regressed out (tested prims)
    mask = nv.brain_mask(d)
    g = nv.roi_timecourse(d, mask)
    t = np.arange(n_tr)
    confounds = np.column_stack([par, t, t ** 2])                 # motion + drift
    y = nv.regress_confounds(g, confounds)
    y = (y - y.mean()) / (y.std() + 1e-8)
    print(f"[fmri] {n_tr} TRs, {int(mask.sum())} brain voxels, motion+drift regressed", flush=True)

    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    Xb = eeg_bandpower_per_tr(raw, n_tr)
    n = min(len(y), len(Xb))
    y, Xb = y[:n], Xb[:n]
    print(f"[eeg] band-power {Xb.shape} ({raw.info['sfreq']}Hz, {len(raw.ch_names)}ch)", flush=True)

    h = nv.hrf(TR)
    Xc = np.column_stack([np.convolve(Xb[:, j], h)[:n] for j in range(Xb.shape[1])])
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)

    r, r2 = cv_r(Xc, y)
    null = np.array([cv_r(np.roll(Xc, RNG.integers(10, n - 10), axis=0), y)[0] for _ in range(500)])
    p = (np.sum(null >= r) + 1) / (len(null) + 1)
    print("\n=== natview EEG->BOLD (band-power, motion-corrected, within-subject) ===")
    print(f"  {sub} {ses} task-{TASK}  n={n} TRs")
    print(f"  global-BOLD (motion+drift-regressed) prediction: r={r:.3f}  R2={r2:.3f}")
    print(f"  circular-shift null: mean r={null.mean():.3f} sd={null.std():.3f}  ->  p={p:.4f}")


if __name__ == "__main__":
    main()
