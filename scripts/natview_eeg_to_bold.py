"""First within-subject EEG->BOLD prediction on natview (simultaneous EEG-fMRI).

The NeuroBOLT/Calhas regime on the newly-landed natview data (/data/datasets/eeg_fmri/
natview): predict the global BOLD timecourse from EEG band-power, HRF-convolved, per TR,
within subject, with a circular-shift null. This is the band-power BASELINE (no GPU);
the frozen-REVE-embedding FM version (does the FM beat band-power?) is the follow-up.

natview: preprocessed EEG (.set) in preproc_data; RAW BOLD (nii.gz, TR=2.1s) in
raw_data. Naturalistic tasks (inscapes movie by default). Usage: natview_eeg_to_bold.py [task]
"""
import glob
import sys

import mne
import nibabel as nib
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

mne.set_log_level("ERROR")
NV = "/data/datasets/eeg_fmri/natview"
TASK = sys.argv[1] if len(sys.argv) > 1 else "inscapes"
TR = 2.1
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
RNG = np.random.default_rng(0)


def canonical_hrf(tr, length=32.0):
    t = np.arange(0, length, tr)
    from scipy.stats import gamma
    return gamma.pdf(t, 6) - 0.35 * gamma.pdf(t, 16)


def band_power_per_tr(raw, n_tr):
    """Filter into bands, Hilbert-envelope power, bin to the TR grid -> (n_tr, n_bands)."""
    x = raw.get_data()                       # (n_chan, n_samples)
    sf = raw.info["sfreq"]
    feats = []
    for lo, hi in BANDS.values():
        b, a = butter(4, [lo / (sf / 2), hi / (sf / 2)], btype="band")
        env = np.abs(hilbert(filtfilt(b, a, x, axis=1), axis=1)) ** 2  # power
        env = env.mean(0)                    # average over channels -> (n_samples,)
        # bin to TR
        spt = int(round(TR * sf))
        usable = (len(env) // spt) * spt
        binned = env[:usable].reshape(-1, spt).mean(1)
        feats.append(binned[:n_tr])
    m = min(len(f) for f in feats)
    return np.column_stack([f[:m] for f in feats])


def global_bold(bold_path):
    img = nib.load(bold_path)
    d = img.get_fdata()                       # (X,Y,Z,T)
    mask = d.mean(3) > 0.15 * d.mean(3).max()  # crude brain mask
    ts = d[mask].mean(0)                       # global mean timecourse (T,)
    ts = np.asarray(ts, float)
    # detrend (linear) + z
    t = np.arange(len(ts)); ts = ts - np.polyval(np.polyfit(t, ts, 2), t)
    return (ts - ts.mean()) / (ts.std() + 1e-8)


def find_pair():
    for eeg in sorted(glob.glob(f"{NV}/preproc_data/sub-*/ses-*/eeg/*task-{TASK}*_eeg.set")):
        parts = eeg.split("/")
        sub = next(p for p in parts if p.startswith("sub-"))
        ses = next(p for p in parts if p.startswith("ses-"))
        bolds = glob.glob(f"{NV}/raw_data/{sub}/{ses}/func/{sub}_{ses}_task-{TASK}*_bold.nii.gz")
        if bolds:
            return sub, ses, eeg, sorted(bolds)[0]
    return None


def main():
    pair = find_pair()
    if pair is None:
        sys.exit(f"no EEG+BOLD pair for task={TASK}")
    sub, ses, eeg_path, bold_path = pair
    print(f"[pair] {sub} {ses} task-{TASK}", flush=True)

    bold = global_bold(bold_path)
    print(f"[bold] {len(bold)} TRs (TR={TR}s)", flush=True)
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    Xb = band_power_per_tr(raw, len(bold))
    n = min(len(bold), len(Xb))
    Xb, y = Xb[:n], bold[:n]
    print(f"[eeg] band-power {Xb.shape} ({raw.info['sfreq']}Hz, {len(raw.ch_names)}ch)", flush=True)

    # HRF-convolve each band, z-score
    hrf = canonical_hrf(TR)
    Xc = np.column_stack([np.convolve(Xb[:, j], hrf)[:n] for j in range(Xb.shape[1])])
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)

    def cv_r(X, yv):
        cv = KFold(5, shuffle=False)  # temporal (no shuffle)
        pred = np.zeros_like(yv)
        for tr_i, te in cv.split(X):
            pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr_i], yv[tr_i]).predict(X[te])
        return float(np.corrcoef(pred, yv)[0, 1]), r2_score(yv, pred)

    r, r2 = cv_r(Xc, y)
    # circular-shift null (shift EEG vs BOLD)
    null = []
    for _ in range(500):
        s = RNG.integers(10, n - 10)
        null.append(cv_r(np.roll(Xc, s, axis=0), y)[0])
    null = np.array(null)
    p = (np.sum(null >= r) + 1) / (len(null) + 1)
    print("\n=== natview EEG->BOLD (band-power, within-subject) ===")
    print(f"  {sub} {ses} task-{TASK}  n={n} TRs")
    print(f"  global-BOLD prediction: r={r:.3f}  R2={r2:.3f}")
    print(f"  circular-shift null: mean r={null.mean():.3f}  ->  p={p:.4f}")
    print(f"  alpha band coef sign (alpha->BOLD): informative for the vigilance link")


if __name__ == "__main__":
    main()
