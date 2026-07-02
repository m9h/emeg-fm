"""Brain-age for braindecode-native EEG foundation models (BIOT / BENDR / CBraMod
/ LUNA), on the SAME cohort/CV as coffeine / REVE / the generic TS-FMs.

Why this file is short: braindecode 1.5.2 ships the whole EEG-FM zoo as first-class
``braindecode.models.*`` classes with ``from_pretrained`` + a uniform EEGModuleMixin
structure, and ``Interpolated*`` wrappers that map ANY input montage onto the
model's pretrained electrode layout. So one loader + one forward-hook on the shared
``final_layer`` yields a pooled embedding for every model — no per-model
re-implementation (contrast scripts/eeg_fm_brain_age.py, which needs a bespoke
HFModelAdapter per model). See docs/braindecode_eegfm_extraction.md.

Per subject: resample -> fit the model's window -> z-score+clamp -> forward with the
pre-hook capturing final_layer's input (the pooled feature) -> epoch-mean; RidgeCV
under identical KFold(seed) splits. Runs via scripts/eegfm_t9.sh (NGC 26.06).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

# model -> (braindecode class name, HF id, target_sfreq, window_samples, extra kwargs).
# Interpolated* accept arbitrary input channels (interpolate to the pretrained montage).
BD_SPEC = {
    "biot":    ("InterpolatedBIOT",  "braindecode/biot-pretrained-six-datasets-18chs", 200.0, 1000, {}),
    "bendr":   ("InterpolatedBENDR", "braindecode/braindecode-bendr",                  250.0, 1000, {}),
    "cbramod": ("CBraMod",           "braindecode/cbramod-pretrained",                 200.0, 1000, {}),
    "luna":    ("LUNA",              "PulpBio/LUNA",                                    200.0, 1000, {}),
}


def _resample_time(x, sfreq_in, sfreq_out):
    if abs(sfreq_in - sfreq_out) < 1e-6:
        return x
    from scipy.signal import resample
    return resample(x, int(round(x.shape[-1] * sfreq_out / sfreq_in)), axis=-1)


def _fit_len(x, target):
    L = x.shape[-1]
    if L == target:
        return x
    if L > target:
        s = (L - target) // 2
        return x[..., s:s + target]
    pad = target - L
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad // 2, pad - pad // 2)], mode="edge")


def _load_ages(participants):
    import pandas as pd
    df = pd.read_csv(participants, sep=None, engine="python")
    idc = "participant_id" if "participant_id" in df.columns else df.columns[0]
    return dict(zip(df[idc].astype(str).str.replace("sub-", "", regex=False),
                    df["age"].astype(float)))


def _build_model(model, ch_names, sfreq, win, device):
    import braindecode.models as bm
    import mne
    import torch
    mne.set_log_level("error")
    cls_name, mid, _, _, extra = BD_SPEC[model]
    info = mne.create_info(list(ch_names), sfreq, "eeg")
    info.set_montage("standard_1005", on_missing="ignore")
    chs_info = info["chs"]
    cls = getattr(bm, cls_name)
    m = cls.from_pretrained(mid, chs_info=chs_info, n_outputs=2, n_times=win,
                            sfreq=sfreq, **extra).to(device).eval()
    cap = {}
    m.final_layer.register_forward_pre_hook(lambda mod, a: cap.__setitem__("z", a[0].detach()))
    return m, cap


def _subject_embedding(m, cap, data, sfreq_in, sfreq_out, win, batch_size, device):
    import torch
    pooled = []
    for i in range(0, data.shape[0], batch_size):
        b = _fit_len(_resample_time(np.asarray(data[i:i + batch_size], np.float64),
                                    sfreq_in, sfreq_out), win)
        mu = b.mean(-1, keepdims=True)
        b = np.clip((b - mu) / (b.std(-1, keepdims=True) + 1e-8), -15, 15).astype(np.float32)
        with torch.no_grad():
            m(torch.tensor(b, device=device))
        z = cap["z"]
        if z.ndim == 3:                       # (B, tokens, D) -> token-mean
            z = z.mean(1)
        pooled.append(np.asarray(z.float().cpu()))
    return np.concatenate(pooled, 0).mean(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(BD_SPEC), required=True)
    p.add_argument("--epochs-glob", required=True)
    p.add_argument("--participants", required=True)
    p.add_argument("--subject-regex", default=r"sub-([A-Za-z0-9]+)")
    p.add_argument("--sfreq", type=float, default=200.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import mne
    import torch
    mne.set_log_level("error")

    out = Path(a.out)
    if out.exists():
        z = np.load(out, allow_pickle=False)
        X, ages = z["X"], z["ages"]
        print(f"loaded cached {X.shape} from {out}")
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _, mid, sfreq_out, win, _ = BD_SPEC[a.model]
        age = _load_ages(a.participants)
        pat = re.compile(a.subject_regex)
        files = sorted(Path("/").glob(a.epochs_glob.lstrip("/")))
        m = cap = None
        embs, ages, t0 = [], [], time.time()
        for f in files:
            g = pat.search(f.name)
            if not g or g.group(1) not in age or not np.isfinite(age[g.group(1)]):
                continue
            ep = mne.read_epochs(f, preload=True, verbose="error")
            if m is None:
                m, cap = _build_model(a.model, ep.ch_names, sfreq_out, win, dev)
                print(f"{a.model} loaded ({mid}, sfreq {a.sfreq}->{sfreq_out}, win={win})")
            emb = _subject_embedding(m, cap, ep.get_data(copy=False), a.sfreq,
                                     sfreq_out, win, a.batch_size, dev)
            embs.append(emb)
            ages.append(age[g.group(1)])
            del ep
            if len(ages) % 25 == 0:
                print(f"  {len(ages)} subjects ({time.time()-t0:.0f}s)", flush=True)
            if a.max_subjects and len(ages) >= a.max_subjects:
                break
        X = np.vstack(embs).astype(np.float32)
        ages = np.asarray(ages)
        np.savez(out, X=X, ages=ages)
        print(f"cached -> {out}")

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    print(f"\n{a.model} brain-age: {X.shape[0]} subjects, d={X.shape[1]}")
    print(f"dummy-mean MAE = {np.mean(np.abs(ages - ages.mean())):.2f} yr")
    cv = KFold(n_splits=a.n_splits, shuffle=True, random_state=a.seed)
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 25)))
    sc = cross_validate(reg, X, ages, cv=cv, scoring=("neg_mean_absolute_error", "r2"))
    mae, r2 = -sc["test_neg_mean_absolute_error"], sc["test_r2"]
    print(f"{a.model}+RidgeCV  MAE = {mae.mean():.2f} +/- {mae.std():.2f} yr   "
          f"R^2 = {r2.mean():.3f} +/- {r2.std():.3f}")


if __name__ == "__main__":
    main()
