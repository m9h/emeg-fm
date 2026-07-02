"""EEG foundation-model brain-age, generalized over the registered adapters, on
the SAME cohort/CV as coffeine / NEOBA / REVE / the generic TS-FMs.

Generalizes reve_brain_age.py to any adapter-backed EEG-FM (REVE, LaBraM, ZUNA)
so the FM column of the brain-age benchmark is representative rather than a single
model. All three share the NeuralBench input contract (per-channel z-score + clamp
±15, dict ``{"eeg","electrode_names","ch_names"}``) — the adapter handles the
model internals — so the only per-model knobs are the window length and layer.

Per subject: z-score+clamp each resting epoch, run the frozen encoder, mean-pool
tokens then epochs -> one embedding/subject; RidgeCV under identical KFold(seed)
splits -> MAE/R^2. Runs in the NGC 26.06 container (torch+transformers+braindecode);
see scripts/eeg_fm_brain_age_t9.sh.

CBraMod/BIOT/BENDR/LUNA have weight-loaders (analyze_eegfm_weightwatcher.py) but no
forward-extraction adapter yet — they are NOT wired here.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

# Each EEG-FM has a rigid input contract. Per-model: (HF id, target_sfreq Hz,
# window_samples at that sfreq | None=flexible). Epochs are resampled to
# target_sfreq then center-cropped/edge-padded to window_samples.
#   REVE   flexible length, 200 Hz.
#   ZUNA   256 Hz, T divisible by 32 -> 1280 (=5 s, 40·32).
#   LaBraM 200 Hz, pos-emb fixed at 15 s = 3000 samples (our epochs are 10 s -> padded).
MODEL_SPEC = {
    "reve": ("brain-bzh/reve-base", 200.0, None),
    "zuna": ("mhough/zuna-base", 256.0, 1280),
    "labram": ("braindecode/labram-pretrained", 200.0, 3000),
}


def _load_adapter(model, layer):
    from emeg_fm.eeg_fm import (LaBraMAdapter, REVEAdapter, REVE_BASE_ID,
                                ZUNA_BASE_ID, ZunaAdapter)
    hf_id = MODEL_SPEC[model][0]
    if model == "reve":
        adapter, hf_id = REVEAdapter(layer=layer), REVE_BASE_ID
    elif model == "zuna":
        adapter, hf_id = ZunaAdapter(layer=layer), ZUNA_BASE_ID
    elif model == "labram":
        adapter = LaBraMAdapter(layer=layer)
    else:
        raise SystemExit(f"unknown model {model!r}; adapter-backed: {list(MODEL_SPEC)}")
    loaded = adapter.load_model(hf_id)
    return adapter, loaded


def _fit_len(x, target_len):
    """Center-crop or edge-pad the last axis to target_len (LaBraM's fixed window)."""
    L = x.shape[-1]
    if target_len is None or L == target_len:
        return x
    if L > target_len:
        s = (L - target_len) // 2
        return x[..., s:s + target_len]
    pad = target_len - L
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad // 2, pad - pad // 2)], mode="edge")


def _load_ages(participants):
    import pandas as pd
    df = pd.read_csv(participants, sep=None, engine="python")
    id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    ids = df[id_col].astype(str).str.replace("sub-", "", regex=False)
    return dict(zip(ids, df["age"].astype(float)))


def _resample_time(x, sfreq_in, sfreq_out):
    """Resample the last axis from sfreq_in to sfreq_out (FFT)."""
    if sfreq_out is None or abs(sfreq_in - sfreq_out) < 1e-6:
        return x
    from scipy.signal import resample
    n = int(round(x.shape[-1] * sfreq_out / sfreq_in))
    return resample(x, n, axis=-1)


def _subject_embedding(adapter, loaded, data, ch_names, sfreq_in, sfreq_out,
                       win_samples, batch_size):
    """One embedding/subject: resample -> fit window -> z-score+clamp -> encode ->
    token+epoch pool."""
    pooled = []
    for i in range(0, data.shape[0], batch_size):
        b = _resample_time(np.asarray(data[i:i + batch_size], dtype=np.float64),
                           sfreq_in, sfreq_out)
        b = _fit_len(b, win_samples)
        mu = b.mean(axis=-1, keepdims=True)
        sd = b.std(axis=-1, keepdims=True) + 1e-8
        b = np.clip((b - mu) / sd, -15.0, 15.0).astype(np.float32)
        feats = adapter.extract_features(
            loaded, {"eeg": b, "electrode_names": list(ch_names), "ch_names": list(ch_names)})
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 3:
            feats = feats.mean(axis=1)
        pooled.append(feats)
    return np.concatenate(pooled, axis=0).mean(axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODEL_SPEC), required=True)
    p.add_argument("--epochs-glob", required=True)
    p.add_argument("--participants", required=True)
    p.add_argument("--subject-regex", default=r"sub-([A-Za-z0-9]+)")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--sfreq", type=float, default=200.0, help="epoch sfreq (for LaBraM 15 s window)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import mne
    mne.set_log_level("error")

    out = Path(a.out)
    if out.exists():
        z = np.load(out, allow_pickle=False)
        X, ages = z["X"], z["ages"]
        print(f"loaded cached embeddings {X.shape} from {out}")
    else:
        adapter, loaded = _load_adapter(a.model, a.layer)
        sfreq_out, win_samples = MODEL_SPEC[a.model][1], MODEL_SPEC[a.model][2]
        print(f"{a.model} loaded (layer={a.layer}, sfreq {a.sfreq}->{sfreq_out}, win={win_samples} samp)")

        age = _load_ages(a.participants)
        pat = re.compile(a.subject_regex)
        files = sorted(Path("/").glob(a.epochs_glob.lstrip("/")))
        embs, ages, t0 = [], [], time.time()
        for f in files:
            m = pat.search(f.name)
            if not m or m.group(1) not in age or not np.isfinite(age[m.group(1)]):
                continue
            ep = mne.read_epochs(f, preload=True, verbose="error")
            emb = _subject_embedding(adapter, loaded, ep.get_data(copy=False),
                                     ep.ch_names, a.sfreq, sfreq_out, win_samples,
                                     a.batch_size)
            embs.append(emb)
            ages.append(age[m.group(1)])
            del ep
            if len(ages) % 25 == 0:
                print(f"  {len(ages)} subjects ({time.time() - t0:.0f}s)", flush=True)
            if a.max_subjects and len(ages) >= a.max_subjects:
                break
        X = np.vstack(embs).astype(np.float32)
        ages = np.asarray(ages)
        np.savez(out, X=X, ages=ages)
        print(f"\ncached embeddings -> {out}")

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print(f"\n{a.model} brain-age: {X.shape[0]} subjects, d={X.shape[1]}")
    print(f"dummy-mean MAE = {np.mean(np.abs(ages - ages.mean())):.2f} yr")
    cv = KFold(n_splits=a.n_splits, shuffle=True, random_state=a.seed)
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 25)))
    sc = cross_validate(reg, X, ages, cv=cv, scoring=("neg_mean_absolute_error", "r2"))
    mae = -sc["test_neg_mean_absolute_error"]
    r2 = sc["test_r2"]
    print(f"{a.model}+RidgeCV  MAE = {mae.mean():.2f} +/- {mae.std():.2f} yr   "
          f"R^2 = {r2.mean():.3f} +/- {r2.std():.3f}")


if __name__ == "__main__":
    main()
