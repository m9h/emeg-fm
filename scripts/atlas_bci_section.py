"""NeuroTechX-Atlas BCI section: frozen EEG-FM / TS-FM leaderboard on the
MOABB∩NEMAR datasets, matching NeuroAtlas's BCI protocol + adding our identity axis.

MOABB∩NEMAR (the only MOABB datasets hosted on OpenNeuro/NEMAR, so the fully-open,
BIDS-reproducible BCI core):
  - TrianaGuzman2024 (ds005342) — Motor Imagery, 32 subj, 17 ch
  - Chailloux2020    (ds003190) — P300 / ERP,   19 subj,  8 ch
Loading them via MOABB *is* the NEMAR ingest (auto-download from OpenNeuro S3).

Per (dataset, model): MOABB paradigm -> per-trial windows -> frozen encoder
embedding (one per trial) -> **LOSO** (leave-one-subject-out) linear probe. We
report NeuroAtlas's metric — **normalized balanced accuracy** (chance=50%,
perfect=100%, regardless of #classes) — PLUS our **identity-free Δ**: LEACE-erase
subject identity from the embeddings and re-probe (how much of the "decoding" was
subject re-identification). Classical baseline = per-channel log-band-power + LR.

EEG-FMs run through ONE braindecode path (Interpolated* handle any montage);
TS-FMs through their packages. Runs in the NGC 26.06 container (scripts/eegfm_t9.sh).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

BCI_DATASETS = {
    # key: (moabb class, paradigm, keep-classes|None = all)
    "triana":    ("TrianaGuzman2024", "imagery", ("imagery_sit_to_stand", "imagery_stand_to_sit")),
    "chailloux": ("Chailloux2020", "p300", None),
}

# EEG-FMs via braindecode (Interpolated* = montage-agnostic). name: (class, hf_id, sfreq, win)
EEGFM = {
    "biot":   ("InterpolatedBIOT",  "braindecode/biot-pretrained-six-datasets-18chs", 200.0, 1000),
    "bendr":  ("InterpolatedBENDR", "braindecode/braindecode-bendr",                  250.0, 1000),
    "labram": ("InterpolatedLaBraM", "braindecode/labram-pretrained",                 200.0, 3000),
}
TSFM = {"moment", "mantis", "chronos-bolt"}
CLASSICAL = {"logbandpower"}


def _resample(x, sf_in, sf_out):
    if abs(sf_in - sf_out) < 1e-6:
        return x
    from scipy.signal import resample
    return resample(x, int(round(x.shape[-1] * sf_out / sf_in)), axis=-1)


def _fit_len(x, n):
    L = x.shape[-1]
    if L == n:
        return x
    if L > n:
        s = (L - n) // 2
        return x[..., s:s + n]
    p = n - L
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(p // 2, p - p // 2)], mode="edge")


def load_dataset(key):
    """MOABB paradigm -> (X (n,C,T), y int labels, subjects int, sfreq, ch_names)."""
    import moabb.datasets as md
    from moabb.paradigms import P300, MotorImagery
    cls, para_kind, keep = BCI_DATASETS[key]
    ds = getattr(md, cls)()
    para = MotorImagery() if para_kind == "imagery" else P300()
    X, y, meta = para.get_data(ds)
    y = np.asarray(y)
    if keep is not None:
        m = np.isin(y, list(keep))
        X, y, meta = X[m], y[m], meta[m]
    classes = sorted(set(y))
    yi = np.array([classes.index(v) for v in y])
    subs = meta["subject"].to_numpy()
    _, subj_int = np.unique(subs, return_inverse=True)
    info = ds._get_single_subject_data(ds.subject_list[0])
    ch_names = list(next(iter(next(iter(info.values())).values())).ch_names) if isinstance(info, dict) else None
    sfreq = float(para.resample or 0) or 250.0
    return X.astype(np.float32), yi, subj_int, sfreq, ch_names, len(classes)


# ---------- embedding extractors ----------
def embed_eegfm(name, X, ch_names, sfreq_in, device, batch=16):
    import braindecode.models as bm
    import mne
    import torch
    mne.set_log_level("error")
    cls, mid, sf, win = EEGFM[name]
    if ch_names is None:
        ch_names = [f"EEG{i}" for i in range(X.shape[1])]
    info = mne.create_info(list(ch_names), sf, "eeg")
    info.set_montage("standard_1005", on_missing="ignore")
    m = getattr(bm, cls).from_pretrained(mid, chs_info=info["chs"], n_outputs=2,
                                         n_times=win, sfreq=sf).to(device).eval()  # may raise on small montages
    cap = {}
    m.final_layer.register_forward_pre_hook(lambda mod, a: cap.__setitem__("z", a[0].detach()))
    out = []
    for i in range(0, len(X), batch):
        b = _fit_len(_resample(np.asarray(X[i:i + batch], np.float64), sfreq_in, sf), win)
        mu = b.mean(-1, keepdims=True)
        b = np.clip((b - mu) / (b.std(-1, keepdims=True) + 1e-8), -15, 15).astype(np.float32)
        with torch.no_grad():
            m(torch.tensor(b, device=device))
        z = cap["z"]
        out.append(np.asarray((z.mean(1) if z.ndim == 3 else z).float().cpu()))
    return np.concatenate(out, 0)


def embed_classical(X, sfreq):
    """Per-channel log band-power (theta/alpha/beta/gamma) — a dependency-free
    classical reference (not CSP, but a fair spectral baseline)."""
    from scipy.signal import welch
    bands = [(4, 8), (8, 13), (13, 30), (30, 45)]
    f, pxx = welch(X, fs=sfreq, nperseg=min(256, X.shape[-1]), axis=-1)  # (n,C,F)
    feats = [np.log(pxx[:, :, (f >= lo) & (f < hi)].mean(-1) + 1e-30) for lo, hi in bands]
    return np.concatenate(feats, axis=1)  # (n, C*4)


def embed_tsfm(name, X, sfreq_in, batch=16):
    """Generic TS-FM per-trial embedding (montage-agnostic; reuse the brain-age
    adapters). One embedding per trial (no epoch-mean)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ts_fm_brain_age import ADAPTERS, _resample_time
    ad = ADAPTERS[name]()
    ad.load(ad.hf_default)
    out = []
    for i in range(0, len(X), batch):
        b = _resample_time(np.asarray(X[i:i + batch], np.float64), ad.seq_len)  # -> (B,C,seq_len)
        out.append(ad.embed(b.astype(np.float32)))
    return np.concatenate(out, 0)


def embed_reve(X, ch_names, sfreq_in, device, batch=16):
    """REVE via our 3D-coordinate adapter (montage-flexible — no interpolation)."""
    from emeg_fm.eeg_fm import REVEAdapter, REVE_BASE_ID
    ad = REVEAdapter(layer=6)
    loaded = ad.load_model(REVE_BASE_ID)
    if ch_names is None:
        ch_names = [f"EEG{i}" for i in range(X.shape[1])]
    out = []
    for i in range(0, len(X), batch):
        b = np.asarray(X[i:i + batch], np.float64)
        mu = b.mean(-1, keepdims=True)
        b = np.clip((b - mu) / (b.std(-1, keepdims=True) + 1e-8), -15, 15).astype(np.float32)
        f = np.asarray(ad.extract_features(
            loaded, {"eeg": b, "electrode_names": list(ch_names), "ch_names": list(ch_names)}),
            dtype=np.float32)
        out.append(f.mean(1) if f.ndim == 3 else f)
    return np.concatenate(out, 0)


# ---------- evaluation ----------
def _norm_ba(ba, n_classes):
    chance = 1.0 / n_classes
    return (ba - chance) / (1 - chance) * 0.5 + 0.5


def loso(emb, y, subj, n_classes, erase_identity=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    X = emb.copy()
    if erase_identity:
        sys.path.insert(0, "/mnt/t9")
        from leace import LeaceEraser
        X = LeaceEraser().fit(StandardScaler().fit_transform(X), subj).transform(
            StandardScaler().fit_transform(X))
    pred = np.zeros(len(y), int)
    for s in np.unique(subj):
        te = subj == s
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return _norm_ba(balanced_accuracy_score(y, pred), n_classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(BCI_DATASETS), required=True)
    ap.add_argument("--model", required=True,
                    help="eeg-fm (biot/bendr/labram), ts-fm (moment/mantis/chronos-bolt), or logbandpower")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[atlas-bci] loading {a.dataset} via MOABB (OpenNeuro/NEMAR ingest)...", flush=True)
    X, y, subj, sfreq, ch_names, n_classes = load_dataset(a.dataset)
    print(f"[atlas-bci] X={X.shape} n_classes={n_classes} subjects={len(np.unique(subj))} sfreq={sfreq}")

    if a.model in EEGFM:
        try:
            emb = embed_eegfm(a.model, X, ch_names, sfreq, dev)
        except Exception as e:  # braindecode Interpolated SVD fails on small BCI montages
            raise SystemExit(f"[atlas-bci] SKIP {a.model}: braindecode extraction failed "
                             f"({type(e).__name__}: {str(e)[:120]})")
    elif a.model in TSFM:
        emb = embed_tsfm(a.model, X, sfreq)
    elif a.model == "reve":
        emb = embed_reve(X, ch_names, sfreq, dev)
    elif a.model in CLASSICAL:
        emb = embed_classical(X, sfreq)
    else:
        raise SystemExit(f"unknown model {a.model!r}")
    print(f"[atlas-bci] embeddings {emb.shape}")

    raw = loso(emb, y, subj, n_classes, erase_identity=False)
    idf = loso(emb, y, subj, n_classes, erase_identity=True)
    print(f"\n[atlas-bci] {a.dataset} / {a.model}")
    print(f"  LOSO normalized-BA        = {raw*100:.1f}%   (50%=chance)")
    print(f"  identity-free normalized-BA = {idf*100:.1f}%   (Δ = {(raw-idf)*100:+.1f} pts)")


if __name__ == "__main__":
    main()
