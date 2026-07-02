"""NeuroTechX-Atlas epilepsy section: seizure detection on TUSZ v2.0.3, FM-vs-classical
under the Sci-Rep-2026 transparent protocol + NeuroAtlas Event-Sens@FA, + patient-id-free.

Patient-disjoint official split: train on TUSZ ``edf/train``, pick threshold+post-process
on ``edf/dev``, score ``edf/eval``. Per recording: continuous EEG -> fixed windows ->
per-window seizure score; a window is positive if it overlaps a ``.csv_bi`` seizure event
(any-overlap). Scoring via ``epilepsy_scorer`` (Event-Sens@FA-AUC over 0.1-100 FA/h).

Models:
  --model logbandpower : expert window features -> CatBoost/HistGBM (the paper's anchor)
  --model reve|biot|... : frozen window embeddings -> balanced logistic-regression probe
Our axis: fit the probe/GBM with PATIENT identity LEACE-erased -> identity-free
Event-Sens@FA (seizure EEG is highly patient-specific -> prime identity-trap territory).

Classical path runs CPU-only (mne+scipy+sklearn); FM path needs the 26.06 container
(scripts/eegfm_t9.sh). Use --limit to validate on a subset before the full run.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epilepsy_scorer as es  # noqa: E402

TUSZ_ROOT = "/data/datasets/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf"
# TUH standard 19 (10-20). .edf channels look like "EEG FP1-REF" / "EEG FP1-LE".
STD19 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
         "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]


def iter_recordings(split, limit=None):
    """(edf_path, csv_bi_path, patient_id) for every recording in a TUSZ split."""
    pat = os.path.join(TUSZ_ROOT, split, "*", "*", "*", "*.edf")
    out = []
    for edf in sorted(glob.glob(pat)):
        cbi = edf[:-4] + ".csv_bi"
        if os.path.exists(cbi):
            patient = edf.split(os.sep)[-4]  # .../split/<patient>/<session>/<montage>/x.edf
            out.append((edf, cbi, patient))
    return out[:limit] if limit else out


def _std_name(ch):
    m = re.match(r"(?:EEG\s+)?([A-Za-z0-9]+?)\s*-?\s*(?:REF|LE)?$", ch.strip(), re.I)
    return (m.group(1).upper() if m else ch.strip().upper())


def load_windows(edf, win_s=10.0, sfreq_out=200.0):
    """Continuous EEG -> non-overlapping windows on the STD19 montage.
    Returns (X (n,19,T) float32, win_starts (n,) sec) or (None, None) if incomplete."""
    import mne
    mne.set_log_level("error")
    try:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
    except Exception:
        return None, None
    ren = {c: _std_name(c) for c in raw.ch_names}
    raw.rename_channels(ren)
    present = [c for c in STD19 if c in raw.ch_names]
    if len(present) < len(STD19):
        return None, None
    raw.pick(present).reorder_channels(STD19)
    if raw.info["sfreq"] != sfreq_out:
        raw.resample(sfreq_out, verbose="error")
    x = raw.get_data()                       # (19, T) Volts
    w = int(win_s * sfreq_out)
    n = x.shape[1] // w
    if n == 0:
        return None, None
    X = np.stack([x[:, i * w:(i + 1) * w] for i in range(n)]).astype(np.float32)
    starts = np.arange(n) * win_s
    return X, starts


def label_windows(starts, win_s, seiz_events):
    """1 if the window [start, start+win_s) overlaps any seizure event, else 0."""
    y = np.zeros(len(starts), int)
    for i, s in enumerate(starts):
        e = s + win_s
        if any(s < b and a < e for a, b in seiz_events):
            y[i] = 1
    return y


def build_split(split, win_s, sfreq, limit=None):
    """Per-recording windows for a split. Returns list of dicts + a stacked
    (features-later) view. Keeps recordings separate (needed for event scoring)."""
    recs = []
    for edf, cbi, patient in iter_recordings(split, limit):
        X, starts = load_windows(edf, win_s, sfreq)
        if X is None:
            continue
        seiz = es.load_csv_bi(cbi)
        y = label_windows(starts, win_s, seiz)
        dur_h = (starts[-1] + win_s) / 3600.0
        recs.append(dict(X=X, y=y, starts=starts, seiz=seiz, patient=patient, dur_h=dur_h))
    return recs


# ---------- embeddings ----------
def embed_recs(recs, model, sfreq, dev):
    """Attach per-window embeddings 'emb' to each recording dict, by model."""
    if model == "logbandpower":
        for r in recs:
            r["emb"] = np.stack([es.window_features(w, sfreq) for w in r["X"]])
    elif model == "reve":
        from atlas_bci_section import embed_reve
        for r in recs:
            r["emb"] = embed_reve(r["X"], STD19, sfreq, dev)
    else:
        from atlas_bci_section import EEGFM, embed_eegfm
        if model not in EEGFM:
            raise SystemExit(f"unknown model {model!r}")
        for r in recs:
            r["emb"] = embed_eegfm(model, r["X"], STD19, sfreq, dev)
    return recs


# ---------- train / score ----------
def _fit_probe(Xtr, ytr, model, erase_patient=None):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    if erase_patient is not None:
        sys.path.insert(0, "/mnt/t9")
        from leace import LeaceEraser
        er = LeaceEraser().fit(Xs, erase_patient)
        Xs = er.transform(Xs)
    else:
        er = None
    if model == "logbandpower":
        clf, _ = es.train_gbm(Xs, ytr)
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, ytr)
    return clf, sc, er


def _score_recs(recs, clf, sc, er):
    scores, starts, seiz = [], [], []
    for r in recs:
        Xs = sc.transform(r["emb"])
        if er is not None:
            Xs = er.transform(Xs)
        p = clf.predict_proba(Xs)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(Xs)
        scores.append(p); starts.append(r["starts"]); seiz.append(r["seiz"])
    total_h = sum(r["dur_h"] for r in recs)
    return scores, starts, seiz, total_h


def evaluate(train, dev, evl, model, win_s, erase_patient=False):
    Xtr = np.concatenate([r["emb"] for r in train])
    ytr = np.concatenate([r["y"] for r in train])
    pid = None
    if erase_patient:
        pmap = {p: i for i, p in enumerate(sorted({r["patient"] for r in train}))}
        pid = np.concatenate([[pmap[r["patient"]]] * len(r["y"]) for r in train])
    clf, sc, er = _fit_probe(Xtr, ytr, model, erase_patient=pid)
    dsc, dst, dse, dh = _score_recs(dev, clf, sc, er)
    esc, est, ese, eh = _score_recs(evl, clf, sc, er)
    dev_curve = es.sens_fa_curve(dsc, dst, win_s, dse, dh)
    eval_curve = es.sens_fa_curve(esc, est, win_s, ese, eh)
    return dict(dev_auc=es.event_sens_at_fa_auc(dev_curve),
                eval_auc=es.event_sens_at_fa_auc(eval_curve),
                eval_sens10=es.sensitivity_at(eval_curve, 10.0),
                eval_sens1=es.sensitivity_at(eval_curve, 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="logbandpower")
    ap.add_argument("--win-s", type=float, default=10.0)
    ap.add_argument("--sfreq", type=float, default=200.0)
    ap.add_argument("--limit", type=int, default=None, help="max recordings/split (dev/smoke)")
    ap.add_argument("--identity-free", action="store_true")
    a = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[epilepsy] model={a.model} win={a.win_s}s limit={a.limit}", flush=True)
    splits = {}
    for s in ("train", "dev", "eval"):
        recs = build_split(s, a.win_s, a.sfreq, a.limit)
        recs = embed_recs(recs, a.model, a.sfreq, dev_)
        pos = sum(int(r["y"].sum()) for r in recs)
        tot = sum(len(r["y"]) for r in recs)
        print(f"  {s}: {len(recs)} recs, {tot} win ({pos} seizure), d={recs[0]['emb'].shape[1] if recs else 0}", flush=True)
        splits[s] = recs

    out = evaluate(splits["train"], splits["dev"], splits["eval"], a.model, a.win_s,
                   erase_patient=a.identity_free)
    tag = " (identity-free)" if a.identity_free else ""
    print(f"\n[epilepsy] {a.model}{tag}  Event-Sens@FA:")
    print(f"  eval AUC(sens vs log FA/h)[0.1-100] = {out['eval_auc']:.3f}")
    print(f"  eval sensitivity @ 10 FA/h = {out['eval_sens10']*100:.1f}% | @ 1 FA/h = {out['eval_sens1']*100:.1f}%")
    print(f"  (dev AUC = {out['dev_auc']:.3f})")


if __name__ == "__main__":
    main()
