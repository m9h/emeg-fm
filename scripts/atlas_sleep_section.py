"""NeuroTechX-Atlas sleep-staging section: Sleep-EDF Expanded (PhysioNet), the
biggest gap vs NeuroAtlas (they report 15 sleep datasets / ~201k h; we had 0 built).

5-class AASM staging (W/N1/N2/N3/REM, N3=merged stages 3+4) on 30 s epochs from the
Fpz-Cz + Pz-Oz EEG channels. Subject-disjoint train/dev/eval split (never mix subjects).
Metric = balanced accuracy + Cohen's kappa (the standard sleep-staging pair; matches
NeuroAtlas's "hypnogram-derived features" framing). Same architecture as the epilepsy
section: streaming build+embed (OOM-safe), classical + FM (REVE; montage-flexible —
Interpolated* needs >=4 head-dig points, Sleep-EDF has only 2 EEG channels so BIOT/
LaBraM/EEGDINO's Interpolated wrapper can't fit; REVE's 3D-name lookup handles it),
and patient-identity-free (LEACE) axis in one embedding pass.

Data: /mnt/t9/sleep_edf/{sleep-cassette,sleep-telemetry}/SC4ssNEc-PSG.edf +
...-Hypnogram.edf (197 recordings, PhysioNet sleep-edfx, no auth).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epilepsy_scorer as es  # noqa: E402  (reuses window_features, LEACE plumbing pattern)

SLEEP_ROOT = os.environ.get(
    "SLEEP_EDF_ROOT", "/mnt/t9/sleep_edf/sleep-edf-database-expanded-1.0.0")
EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
STAGE_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3,   # AASM merges N3+N4 -> N3
    "Sleep stage R": 4,
}
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
EPOCH_S = 30.0


def iter_recordings(limit=None):
    """(psg_path, hypnogram_path, subject_id) for every Sleep-EDF recording."""
    out = []
    for psg in sorted(glob.glob(os.path.join(SLEEP_ROOT, "*", "*-PSG.edf"))):
        base = os.path.basename(psg)
        m = re.match(r"(SC4|ST7)(\d\d)", base)
        if not m:
            continue
        subject = m.group(1) + m.group(2)
        prefix = base.split("-PSG.edf")[0][:-1]  # drop the trailing recording-variant letter
        hyps = glob.glob(os.path.join(os.path.dirname(psg), prefix + "*-Hypnogram.edf"))
        if not hyps:
            continue
        out.append((psg, hyps[0], subject))
    return out[:limit] if limit else out


def load_epochs(psg, hyp, crop_wake_min=30.0):
    """PSG + Hypnogram -> (X (n,2,30*sfreq) float32, y (n,) int 0-4). Crops the long
    wake padding at the start/end to +/- crop_wake_min around the first/last sleep
    epoch (standard practice; avoids a dataset that is >80% class W)."""
    import mne
    mne.set_log_level("error")
    try:
        raw = mne.io.read_raw_edf(psg, preload=True, verbose="error")
        annot = mne.read_annotations(hyp)
    except Exception:
        return None, None
    raw.set_annotations(annot, emit_warning=False)
    present = [c for c in EEG_CHANNELS if c in raw.ch_names]
    if len(present) < 2:
        return None, None
    raw.pick(present).reorder_channels(present)
    sfreq = raw.info["sfreq"]

    sleep_onsets = [a["onset"] for a in annot if a["description"] != "Sleep stage ?"
                    and a["description"] != "Sleep stage W"]
    if sleep_onsets:
        t0 = max(0.0, min(sleep_onsets) - crop_wake_min * 60)
        t1 = min(raw.times[-1], max(sleep_onsets) + crop_wake_min * 60)
        raw.crop(t0, t1)

    events, _ = mne.events_from_annotations(
        raw, event_id=STAGE_MAP, chunk_duration=EPOCH_S, verbose="error")
    if len(events) == 0:
        return None, None
    n = int(EPOCH_S * sfreq)
    data = raw.get_data()
    X, y = [], []
    for onset_samp, _, label in events:
        seg = data[:, onset_samp:onset_samp + n]
        if seg.shape[1] == n:
            X.append(seg); y.append(label)
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.asarray(y, int)


def build_split(subjects, limit=None, embed_fn=None):
    """Per-recording epochs for the given subject set, streaming embed like epilepsy."""
    recs = []
    for psg, hyp, subj in iter_recordings(limit):
        if subj not in subjects:
            continue
        X, y = load_epochs(psg, hyp)
        if X is None:
            continue
        rec = dict(y=y, patient=subj)
        if embed_fn is not None:
            rec["emb"] = embed_fn(X)
        else:
            rec["X"] = X
        recs.append(rec)
    return recs


def make_embed_fn(model, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        return lambda X: embed_reve(X, EEG_CHANNELS, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (sleep section supports logbandpower, reve)")


def subject_splits(limit=None, seed_frac=(0.7, 0.15, 0.15)):
    """Subject-disjoint 70/15/15. Floors dev/eval at >=1 subject (else a small
    --limit smoke test rounds a split to 0 -> degenerate untuned evaluation)."""
    subs = sorted({s for _, _, s in iter_recordings(limit)})
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def _fit_probe(Xtr, ytr, model, erase_patient=None):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    er = None
    if erase_patient is not None:
        sys.path.insert(0, "/mnt/t9")
        from leace import LeaceEraser
        er = LeaceEraser().fit(Xs, erase_patient)
        Xs = er.transform(Xs)
    if model == "logbandpower":
        clf, _ = es.train_gbm(Xs, ytr)
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 multi_class="multinomial").fit(Xs, ytr)
    return clf, sc, er


def _predict(recs, clf, sc, er):
    Xs = sc.transform(np.concatenate([r["emb"] for r in recs]))
    if er is not None:
        Xs = er.transform(Xs)
    yhat = clf.predict(Xs)
    y = np.concatenate([r["y"] for r in recs])
    return y, yhat


def evaluate(train, dev, evl, model, erase_patient=False):
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
    Xtr = np.concatenate([r["emb"] for r in train])
    ytr = np.concatenate([r["y"] for r in train])
    pid = None
    if erase_patient:
        pmap = {p: i for i, p in enumerate(sorted({r["patient"] for r in train}))}
        pid = np.concatenate([[pmap[r["patient"]]] * len(r["y"]) for r in train])
    clf, sc, er = _fit_probe(Xtr, ytr, model, erase_patient=pid)
    ydev, yhat_dev = _predict(dev, clf, sc, er)
    yeval, yhat_eval = _predict(evl, clf, sc, er)
    return dict(
        dev_bacc=balanced_accuracy_score(ydev, yhat_dev),
        dev_kappa=cohen_kappa_score(ydev, yhat_dev),
        eval_bacc=balanced_accuracy_score(yeval, yhat_eval),
        eval_kappa=cohen_kappa_score(yeval, yhat_eval),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="logbandpower")
    ap.add_argument("--sfreq", type=float, default=100.0)
    ap.add_argument("--limit", type=int, default=None, help="max recordings (dev/smoke)")
    ap.add_argument("--identity-free", action="store_true")
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = subject_splits(a.limit)
    print(f"[sleep] model={a.model} limit={a.limit} subjects train/dev/eval="
          f"{len(tr_s)}/{len(dv_s)}/{len(ev_s)}", flush=True)
    embed_fn = make_embed_fn(a.model, a.sfreq, dev_)
    splits = {}
    for name, subs in (("train", tr_s), ("dev", dv_s), ("eval", ev_s)):
        recs = build_split(subs, a.limit, embed_fn=embed_fn)
        tot = sum(len(r["y"]) for r in recs)
        splits[name] = recs
        print(f"  {name}: {len(recs)} recs, {tot} epochs, d={recs[0]['emb'].shape[1] if recs else 0}", flush=True)

    configs = [False, True] if (a.both or a.identity_free) else [False]
    results = {}
    for erase in configs:
        out = evaluate(splits["train"], splits["dev"], splits["eval"], a.model, erase_patient=erase)
        results[erase] = out
        tag = " (identity-free)" if erase else ""
        print(f"\n[sleep] {a.model}{tag} 5-class AASM staging:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        print(f"  (dev balanced-acc = {out['dev_bacc']*100:.1f}%  kappa = {out['dev_kappa']:.3f})")
    if False in results and True in results:
        d = results[False]["eval_kappa"] - results[True]["eval_kappa"]
        print(f"\n[sleep] {a.model} identity-free Δ(eval kappa) = {d:+.3f} "
              f"(normal {results[False]['eval_kappa']:.3f} - idfree {results[True]['eval_kappa']:.3f})")


if __name__ == "__main__":
    main()
