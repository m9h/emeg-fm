"""NeuroTechX-Atlas sleep-staging section, dataset 4: Ear-EEG Sleep Monitoring
2023 (EESM23, OpenNeuro ds005178). Found via NeuroAdapt-Bench (github.com/
leegabriel/NeuroAdapt-Bench, arXiv 2604.16926), which uses it as their
"extreme modality shift" case (wearable ear-EEG vs conventional scalp PSG).
Same 30 s-epoch sleep-staging task as Sleep-EDF/HMC/DOD but a genuinely
different ACQUISITION MODALITY: 4-channel ear-EEG (RB/RT/LB/LT -- right/left,
back/top ear electrodes), not scalp.

Only 10 subjects have manually-scored sessions (usually ses-001/ses-002 of
each subject's ~12 sessions; other sessions are unscored, dropped). Scoring
uses 6 classes (Wake/REM/N1/N2/N3/Artefact) -- Artefact is dropped here (not
a real sleep stage) to keep the SAME 5-class AASM label space as our other
3 sleep datasets (0=W,1=N1,2=N2,3=N3,4=REM, per atlas_sleep_section.STAGE_MAP
convention), for a fair cross-dataset comparison.

REVE channel handling: reuses NeuroAdapt-Bench's own published ear-electrode
alias mapping (their config.py ELECTRODE_ALIASES) rather than inventing a new
approximation -- RB->A2, RT->T8, LB->A1, LT->T7 (nearest standard 10-20 scalp
positions to each ear electrode).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_sleep_section as sl  # noqa: E402  (reuses EPOCH_S/_fit_probe/evaluate)
import epilepsy_scorer as es  # noqa: E402  (window_features)

ROOT = os.environ.get("EAREEG_ROOT", "/data/datasets/eeg_fmri/eareeg")
EEG_CHANNELS = ["RB", "RT", "LB", "LT"]
REVE_ALIASES = {"RB": "A2", "RT": "T8", "LB": "A1", "LT": "T7"}
SCORING_MAP = {1: 0, 3: 1, 4: 2, 5: 3, 2: 4}  # Wake/N1/N2/N3/REM -> 0-4; 6=Artefact dropped


def iter_recordings(limit=None):
    """(eeg_set_path, scoring_tsv_path, subject_id) for every SCORED session."""
    out = []
    for tsv in sorted(glob.glob(os.path.join(
            ROOT, "sub-*", "ses-*", "eeg", "*_acq-scoring_events.tsv"))):
        prefix = tsv.replace("_acq-scoring_events.tsv", "")
        eeg_set = prefix + "_acq-earEEG_eeg.set"
        if not os.path.exists(eeg_set):
            continue
        subj = os.path.basename(prefix).split("_")[0]
        out.append((eeg_set, tsv, subj))
    return out[:limit] if limit else out


def load_epochs(eeg_set, scoring_tsv):
    """EEGLAB .set + scoring tsv -> (X (n,4,30*fs) float32, y (n,) int 0-4)."""
    import mne
    mne.set_log_level("error")
    raw = mne.io.read_raw_eeglab(eeg_set, preload=True, verbose="error")
    present = [c for c in EEG_CHANNELS if c in raw.ch_names]
    if len(present) < len(EEG_CHANNELS):
        return None, None
    raw.pick(present).reorder_channels(present)
    sfreq = raw.info["sfreq"]
    n = int(sl.EPOCH_S * sfreq)
    data = raw.get_data()

    df = pd.read_csv(scoring_tsv, sep="\t")
    X, y = [], []
    for _, row in df.iterrows():
        label = SCORING_MAP.get(int(row["scoring_idx"]))
        if label is None:  # Artefact (6), or any unmapped code
            continue
        onset_samp = int(round(row["onset"] * sfreq))
        seg = data[:, onset_samp:onset_samp + n]
        if seg.shape[1] == n:
            X.append(seg); y.append(label)
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.asarray(y, int)


def _reve_channel_names():
    return [REVE_ALIASES[c] for c in EEG_CHANNELS]


def make_embed_fn(model, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        ch = _reve_channel_names()
        return lambda X: embed_reve(X, ch, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (eareeg supports logbandpower, reve)")


def subject_splits(limit=None, seed_frac=(0.7, 0.15, 0.15)):
    subs = sorted({s for _, _, s in iter_recordings(limit)})
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def _drop_nan_rows(emb, y):
    """Drop epochs whose embedding contains a NaN (e.g. wearable ear-EEG
    segments with intermittent electrode dropout/flat-line artifacts that
    propagate NaN through log-band-power or REVE's channel interpolation).
    Returns (emb, y, n_dropped)."""
    keep = ~np.isnan(emb).any(axis=1)
    return emb[keep], y[keep], int((~keep).sum())


def build_split(subjects, limit=None, embed_fn=None):
    recs = []
    total_dropped, total_n = 0, 0
    for eeg_set, tsv, subj in iter_recordings(limit):
        if subj not in subjects:
            continue
        X, y = load_epochs(eeg_set, tsv)
        if X is None:
            continue
        emb = embed_fn(X) if embed_fn is not None else None
        if emb is not None:
            total_n += len(y)
            emb, y, n_dropped = _drop_nan_rows(emb, y)
            total_dropped += n_dropped
            if len(y) == 0:
                continue
        recs.append(dict(y=y, patient=subj, emb=emb))
    if total_dropped:
        print(f"  [eareeg] dropped {total_dropped}/{total_n} NaN-embedding epochs "
              f"({100*total_dropped/total_n:.1f}%)", flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="logbandpower")
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--identity-free", action="store_true")
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = subject_splits(a.limit)
    print(f"[eareeg] model={a.model} limit={a.limit} subjects train/dev/eval="
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
        out = sl.evaluate(splits["train"], splits["dev"], splits["eval"], a.model, erase_patient=erase)
        results[erase] = out
        tag = " (identity-free)" if erase else ""
        print(f"\n[eareeg] {a.model}{tag} 5-class AASM staging:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        print(f"  (dev balanced-acc = {out['dev_bacc']*100:.1f}%  kappa = {out['dev_kappa']:.3f})")
    if False in results and True in results:
        d = results[False]["eval_kappa"] - results[True]["eval_kappa"]
        print(f"\n[eareeg] {a.model} identity-free Δ(eval kappa) = {d:+.3f} "
              f"(normal {results[False]['eval_kappa']:.3f} - idfree {results[True]['eval_kappa']:.3f})")


if __name__ == "__main__":
    main()
