"""NeuroTechX-Atlas sleep-staging section, dataset 2: HMC (Haaglanden Medisch
Centrum sleep staging database, PhysioNet, NeuroAtlas-named).

Same 5-class AASM task/metric as Sleep-EDF (atlas_sleep_section.py) -- reuses its
STAGE_MAP/EPOCH_S/_fit_probe/evaluate directly -- but a modern reduced clinical
montage (4 EEG channels referenced to mastoids: F4-M1, C4-M1, O2-M1, C3-M2, vs
Sleep-EDF's 2) and a plain-text scoring export (not an EDF+ Hypnogram), so its
own loader. 151 recordings, one session/subject each (SN001..SN151).

Data: /mnt/t9/hmc_sleep/extracted/.../recordings/SNnnn.edf + SNnnn_sleepscoring.txt
(PhysioNet, no auth). Real 10-20 anchor electrodes (F4/C4/O2/C3) -> REVE supported
via the same anchor-electrode approximation as Sleep-EDF (drop the -M1/-M2 ref).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_sleep_section as sl  # noqa: E402  (reuses STAGE_MAP/EPOCH_S/_fit_probe/evaluate)
import epilepsy_scorer as es  # noqa: E402  (window_features)

ROOT = os.environ.get(
    "HMC_ROOT",
    "/mnt/t9/hmc_sleep/extracted/haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings")
EEG_CHANNELS = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2"]


def iter_recordings(limit=None):
    """(edf_path, txt_path, subject_id) for every HMC recording."""
    out = []
    for edf in sorted(glob.glob(os.path.join(ROOT, "SN*.edf"))):
        if "sleepscoring" in edf:
            continue
        subj = re.search(r"(SN\d+)\.edf", edf).group(1)
        txt = os.path.join(ROOT, f"{subj}_sleepscoring.txt")
        if os.path.exists(txt):
            out.append((edf, txt, subj))
    return out[:limit] if limit else out


def _parse_scoring_txt(path):
    """'Date, Time, Recording onset, Duration, Annotation, Linked channel' rows
    -> [(onset_s, duration_s, description)] for 'Sleep stage *' rows only
    (drops 'Lights off'/body-position/other non-stage annotations)."""
    events = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if len(parts) < 5:
                continue
            desc = parts[4]
            if not desc.startswith("Sleep stage"):
                continue
            onset, dur = float(parts[2]), float(parts[3])
            events.append((onset, dur, desc))
    return events


def load_epochs(edf, txt, crop_wake_min=30.0):
    """EDF signal + txt scoring -> (X (n,4,30*sfreq), y (n,)), same convention
    as atlas_sleep_section.load_epochs (crop + first_samp-safe indexing)."""
    import mne
    mne.set_log_level("error")
    try:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
        rows = _parse_scoring_txt(txt)
    except Exception:
        return None, None
    if not rows:
        return None, None
    onsets, durations, descs = zip(*rows)
    raw.set_annotations(mne.Annotations(list(onsets), list(durations), list(descs)),
                        emit_warning=False)
    present = [c for c in EEG_CHANNELS if c in raw.ch_names]
    if len(present) < len(EEG_CHANNELS):
        return None, None
    raw.pick(present).reorder_channels(present)
    sfreq = raw.info["sfreq"]

    sleep_onsets = [o for o, _, d in rows if d != "Sleep stage W"]
    if sleep_onsets:
        t0 = max(0.0, min(sleep_onsets) - crop_wake_min * 60)
        t1 = min(raw.times[-1], max(sleep_onsets) + crop_wake_min * 60)
        raw.crop(t0, t1)

    events, _ = mne.events_from_annotations(
        raw, event_id=sl.STAGE_MAP, chunk_duration=sl.EPOCH_S, verbose="error")
    if len(events) == 0:
        return None, None
    n = int(sl.EPOCH_S * sfreq)
    data = raw.get_data()
    off = raw.first_samp   # see atlas_sleep_section.load_epochs for why this matters
    X, y = [], []
    for onset_samp, _, label in events:
        i = onset_samp - off
        seg = data[:, i:i + n]
        if seg.shape[1] == n:
            X.append(seg); y.append(label)
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.asarray(y, int)


def _reve_channel_names():
    """Bipolar-to-mastoid derivations -> anchor (first/active) electrode, same
    approximation convention as atlas_sleep_section: F4-M1->F4, C3-M2->C3 etc."""
    out = []
    for c in EEG_CHANNELS:
        name = c.replace("EEG ", "").strip()
        out.append(name.split("-")[0] if "-" in name else name)
    return out


def make_embed_fn(model, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        ch = _reve_channel_names()
        return lambda X: embed_reve(X, ch, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (hmc supports logbandpower, reve)")


def subject_splits(limit=None, seed_frac=(0.7, 0.15, 0.15)):
    subs = sorted({s for _, _, s in iter_recordings(limit)})
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def build_split(subjects, limit=None, embed_fn=None):
    recs = []
    for edf, txt, subj in iter_recordings(limit):
        if subj not in subjects:
            continue
        X, y = load_epochs(edf, txt)
        if X is None:
            continue
        rec = dict(y=y, patient=subj)
        rec["emb"] = embed_fn(X) if embed_fn is not None else None
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="logbandpower")
    ap.add_argument("--sfreq", type=float, default=100.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--identity-free", action="store_true")
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = subject_splits(a.limit)
    print(f"[hmc] model={a.model} limit={a.limit} subjects train/dev/eval="
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
        print(f"\n[hmc] {a.model}{tag} 5-class AASM staging:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        print(f"  (dev balanced-acc = {out['dev_bacc']*100:.1f}%  kappa = {out['dev_kappa']:.3f})")
    if False in results and True in results:
        d = results[False]["eval_kappa"] - results[True]["eval_kappa"]
        print(f"\n[hmc] {a.model} identity-free Δ(eval kappa) = {d:+.3f} "
              f"(normal {results[False]['eval_kappa']:.3f} - idfree {results[True]['eval_kappa']:.3f})")


if __name__ == "__main__":
    main()
