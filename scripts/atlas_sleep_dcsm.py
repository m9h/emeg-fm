"""NeuroTechX-Atlas sleep-staging section, dataset 7: DCSM (Danish Center for
Sleep Medicine, sleep-EDF-adjacent open PSG cohort, 255 subjects). A 7th
sleep dataset (after Sleep-EDF/HMC/DOD-H/DOD-O/Ear-EEG/ISRUC) and the
**largest single sleep cohort** in the program by subject count.

Original archive is a 392GB zip holding BOTH ``psg.edf`` and a redundant
``psg.h5`` copy of the same signal per subject -- only ``psg.edf`` +
``hypnogram.ids`` were extracted (selective ``unzip`` glob), roughly halving
disk usage since the .h5 duplicates carry no additional information.

``hypnogram.ids`` is a variable-duration interval format (``start_s,
duration_s, stage``), NOT pre-epoched at a fixed 30s grid like Dreem DOD --
but every boundary in the corpus is an exact multiple of 30s (verified), so
expanding each interval into ``duration_s // 30`` repeated 30s epochs is
lossless. Labels are ``{W,N1,N2,N3,REM}`` -> remapped to the shared AASM 0-4
convention (W=0,N1=1,N2=2,N3=3,REM=4).

EEG channels are bipolar-to-mastoid derivations (F3-M2, F4-M1, C3-M2, C4-M1,
O1-M2, O2-M1) -- same REVE anchor-electrode convention as Dreem DOD/ISRUC
(first electrode of each pair).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_sleep_section as sl  # noqa: E402
import epilepsy_scorer as es  # noqa: E402

ROOT = os.environ.get("DCSM_ROOT", "/data/datasets/eeg_fmri/dcsm_sleep/extracted/data/sleep/DCSM")
EEG_CHANNELS = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]
LABEL_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}


def _parse_hypnogram(path, epoch_s=30.0):
    """(start_s,duration_s,stage) intervals -> per-epoch label array (n,)."""
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _start, dur, stage = line.split(",")
            n_ep = int(round(float(dur) / epoch_s))
            code = LABEL_MAP.get(stage)
            if code is None:
                labels.extend([-1] * n_ep)  # unmapped/unscored -> dropped downstream
            else:
                labels.extend([code] * n_ep)
    return np.asarray(labels, dtype=int)


def iter_recordings(limit=None):
    """(edf_path, hyp_path, subject_id) for every DCSM recording."""
    out = []
    for hyp in sorted(glob.glob(os.path.join(ROOT, "*", "hypnogram.ids"))):
        subj_dir = os.path.dirname(hyp)
        edf = os.path.join(subj_dir, "psg.edf")
        if os.path.exists(edf):
            out.append((edf, hyp, os.path.basename(subj_dir)))
    return out[:limit] if limit else out


def load_epochs(edf_path, hyp_path):
    """psg.edf + hypnogram.ids -> (X (n,6,30*fs) float32, y (n,) int 0-4)."""
    import mne
    mne.set_log_level("error")
    labels = _parse_hypnogram(hyp_path)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="error")
    present = [c for c in EEG_CHANNELS if c in raw.ch_names]
    if len(present) < len(EEG_CHANNELS):
        return None, None
    raw.pick(present).reorder_channels(present)
    sfreq = raw.info["sfreq"]
    n = int(sl.EPOCH_S * sfreq)
    data = raw.get_data()

    n_ep = min(len(labels), data.shape[1] // n)
    X, y = [], []
    for i in range(n_ep):
        if labels[i] < 0:
            continue
        X.append(data[:, i * n:(i + 1) * n])
        y.append(labels[i])
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.asarray(y, int)


def _reve_channel_names():
    return [c.split("-")[0] for c in EEG_CHANNELS]


def make_embed_fn(model, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        ch = _reve_channel_names()
        return lambda X: embed_reve(X, ch, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (dcsm supports logbandpower, reve)")


def subject_splits(limit=None, seed_frac=(0.7, 0.15, 0.15)):
    subs = sorted({s for _, _, s in iter_recordings(limit)})
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def build_split(subjects, limit=None, embed_fn=None):
    recs = []
    for edf, hyp, subj in iter_recordings(limit):
        if subj not in subjects:
            continue
        X, y = load_epochs(edf, hyp)
        if X is None:
            continue
        rec = dict(y=y, patient=subj)
        rec["emb"] = embed_fn(X) if embed_fn is not None else None
        recs.append(rec)
    return recs


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="logbandpower")
    ap.add_argument("--sfreq", type=float, default=256.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--identity-free", action="store_true")
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = subject_splits(a.limit)
    print(f"[dcsm] model={a.model} limit={a.limit} subjects train/dev/eval="
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
        print(f"\n[dcsm] {a.model}{tag} 5-class AASM staging:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        print(f"  (dev balanced-acc = {out['dev_bacc']*100:.1f}%  kappa = {out['dev_kappa']:.3f})")
    if False in results and True in results:
        d = results[False]["eval_kappa"] - results[True]["eval_kappa"]
        print(f"\n[dcsm] {a.model} identity-free Δ(eval kappa) = {d:+.3f} "
              f"(normal {results[False]['eval_kappa']:.3f} - idfree {results[True]['eval_kappa']:.3f})")


if __name__ == "__main__":
    main()
