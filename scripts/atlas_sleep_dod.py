"""NeuroTechX-Atlas sleep-staging section, dataset 3: Dreem Open Datasets
(DOD-H healthy, DOD-O obstructive sleep apnea; Guillot et al. 2020, Zenodo
15900394). Same 5-class AASM task/metric as Sleep-EDF/HMC (atlas_sleep_section
.py) -- reuses its EPOCH_S/_fit_probe/evaluate directly -- but HDF5-native
storage with a PRE-EPOCHED, PRE-ALIGNED hypnogram (no crop/annotation-parsing
needed: hypnogram[i] is exactly epoch i, i*EPOCH_S seconds in) and its own
per-recording bipolar montage.

DOD-H (25 subj, healthy) and DOD-O (55 subj, OSA) ship DIFFERENT channel sets
(12 vs 8 EEG derivations per the paper) -- kept as two separate `--which`
selections here (own subject pools, own results), same precedent as Sleep-EDF
vs HMC being kept separate rather than merged.

Data: /mnt/t9/dreem_dod_extracted/{dodh,dodo}/<uuid>.h5. HDF5 layout (verified
2026-07-07 against a real file, not assumed from docs):
  hypnogram: (n_epochs,) int64, values {-1:not-scored, 0:W, 1:N1, 2:N2, 3:N3,
    4:REM} -- matches our STAGE_MAP convention already (no string parsing).
  signals/eeg/<NAME>: (n_epochs*EPOCH_S*fs,) float32, fs in signals/eeg attrs.
  Channel dataset keys use underscores (C3_M2); the human-readable hyphenated
  name (C3-M2) is only in the root 'description' JSON attr -- reconstructed
  here by replacing '_'->'-' (verified consistent on the sample checked).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_sleep_section as sl  # noqa: E402  (reuses EPOCH_S/_fit_probe/evaluate)
import epilepsy_scorer as es  # noqa: E402  (window_features)

ROOT = os.environ.get("DOD_ROOT", "/mnt/t9/dreem_dod_extracted")

# Full EEG montage per cohort, in the dataset's own hyphenated naming
# (verified against real files -- DOD-H has 12, DOD-O has 8).
EEG_CHANNELS = {
    "dodh": ["C3-M2", "F4-M1", "F3-F4", "F3-M2", "F4-O2", "F3-O1",
             "FP1-F3", "FP1-M2", "FP1-O1", "FP2-F4", "FP2-M1", "FP2-O2"],
    "dodo": ["C3-M2", "C4-M1", "F3-F4", "F3-M2", "F3-O1", "F4-O2",
             "O1-M2", "O2-M1"],
}


def iter_recordings(which, limit=None):
    """(h5_path, subject_id) for every recording in the given cohort."""
    out = [(p, f"{which}_{os.path.splitext(os.path.basename(p))[0]}")
           for p in sorted(glob.glob(os.path.join(ROOT, which, "*.h5")))]
    return out[:limit] if limit else out


def load_epochs(h5_path, which):
    """HDF5 -> (X (n,C,EPOCH_S*fs) float32, y (n,) int 0-4). Hypnogram is
    already epoch-aligned (epoch i == seconds [i*EPOCH_S, (i+1)*EPOCH_S)) --
    no annotation parsing or wake-cropping needed, unlike EDF+-based loaders."""
    import h5py
    chans = EEG_CHANNELS[which]
    with h5py.File(h5_path, "r") as f:
        hyp = f["hypnogram"][:]
        fs = int(f["signals/eeg"].attrs["fs"])
        n = int(sl.EPOCH_S * fs)
        data = []
        for c in chans:
            key = c.replace("-", "_")
            if key not in f["signals/eeg"]:
                return None, None
            data.append(f[f"signals/eeg/{key}"][:])
        data = np.stack(data)
    X, y = [], []
    for i, label in enumerate(hyp):
        if label < 0:  # NOT SCORED
            continue
        seg = data[:, i * n:(i + 1) * n]
        if seg.shape[1] == n:
            X.append(seg); y.append(int(label))
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.asarray(y, int)


def _reve_channel_names(which):
    """Bipolar derivation -> anchor (first/active) electrode, same convention
    as atlas_sleep_hmc.py: C3-M2->C3, F3-F4->F3 (approximation for the
    scalp-scalp pairs, exact convention for the mastoid-referenced ones)."""
    return [c.split("-")[0] for c in EEG_CHANNELS[which]]


def make_embed_fn(model, which, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        ch = _reve_channel_names(which)
        return lambda X: embed_reve(X, ch, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (dod supports logbandpower, reve)")


def subject_splits(which, limit=None, seed_frac=(0.7, 0.15, 0.15)):
    subs = sorted({s for _, s in iter_recordings(which, limit)})
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def build_split(which, subjects, limit=None, embed_fn=None):
    recs = []
    for h5_path, subj in iter_recordings(which, limit):
        if subj not in subjects:
            continue
        X, y = load_epochs(h5_path, which)
        if X is None:
            continue
        rec = dict(y=y, patient=subj)
        rec["emb"] = embed_fn(X) if embed_fn is not None else None
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["dodh", "dodo"], required=True)
    ap.add_argument("--model", default="logbandpower")
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--identity-free", action="store_true")
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = subject_splits(a.which, a.limit)
    print(f"[dod-{a.which}] model={a.model} limit={a.limit} subjects train/dev/eval="
          f"{len(tr_s)}/{len(dv_s)}/{len(ev_s)}", flush=True)
    embed_fn = make_embed_fn(a.model, a.which, a.sfreq, dev_)
    splits = {}
    for name, subs in (("train", tr_s), ("dev", dv_s), ("eval", ev_s)):
        recs = build_split(a.which, subs, a.limit, embed_fn=embed_fn)
        tot = sum(len(r["y"]) for r in recs)
        splits[name] = recs
        print(f"  {name}: {len(recs)} recs, {tot} epochs, d={recs[0]['emb'].shape[1] if recs else 0}", flush=True)

    configs = [False, True] if (a.both or a.identity_free) else [False]
    results = {}
    for erase in configs:
        out = sl.evaluate(splits["train"], splits["dev"], splits["eval"], a.model, erase_patient=erase)
        results[erase] = out
        tag = " (identity-free)" if erase else ""
        print(f"\n[dod-{a.which}] {a.model}{tag} 5-class AASM staging:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        print(f"  (dev balanced-acc = {out['dev_bacc']*100:.1f}%  kappa = {out['dev_kappa']:.3f})")
    if False in results and True in results:
        d = results[False]["eval_kappa"] - results[True]["eval_kappa"]
        print(f"\n[dod-{a.which}] {a.model} identity-free Δ(eval kappa) = {d:+.3f} "
              f"(normal {results[False]['eval_kappa']:.3f} - idfree {results[True]['eval_kappa']:.3f})")


if __name__ == "__main__":
    main()
