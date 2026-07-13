"""NeuroTechX-Atlas sleep-staging section, dataset 5: ISRUC-Sleep (subgroups
I/II/III). A 6th sleep dataset (after Sleep-EDF/HMC/DOD-H/DOD-O/Ear-EEG) --
polysomnography, standard AASM 30s-epoch staging, but a distinct clinical
population (adult sleep-disorder patients, some with 2 recorded nights) and
distinct EEG referencing (bipolar-to-contralateral-mastoid, e.g. "F3-A2").

Directory layout differs by subgroup (real, not a bug to normalize away):
  subgroup 1: {ROOT}/1/<subj>/<subj>.rec + <subj>_{1,2}.txt   (1 night, 100 subj)
  subgroup 2: {ROOT}/2/<subj>/<night>/<night>.rec + ..._{1,2}.txt  (2 nights, 8 subj)
  subgroup 3: {ROOT}/3/<subj>/<subj>.rec + <subj>_{1,2}.txt   (1 night, 10 subj)
Patient ids are prefixed with the subgroup ("g1_<subj>") since subject
numbering restarts at 1 in each subgroup and would otherwise collide.

Two independent human scorers provide per-30s-epoch hypnograms
(``<name>_1.txt``/``<name>_2.txt``); this loader uses scorer 1 as ground
truth (single-scorer convention, the common choice in ISRUC ML literature,
unlike Helsinki's 3-annotator majority-vote design). ISRUC's on-disk label
convention is {0,1,2,3,5} (no "4") -> remapped to the AASM 0-4 (W/N1/N2/N3/
REM) convention shared with Sleep-EDF/HMC/DOD/Ear-EEG.

``.rec`` files are plain EDF renamed; MNE's ``read_raw_edf`` refuses non-
``.edf`` extensions by inspecting the path, so a temporary ``.edf`` symlink
is used to read them without copying the underlying data.

REVE channel handling: EEG channels are bipolar-to-mastoid derivations
(F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1) -- same anchor-electrode
convention as Dreem DOD/CHB-MIT (first electrode of each pair).
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_sleep_section as sl  # noqa: E402
import epilepsy_scorer as es  # noqa: E402

ROOT = os.environ.get("ISRUC_ROOT", "/data/datasets/eeg_fmri/isruc_sleep")
EEG_CHANNELS = ["F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1"]
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}  # ISRUC {0,1,2,3,5} -> AASM 0-4


def iter_recordings(limit=None):
    """(rec_path, hyp1_path, patient_id) for every ISRUC recording."""
    out = []
    for grp in ("1", "2", "3"):
        grp_root = os.path.join(ROOT, grp)
        if not os.path.isdir(grp_root):
            continue
        for subj_dir in sorted(glob.glob(os.path.join(grp_root, "*"))):
            subj = os.path.basename(subj_dir)
            # subgroup 2 nests an extra <night> level; 1/3 don't
            night_dirs = sorted(glob.glob(os.path.join(subj_dir, "*"))) \
                if grp == "2" else [subj_dir]
            for night_dir in night_dirs:
                name = os.path.basename(night_dir)
                rec = os.path.join(night_dir, f"{name}.rec")
                hyp1 = os.path.join(night_dir, f"{name}_1.txt")
                if os.path.exists(rec) and os.path.exists(hyp1):
                    out.append((rec, hyp1, f"g{grp}_{subj}"))
    return out[:limit] if limit else out


def load_epochs(rec_path, hyp1_path):
    """.rec (EDF) + scorer-1 hypnogram -> (X (n,6,30*fs), y (n,))."""
    import mne
    mne.set_log_level("error")
    with open(hyp1_path) as f:
        raw_labels = [int(x) for x in f.read().split() if x.strip() != ""]
    y = np.array([LABEL_MAP[v] for v in raw_labels if v in LABEL_MAP], dtype=int)

    with tempfile.TemporaryDirectory() as d:
        link = os.path.join(d, "x.edf")
        os.symlink(os.path.abspath(rec_path), link)
        raw = mne.io.read_raw_edf(link, preload=True, verbose="error")
        present = [c for c in EEG_CHANNELS if c in raw.ch_names]
        if len(present) < len(EEG_CHANNELS):
            return None, None
        raw.pick(present).reorder_channels(present)
        sfreq = raw.info["sfreq"]
        n = int(sl.EPOCH_S * sfreq)
        data = raw.get_data()

    n_ep = min(len(y), data.shape[1] // n)
    if n_ep == 0:
        return None, None
    X = np.stack([data[:, i * n:(i + 1) * n] for i in range(n_ep)]).astype(np.float32)
    return X, y[:n_ep]


def _reve_channel_names():
    return [c.split("-")[0] for c in EEG_CHANNELS]


def make_embed_fn(model, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        ch = _reve_channel_names()
        return lambda X: embed_reve(X, ch, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (isruc supports logbandpower, reve)")


def subject_splits(limit=None, seed_frac=(0.7, 0.15, 0.15)):
    subs = sorted({s for _, _, s in iter_recordings(limit)})
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def build_split(subjects, limit=None, embed_fn=None):
    recs = []
    for rec, hyp1, subj in iter_recordings(limit):
        if subj not in subjects:
            continue
        X, y = load_epochs(rec, hyp1)
        if X is None:
            continue
        rec_d = dict(y=y, patient=subj)
        rec_d["emb"] = embed_fn(X) if embed_fn is not None else None
        recs.append(rec_d)
    return recs


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="logbandpower")
    ap.add_argument("--sfreq", type=float, default=200.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--identity-free", action="store_true")
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = subject_splits(a.limit)
    print(f"[isruc] model={a.model} limit={a.limit} subjects train/dev/eval="
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
        print(f"\n[isruc] {a.model}{tag} 5-class AASM staging:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        print(f"  (dev balanced-acc = {out['dev_bacc']*100:.1f}%  kappa = {out['dev_kappa']:.3f})")
    if False in results and True in results:
        d = results[False]["eval_kappa"] - results[True]["eval_kappa"]
        print(f"\n[isruc] {a.model} identity-free Δ(eval kappa) = {d:+.3f} "
              f"(normal {results[False]['eval_kappa']:.3f} - idfree {results[True]['eval_kappa']:.3f})")


if __name__ == "__main__":
    main()
