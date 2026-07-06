"""NeuroTechX-Atlas CogNeuro section, extra datasets: independent NEMAR/OpenNeuro
replications of ERP CORE-style components, beyond the ERP CORE battery itself
(atlas_cogneuro_section.py). Reuses its extract()/loso()/within_subject_ba()/
_norm_ba() directly -- own loader per dataset (raw BIDS, not MOABB-wrapped).

Datasets (found via NEMAR search, 2026-07-04):
  n170     ds002718  Wakeman-Henson face processing, 18 subj, EEGLAB .set,
                     face (famous/unfamiliar) vs scrambled -- canonical N170.
  ern      ds004883  2-site flanker registered report, 172 subj, error vs
                     correct trials -- canonical ERN (+ LRP from the same data).
  p3_aud   ds003061  auditory oddball, 13 subj, target vs standard tone.
  p3_vis   ds006018  active visual oddball (P3b), 127 subj, target vs standard.

Why these matter: independent-cohort replication of the ERP CORE result (REVE
beats classical on evoked/cognitive components, Δidentity≈0) at MUCH larger N
for ern/p3_vis -- ERP CORE's own N=40 is shared across all 7 components, so an
independent 127-172 subject replication is a real generalization test.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_cogneuro_section import extract, loso, within_subject_ba  # noqa: E402

ROOT = os.environ.get("NEMAR_COGNEURO_ROOT", "/mnt/t9/nemar_cogneuro")


def _eeg_channel_names(channels_tsv):
    df = pd.read_csv(channels_tsv, sep="\t")
    return df[df["type"] == "EEG"]["name"].tolist()


_STD1005_CACHE = None


def _reve_names_from_electrode_positions(electrodes_tsv, eeg_ch_names):
    """Map ds002718's generic channel names ('EEG001'...) to real standard_1005
    electrode names via nearest-neighbor 3D position matching, for REVE's
    name-based lookup. electrodes.tsv gives REAL 3D positions (unlike the
    bipolar-derivation datasets) but in an uncalibrated unit/origin -- both
    point sets are rescaled to unit RMS radius before matching (a documented
    approximation: mean residual ~0.19 in normalized-radius units on the
    sample checked, not exact registration). Per-subject file (positions can
    differ slightly/have missing electrodes across subjects -- NaN rows dropped)."""
    global _STD1005_CACHE
    import mne
    if _STD1005_CACHE is None:
        mont = mne.channels.make_standard_montage("standard_1005")
        cpos = mont.get_positions()["ch_pos"]
        names = list(cpos.keys())
        cxyz = np.stack([cpos[n] for n in names])
        _STD1005_CACHE = (names, cxyz, np.sqrt((cxyz**2).sum(axis=1)).mean())
    names, cxyz, crms = _STD1005_CACHE

    df = pd.read_csv(electrodes_tsv, sep="\t")
    df = df[df["name"].isin(eeg_ch_names)].dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    pos = df[["x", "y", "z"]].to_numpy()
    rms = np.sqrt((pos**2).sum(axis=1)).mean()
    pos_n, cxyz_n = pos / rms, cxyz / crms
    mapping = {}
    for i, row in df.iterrows():
        d = np.linalg.norm(cxyz_n - pos_n[i], axis=1)
        mapping[row["name"]] = names[int(np.argmin(d))]
    # channels with no valid position (dropped above) fall back to their own
    # generic name (embed_reve will simply not find a position for them).
    return [mapping.get(c, c) for c in eeg_ch_names]


def _epoch_one(raw, events_df, label_fn, tmin, tmax, fmax, sfreq_out, eeg_chs):
    """Shared epoching: pick EEG chs, filter, epoch per event, resample.
    events_df must have an 'onset' column (seconds) and whatever columns
    label_fn needs; label_fn(row) -> int label or None to skip."""
    import mne
    raw = raw.copy().pick(eeg_chs)
    raw.load_data()
    raw.filter(0.5, fmax, verbose="error")
    sfreq = raw.info["sfreq"]
    X, y = [], []
    n = int((tmax - tmin) * sfreq)
    data = raw.get_data()
    for _, row in events_df.iterrows():
        lab = label_fn(row)
        if lab is None:
            continue
        onset_samp = int(round((row["onset"] + tmin) * sfreq))
        if onset_samp < 0 or onset_samp + n > data.shape[1]:
            continue
        X.append(data[:, onset_samp:onset_samp + n])
        y.append(lab)
    if not X:
        return None, None
    X = np.stack(X).astype(np.float32)
    if sfreq != sfreq_out:
        import scipy.signal
        n_out = int(round(X.shape[-1] * sfreq_out / sfreq))
        X = scipy.signal.resample(X, n_out, axis=-1).astype(np.float32)
    return X, np.asarray(y, int)


def load_n170_ds002718(fmax=45.0, sfreq_out=200.0, max_subjects=None):
    """Face (famous/unfamiliar) vs scrambled -- canonical N170 contrast."""
    import mne
    mne.set_log_level("error")
    subs = sorted(glob.glob(os.path.join(ROOT, "ds002718", "sub-*")))
    if max_subjects:
        subs = subs[:max_subjects]
    all_X, all_y, all_subj, ch_names, reve_ch_names = [], [], [], None, None
    for sd in subs:
        subj = os.path.basename(sd)
        eeg_dir = os.path.join(sd, "eeg")
        sets = glob.glob(os.path.join(eeg_dir, "*_eeg.set"))
        evs = glob.glob(os.path.join(eeg_dir, "*_events.tsv"))
        chs = glob.glob(os.path.join(eeg_dir, "*_channels.tsv"))
        if not (sets and evs and chs):
            continue
        eeg_chs = _eeg_channel_names(chs[0])
        raw = mne.io.read_raw_eeglab(sets[0], preload=False, verbose="error")
        events_df = pd.read_csv(evs[0], sep="\t")
        events_df = events_df[events_df["event_type"] == "faces"].copy()
        events_df["onset"] = events_df["onset"].astype(float)

        def label_fn(row):
            ft = row.get("face_type")
            if ft in ("famous", "unfamiliar"):
                return 1
            if ft == "scrambled":
                return 0
            return None

        X, y = _epoch_one(raw, events_df, label_fn, -0.2, 0.8, fmax, sfreq_out, eeg_chs)
        if X is None:
            continue
        all_X.append(X); all_y.append(y); all_subj.extend([subj] * len(y))
        if ch_names is None:
            ch_names = eeg_chs
            elec_tsv = glob.glob(os.path.join(eeg_dir, "*_electrodes.tsv"))
            reve_ch_names = (_reve_names_from_electrode_positions(elec_tsv[0], eeg_chs)
                             if elec_tsv else eeg_chs)
    return (np.concatenate(all_X), np.concatenate(all_y),
            np.asarray(all_subj), ch_names, reve_ch_names)


def load_ern_ds004883(fmax=45.0, sfreq_out=200.0, max_subjects=None, session="ses-1"):
    """Error (err) vs correct (cor) response-locked trials -- canonical ERN
    contrast. 172-subject 2-site flanker registered report -- much larger than
    ERP CORE's shared N=40. Only `session` used per subject (task name varies
    by counterbalance order: ffa/ffb/ffc) to keep one recording/subject, like
    the other loaders here; the dataset's other 2 sessions/subject are
    available for a future within-subject task-generalization extension."""
    import mne
    mne.set_log_level("error")
    subs = sorted(glob.glob(os.path.join(ROOT, "ds004883", "sub-*")))
    if max_subjects:
        subs = subs[:max_subjects]
    all_X, all_y, all_subj, ch_names, reve_ch_names = [], [], [], None, None
    for sd in subs:
        subj = os.path.basename(sd)
        eeg_dir = os.path.join(sd, session, "eeg")
        sets = glob.glob(os.path.join(eeg_dir, "*_eeg.set"))
        evs = glob.glob(os.path.join(eeg_dir, "*_events.tsv"))
        chs = glob.glob(os.path.join(eeg_dir, "*_channels.tsv"))
        if not (sets and evs and chs):
            continue
        eeg_chs = _eeg_channel_names(chs[0])
        raw = mne.io.read_raw_eeglab(sets[0], preload=False, verbose="error")
        events_df = pd.read_csv(evs[0], sep="\t")
        events_df = events_df[events_df["trial_type"].isin(["err", "cor"])].copy()
        events_df["onset"] = events_df["onset"].astype(float)

        def label_fn(row):
            return 1 if row["trial_type"] == "err" else 0

        # response-locked: -0.4 to 0.8 s around the response event itself
        X, y = _epoch_one(raw, events_df, label_fn, -0.4, 0.8, fmax, sfreq_out, eeg_chs)
        if X is None:
            continue
        all_X.append(X); all_y.append(y); all_subj.extend([subj] * len(y))
        if ch_names is None:
            ch_names = eeg_chs
            elec_tsv = glob.glob(os.path.join(eeg_dir, "*_electrodes.tsv"))
            reve_ch_names = (_reve_names_from_electrode_positions(elec_tsv[0], eeg_chs)
                             if elec_tsv else eeg_chs)
    return (np.concatenate(all_X), np.concatenate(all_y),
            np.asarray(all_subj), ch_names, reve_ch_names)


DATASETS = {"n170": load_n170_ds002718, "ern": load_ern_ds004883}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--sfreq", type=float, default=200.0)
    ap.add_argument("--fmax", type=float, default=45.0)
    ap.add_argument("--max-subjects", type=int, default=None)
    a = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[cogneuro-extra] dataset={a.dataset} model={a.model}", flush=True)
    X, y, subj, ch_names, reve_ch_names = DATASETS[a.dataset](a.fmax, a.sfreq, a.max_subjects)
    n_subj = len(set(subj.tolist()))
    print(f"  {len(y)} trials, {n_subj} subjects, d_in={X.shape[1]}ch x {X.shape[2]}samp", flush=True)

    # REVE needs real electrode names (nearest-neighbor mapped from generic
    # 'EEGnnn' labels via 3D position matching); classical/braindecode use the
    # raw channel identity (doesn't matter for band-power features).
    names_for_model = reve_ch_names if a.model == "reve" else ch_names
    emb = extract(a.model, X, names_for_model, a.sfreq, dev)
    nc = len(set(y.tolist()))
    losoc = loso(emb, y, subj, nc, erase_identity=False)
    idf = loso(emb, y, subj, nc, erase_identity=True)
    wsub = within_subject_ba(emb, y, subj, nc)
    print(f"\n[cogneuro-extra] {a.dataset} {a.model}: LOSO={losoc*100:.1f}% "
          f"within={wsub*100:.1f}% idfree={idf*100:.1f}% idΔ={(losoc-idf)*100:+.1f}")


if __name__ == "__main__":
    main()
