"""NeuroTechX-Atlas affect section: FACED (Finer-grained Affective Computing
EEG Dataset, Synapse syn50614194). New TASK DOMAIN for the program (emotion
recognition, not sleep/epilepsy/BCI) and one of the 12 OpenEEGBench datasets
(github.com/braindecode/OpenEEGBench) we didn't have coverage of.

123 subjects, 32-channel EEG (30 real EEG + HEOL/HEOR EOG, excluded here),
28 video-clip trials each, 9-class emotion label (anger/disgust/fear/sadness/
neutral/amusement/inspiration/joy/tenderness). Uses the PRE-PROCESSED
``Processed_data/subXXX.pkl`` release (28,32,7500) float64 @ 250 Hz, 30s/trial
-- NOT the raw .bdf (Data.zip, ~22GB) -- the dataset's own Readme confirms
"The order of video clips in the pre-processed data was reorganized according
to the index of video clips as reported in Stimuli_info.xlsx", so trial index
i always corresponds to video (i+1) for every subject, and video->emotion
label is looked up once from Stimuli_info.xlsx (constant across subjects, no
per-subject randomized-order reconstruction needed).

Electrode order (post-processing, confirmed via Electrode_Location.xlsx's
note that cohort-1 recordings were reordered to match cohort-2 naming): real
monopolar 10-20 names throughout (Fp1/Fp2/Fz/F3/F4/... /O1/O2), so REVE gets
genuine channel positions -- no anchor-electrode approximation needed, unlike
several of the bipolar-montage sleep/epilepsy loaders in this program.

Reuses atlas_sleep_section.evaluate/_fit_probe directly (generic multi-class
classification + patient-id-free LEACE harness, not sleep-specific) rather
than re-implementing the same probe/erasure/metric plumbing a fourth time.
"""
from __future__ import annotations

import glob
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_sleep_section as sl  # noqa: E402  (reuses _fit_probe/evaluate)
import epilepsy_scorer as es  # noqa: E402  (window_features)

ROOT = os.environ.get("FACED_ROOT", "/data/datasets/eeg_fmri/faced")
PROCESSED_DIR = os.path.join(ROOT, "extracted", "Processed_data")
STIMULI_INFO = os.path.join(ROOT, "Stimuli_info.xlsx")

# Sequential channel order 1-32 per Electrode_Location.xlsx's second-cohort
# layout (post-processing canonical order); last 2 are EOG, excluded below.
EEG_CHANNELS = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1",
    "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8",
    "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7",
    "P8", "PO3", "PO4", "Oz", "O1", "O2",
]  # 30 real EEG channels; HEOR/HEOL (indices 31,32) excluded
SFREQ = 250.0


def _build_label_map():
    """Stimuli_info.xlsx -> {video_index (1-28 int): emotion_name (str)}.
    Neutral videos have "\\" in Targeted Emotion -- use Valence ("Neutral")
    for those instead of the placeholder."""
    import pandas as pd
    df = pd.read_excel(STIMULI_INFO)
    # Footer "Notes: ..." rows leak text into the Video index column (not
    # just NaN) -- coerce to numeric and drop anything that doesn't parse,
    # rather than relying on dropna alone.
    df["Video index"] = pd.to_numeric(df["Video index"], errors="coerce")
    df = df.dropna(subset=["Video index"])
    out = {}
    for _, row in df.iterrows():
        idx = int(row["Video index"])
        valence = str(row["Valence"]).strip()
        emotion = str(row["Targeted Emotion"]).strip()
        out[idx] = valence if emotion in ("\\", "nan", "") else emotion
    return out


def _label_to_int_map(label_map):
    classes = sorted(set(label_map.values()))
    return {c: i for i, c in enumerate(classes)}


def iter_recordings(limit=None):
    """(pkl_path, subject_id) for every FACED subject."""
    out = []
    for pkl in sorted(glob.glob(os.path.join(PROCESSED_DIR, "sub*.pkl"))):
        m = re.search(r"(sub\d+)\.pkl$", pkl)
        if m:
            out.append((pkl, m.group(1)))
    return out[:limit] if limit else out


def load_epochs(pkl_path, label_map, class_map):
    """subXXX.pkl -> (X (28,30,7500) float32, y (28,) int) -- trial i is
    always video (i+1) per the dataset's own reorganization guarantee."""
    with open(pkl_path, "rb") as f:
        arr = pickle.load(f)
    X = arr[:, :len(EEG_CHANNELS), :].astype(np.float32)
    y = np.asarray([class_map[label_map[i + 1]] for i in range(X.shape[0])], dtype=int)
    return X, y


def _reve_channel_names():
    return list(EEG_CHANNELS)


def make_embed_fn(model, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        ch = _reve_channel_names()
        return lambda X: embed_reve(X, ch, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (faced supports logbandpower, reve)")


def subject_splits(limit=None, seed_frac=(0.7, 0.15, 0.15)):
    subs = sorted({s for _, s in iter_recordings(limit)})
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def build_split(subjects, limit=None, embed_fn=None):
    label_map = _build_label_map()
    class_map = _label_to_int_map(label_map)
    recs = []
    for pkl, subj in iter_recordings(limit):
        if subj not in subjects:
            continue
        X, y = load_epochs(pkl, label_map, class_map)
        rec = dict(y=y, patient=subj)
        rec["emb"] = embed_fn(X) if embed_fn is not None else None
        recs.append(rec)
    return recs


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="logbandpower")
    ap.add_argument("--sfreq", type=float, default=SFREQ)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--identity-free", action="store_true")
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = subject_splits(a.limit)
    print(f"[faced] model={a.model} limit={a.limit} subjects train/dev/eval="
          f"{len(tr_s)}/{len(dv_s)}/{len(ev_s)}", flush=True)
    embed_fn = make_embed_fn(a.model, a.sfreq, dev_)
    splits = {}
    for name, subs in (("train", tr_s), ("dev", dv_s), ("eval", ev_s)):
        recs = build_split(subs, a.limit, embed_fn=embed_fn)
        tot = sum(len(r["y"]) for r in recs)
        splits[name] = recs
        print(f"  {name}: {len(recs)} recs, {tot} trials, d={recs[0]['emb'].shape[1] if recs else 0}", flush=True)

    configs = [False, True] if (a.both or a.identity_free) else [False]
    results = {}
    for erase in configs:
        out = sl.evaluate(splits["train"], splits["dev"], splits["eval"], a.model, erase_patient=erase)
        results[erase] = out
        tag = " (identity-free)" if erase else ""
        print(f"\n[faced] {a.model}{tag} 9-class emotion recognition:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        print(f"  (dev balanced-acc = {out['dev_bacc']*100:.1f}%  kappa = {out['dev_kappa']:.3f})")
    if False in results and True in results:
        d = results[False]["eval_kappa"] - results[True]["eval_kappa"]
        print(f"\n[faced] {a.model} identity-free Δ(eval kappa) = {d:+.3f} "
              f"(normal {results[False]['eval_kappa']:.3f} - idfree {results[True]['eval_kappa']:.3f})")


if __name__ == "__main__":
    main()
