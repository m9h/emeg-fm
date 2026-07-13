"""NeuroTechX-Atlas epilepsy section, dataset 4: CHB-MIT Scalp EEG (PhysioNet
1.0.0, Shoeb 2010). A 4th independent epilepsy population -- pediatric (23
subjects, ages 1.5-22) intractable-seizure scalp EEG, continuous multi-hour
recordings, bipolar 10-20 derivation montage (not the monopolar TUSZ/Helsinki
convention). Same SzCORE scoring + patient-identity-free (LEACE) axis as
TUSZ/SeizeIT2/Helsinki (reuses epilepsy_scorer.py).

Per-subject ``chbNN-summary.txt`` gives, per EDF file, seizure count and
Start/End Time offsets in seconds relative to that file's own start -- two
on-disk formats coexist (single "Seizure Start Time:"/"Seizure End Time:" for
1-seizure files, numbered "Seizure N Start Time:" for multi-seizure files,
plus chb24 which omits the numbering AND the file start/end-time lines
entirely) -- ``_parse_summary`` handles all three via one seizure-time regex
applied per per-file block, not by keying off the numbering style.

Known dataset quirk, handled here: **chb21 is the SAME patient as chb01**
(re-recorded ~1.5y later per the corpus documentation) -- treated as a single
patient id (``chb01``) for the identity-free split so this doesn't leak
identity across a naive folder-name-as-subject assumption.

REVE channel handling: CHB-MIT channels are bipolar derivations (e.g.
"FP1-F7"), not monopolar 10-20 sites, so there is no single standard-montage
position per channel. Reuses the same anchor-electrode convention as Dreem
DOD (``atlas_sleep_dod.py``): the first electrode of each bipolar pair stands
in for that channel's spatial position (duplicate anchors across channels are
fine -- same precedent as DOD-O's repeated F3 anchor).
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

ROOT = os.environ.get("CHBMIT_ROOT", "/data/datasets/eeg_fmri/chbmit/1.0.0")
WIN_S = 10.0
# Common 18-channel bipolar montage present in nearly all CHB-MIT subjects.
STD_BIPOLAR = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
_SAME_PATIENT = {"chb21": "chb01"}  # chb21 re-records chb01's patient, ~1.5y later


def _patient_id(subj):
    return _SAME_PATIENT.get(subj, subj)


def _parse_summary(text):
    """chbNN-summary.txt -> {filename: [(start_s, end_s), ...]}.

    Handles: "Seizure Start Time:"/"Seizure End Time:" (1-seizure files),
    "Seizure N Start Time:"/"Seizure N End Time:" (multi-seizure files), and
    chb24's variant (numbered seizures, no File Start/End Time lines) -- all
    via one regex per block, independent of the numbering style."""
    out = {}
    blocks = re.split(r"(?=File Name: )", text)
    for block in blocks:
        m = re.search(r"File Name:\s*(\S+)", block)
        if not m:
            continue
        fname = m.group(1)
        starts = [float(x) for x in re.findall(
            r"Seizure(?:\s*\d+)?\s*Start Time:\s*([\d.]+)\s*seconds", block)]
        ends = [float(x) for x in re.findall(
            r"Seizure(?:\s*\d+)?\s*End Time:\s*([\d.]+)\s*seconds", block)]
        out[fname] = list(zip(starts, ends))
    return out


def iter_recordings(limit=None):
    """(edf_path, seizure_events, subject_id) for every CHB-MIT recording."""
    out = []
    for summary_path in sorted(glob.glob(os.path.join(ROOT, "chb*", "*-summary.txt"))):
        subj = os.path.basename(os.path.dirname(summary_path))
        with open(summary_path, errors="ignore") as f:
            events_by_file = _parse_summary(f.read())
        for fname, events in sorted(events_by_file.items()):
            edf = os.path.join(ROOT, subj, fname)
            if os.path.exists(edf):
                out.append((edf, events, subj))
    return out[:limit] if limit else out


def load_windows(edf, win_s=WIN_S, sfreq_out=100.0):
    import mne
    mne.set_log_level("error")
    try:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
    except Exception:
        return None, None
    raw.rename_channels({c: c.strip().upper() for c in raw.ch_names})
    # some files carry duplicate channel names (e.g. repeated T8-P8) -- keep
    # the first occurrence of each wanted channel
    present = []
    seen = set()
    for c in STD_BIPOLAR:
        if c in raw.ch_names and c not in seen:
            present.append(c); seen.add(c)
    if len(present) < len(STD_BIPOLAR):
        return None, None
    raw.pick(present).reorder_channels(present)
    if raw.info["sfreq"] != sfreq_out:
        raw.resample(sfreq_out, verbose="error")
    x = raw.get_data()
    w = int(win_s * sfreq_out)
    n = x.shape[1] // w
    if n == 0:
        return None, None
    X = np.stack([x[:, i * w:(i + 1) * w] for i in range(n)]).astype(np.float32)
    starts = np.arange(n) * win_s
    return X, starts


def label_windows(starts, win_s, seiz_events):
    y = np.zeros(len(starts), int)
    for i, s in enumerate(starts):
        e = s + win_s
        if any(s < b and a < e for a, b in seiz_events):
            y[i] = 1
    return y


def _reve_channel_names():
    return [c.split("-")[0] for c in STD_BIPOLAR]


def make_embed_fn(model, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        ch = _reve_channel_names()
        return lambda X: embed_reve(X, ch, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (chbmit supports logbandpower, reve)")


def subject_splits(limit=None, seed_frac=(0.6, 0.2, 0.2)):
    subs = sorted({_patient_id(s) for _, _, s in iter_recordings(limit)})
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def build_split(subjects, sfreq_out, limit=None, embed_fn=None):
    recs = []
    for edf, events, subj in iter_recordings(limit):
        if _patient_id(subj) not in subjects:
            continue
        X, starts = load_windows(edf, WIN_S, sfreq_out)
        if X is None:
            continue
        y = label_windows(starts, WIN_S, events)
        dur_s = starts[-1] + WIN_S if len(starts) else 0.0
        rec = dict(y=y, starts=starts, seiz=events, patient=_patient_id(subj), dur_h=dur_s / 3600.0)
        rec["emb"] = embed_fn(X) if embed_fn is not None else None
        recs.append(rec)
    return recs


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
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, ytr)
    return clf, sc, er


def _score_recs(recs, clf, sc, er):
    scores, starts, seiz, durs = [], [], [], []
    for r in recs:
        Xs = sc.transform(r["emb"])
        if er is not None:
            Xs = er.transform(Xs)
        p = clf.predict_proba(Xs)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(Xs)
        scores.append(np.asarray(p, np.float32)); starts.append(np.asarray(r["starts"], np.float32))
        seiz.append(list(r["seiz"])); durs.append(r["dur_h"] * 3600.0)
    return dict(scores=scores, starts=starts, seiz=seiz, durs=durs)


def evaluate(train, dev, evl, model, erase_patient=False):
    Xtr = np.concatenate([r["emb"] for r in train])
    ytr = np.concatenate([r["y"] for r in train])
    pid = None
    if erase_patient:
        pmap = {p: i for i, p in enumerate(sorted({r["patient"] for r in train}))}
        pid = np.concatenate([[pmap[r["patient"]]] * len(r["y"]) for r in train])
    clf, sc, er = _fit_probe(Xtr, ytr, model, erase_patient=pid)
    D, E = _score_recs(dev, clf, sc, er), _score_recs(evl, clf, sc, er)
    dev_sz = es.szcore_curve(D["scores"], D["starts"], WIN_S, D["seiz"], D["durs"])
    eval_sz = es.szcore_curve(E["scores"], E["starts"], WIN_S, E["seiz"], E["durs"])
    op = es.szcore_operating_point(dev_sz, eval_sz)
    return dict(
        szcore_auc=es.event_sens_at_fa_auc(eval_sz, 0.1, 10.0, fa_key="fa_per_day"),
        szcore_sens1=es.sensitivity_at_fa_day(eval_sz, 1.0),
        szcore_sens10=es.sensitivity_at_fa_day(eval_sz, 10.0),
        op_f1=op["eval_f1"], op_sens=op["eval_sensitivity"], op_faday=op["eval_fa_per_day"])


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
    print(f"[chbmit] model={a.model} limit={a.limit} patients train/dev/eval="
          f"{len(tr_s)}/{len(dv_s)}/{len(ev_s)}", flush=True)
    embed_fn = make_embed_fn(a.model, a.sfreq, dev_)
    splits = {}
    for name, subs in (("train", tr_s), ("dev", dv_s), ("eval", ev_s)):
        recs = build_split(subs, a.sfreq, a.limit, embed_fn=embed_fn)
        pos = sum(int(r["y"].sum()) for r in recs)
        tot = sum(len(r["y"]) for r in recs)
        print(f"  {name}: {len(recs)} recs, {tot} win ({pos} seizure)", flush=True)
        splits[name] = recs

    configs = [False, True] if (a.both or a.identity_free) else [False]
    results = {}
    for erase in configs:
        out = evaluate(splits["train"], splits["dev"], splits["eval"], a.model, erase_patient=erase)
        results[erase] = out
        tag = " (identity-free)" if erase else ""
        print(f"\n[chbmit] {a.model}{tag} SzCORE:")
        print(f"  eval AUC[.1-10 FP/day] = {out['szcore_auc']:.3f}  sens@1/d={out['szcore_sens1']*100:.1f}%"
              f"  sens@10/d={out['szcore_sens10']*100:.1f}%  F1={out['op_f1']:.3f} @ {out['op_faday']:.1f} FP/day")
    if False in results and True in results:
        d = results[False]["szcore_auc"] - results[True]["szcore_auc"]
        print(f"\n[chbmit] {a.model} identity-free Δ(eval AUC) = {d:+.3f}")


if __name__ == "__main__":
    main()
