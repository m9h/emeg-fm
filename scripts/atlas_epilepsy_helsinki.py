"""NeuroTechX-Atlas epilepsy section, dataset 3: Helsinki neonatal seizure EEG
(Zenodo 10.5281/zenodo.2547147, Stevenson et al.).

A third, independent epilepsy dataset -- Helsinki is one of NeuroAtlas's actual
named epilepsy datasets. Unlike TUSZ (clinical adult scalp EEG, Temple) and
SeizeIT2 (wearable behind-the-ear, adult), Helsinki is **neonatal** (79 term/
near-term infants, continuous ~1-12h EEG, 19-channel 10-20 montage) with THREE
independent expert annotators per recording -- ground truth here is the
per-second majority-vote consensus (>=2 of 3 agree), the standard convention
for this corpus. A third population entirely (developmental/neonatal vs adult
clinical vs adult wearable) further stress-tests whether the TUSZ/SeizeIT2
identity-trap pattern generalizes.

Data: /mnt/t9/helsinki_neonatal/ -- eegN.edf (N=1..79) + annotations_2017_{A,B,C}.csv
(each: rows=1 Hz samples, columns=recording N, 0/1 seizure mask per annotator).
Same SzCORE scoring + patient-identity-free (LEACE) axis as TUSZ/SeizeIT2 (reuses
epilepsy_scorer.py). Real 10-20 electrode names (Fp1/Fp2/F3/F4/C3/C4/P3/P4/O1/O2/
F7/F8/T3/T4/T5/T6/Fz/Cz/Pz) -> BOTH classical and REVE supported (unlike SeizeIT2's
proprietary wearable labels).
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

ROOT = os.environ.get("HELSINKI_ROOT", "/mnt/t9/helsinki_neonatal")
WIN_S = 10.0
ANNOTATORS = ["A", "B", "C"]
STD19 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
         "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
_TL_ALIAS = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _to_mne_casing(ch_names):
    """Map to standard_1005 canonical casing + old-10-20 aliases (T3->T7 etc),
    same convention as atlas_bci_section -- required for REVE's name lookup."""
    import mne
    canon = {c.upper(): c for c in mne.channels.make_standard_montage("standard_1005").ch_names}
    out = []
    for c in ch_names:
        u = _TL_ALIAS.get(c.strip().upper(), c.strip().upper())
        out.append(canon.get(u, c))
    return out


def _load_consensus(recording_num):
    """Majority-vote (>=2 of 3 annotators) 1 Hz seizure mask for one recording
    column. Returns a bool array, one sample per second."""
    cols = []
    for a in ANNOTATORS:
        path = os.path.join(ROOT, f"annotations_2017_{a}.csv")
        with open(path) as f:
            header = f.readline().strip().split(",")
            idx = header.index(str(recording_num))
            vals = [int(line.strip().split(",")[idx]) for line in f]
        cols.append(np.asarray(vals, dtype=int))
    n = min(len(c) for c in cols)
    stacked = np.stack([c[:n] for c in cols])
    return (stacked.sum(axis=0) >= 2)


def _mask_to_events(mask, sfreq_mask=1.0):
    """1 Hz boolean mask -> [(start_s, stop_s)] seizure intervals."""
    events = []
    in_evt, start = False, 0.0
    for i, v in enumerate(mask):
        t = i / sfreq_mask
        if v and not in_evt:
            in_evt, start = True, t
        elif not v and in_evt:
            events.append((start, t)); in_evt = False
    if in_evt:
        events.append((start, len(mask) / sfreq_mask))
    return events


def iter_recordings(limit=None):
    """(edf_path, recording_num, subject_id) for every Helsinki recording."""
    out = []
    for edf in sorted(glob.glob(os.path.join(ROOT, "eeg*.edf")),
                      key=lambda p: int(re.search(r"eeg(\d+)\.edf", p).group(1))):
        n = int(re.search(r"eeg(\d+)\.edf", edf).group(1))
        out.append((edf, n, f"neonate{n:03d}"))
    return out[:limit] if limit else out


def load_windows(edf, win_s=WIN_S, sfreq_out=100.0):
    import mne
    mne.set_log_level("error")
    try:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
    except Exception:
        return None, None, None
    # Helsinki's raw channel names are mixed-case ("Fp1", not "FP1" like TUSZ) --
    # uppercase after stripping so they match STD19's all-caps convention (else
    # every recording's channel-match fails silently -> 0 recordings loaded).
    ren = {c: re.sub(r"^EEG\s+|-REF$", "", c).strip().upper() for c in raw.ch_names}
    raw.rename_channels(ren)
    present = [c for c in STD19 if c in raw.ch_names]
    if len(present) < len(STD19):
        return None, None, None
    raw.pick(present).reorder_channels(present)
    if raw.info["sfreq"] != sfreq_out:
        raw.resample(sfreq_out, verbose="error")
    x = raw.get_data()
    w = int(win_s * sfreq_out)
    n = x.shape[1] // w
    if n == 0:
        return None, None, None
    X = np.stack([x[:, i * w:(i + 1) * w] for i in range(n)]).astype(np.float32)
    starts = np.arange(n) * win_s
    return X, starts, present


def label_windows(starts, win_s, seiz_events):
    y = np.zeros(len(starts), int)
    for i, s in enumerate(starts):
        e = s + win_s
        if any(s < b and a < e for a, b in seiz_events):
            y[i] = 1
    return y


def make_embed_fn(model, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        ch = _to_mne_casing(STD19)
        return lambda X: embed_reve(X, ch, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (helsinki supports logbandpower, reve)")


def subject_splits(limit=None, seed_frac=(0.6, 0.2, 0.2)):
    """Only 79 neonates total -> more dev/eval share than the larger corpora."""
    recs = sorted({s for _, _, s in iter_recordings(limit)})
    n = len(recs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(recs[:n_tr]), set(recs[n_tr:n_tr + n_dv]), set(recs[n_tr + n_dv:])


def build_split(subjects, sfreq_out, limit=None, embed_fn=None):
    recs = []
    for edf, num, subj in iter_recordings(limit):
        if subj not in subjects:
            continue
        X, starts, chs = load_windows(edf, WIN_S, sfreq_out)
        if X is None:
            continue
        mask = _load_consensus(num)
        seiz = _mask_to_events(mask)
        y = label_windows(starts, WIN_S, seiz)
        dur_s = starts[-1] + WIN_S if len(starts) else 0.0
        rec = dict(y=y, starts=starts, seiz=seiz, patient=subj, dur_h=dur_s / 3600.0)
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
    print(f"[helsinki] model={a.model} limit={a.limit} neonates train/dev/eval="
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
        print(f"\n[helsinki] {a.model}{tag} SzCORE:")
        print(f"  eval AUC[.1-10 FP/day] = {out['szcore_auc']:.3f}  sens@1/d={out['szcore_sens1']*100:.1f}%"
              f"  sens@10/d={out['szcore_sens10']*100:.1f}%  F1={out['op_f1']:.3f} @ {out['op_faday']:.1f} FP/day")
    if False in results and True in results:
        d = results[False]["szcore_auc"] - results[True]["szcore_auc"]
        print(f"\n[helsinki] {a.model} identity-free Δ(eval AUC) = {d:+.3f}")


if __name__ == "__main__":
    main()
