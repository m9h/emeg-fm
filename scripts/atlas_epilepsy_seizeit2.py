"""NeuroTechX-Atlas epilepsy section, dataset 2: SeizeIT2 (OpenNeuro ds005873).

A second, independent epilepsy dataset alongside TUSZ — SeizeIT2 is one of
NeuroAtlas's actual named epilepsy datasets (wearable, behind-the-ear EEG, 5
European Epileptic Monitoring Centers, SCORE HED seizure vocabulary). Having
BOTH TUSZ (clinical scalp, Temple) and SeizeIT2 (wearable, behind-the-ear,
multi-center Europe) tests whether the identity trap (see TUSZ results:
docs/epilepsy_tusz_results.md) generalizes across recording modality/population,
not just one corpus.

Data: /mnt/t9/seizeit2/ (BIDS, OpenNeuro ds005873). Only 2 EEG channels
(behind-the-ear) -> same montage constraint as the sleep section: classical
band-power + REVE (montage-flexible name lookup); Interpolated* (BIOT/LaBraM/
EEGDINO) can't fit (<4 head-dig points). Events: `*_events.tsv` with
`eventType` in the SCORE HED vocabulary; any `sz_*` prefix = seizure, `bckg`/
`impd` = non-seizure/artifact. Same scoring as TUSZ (SzCORE + our identity-free
LEACE axis; same epilepsy_scorer.py primitives), subject-disjoint 70/15/15
split (no official train/dev/eval like TUSZ).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epilepsy_scorer as es  # noqa: E402

ROOT = os.environ.get("SEIZEIT2_ROOT", "/mnt/t9/seizeit2")
WIN_S = 10.0
SFREQ_NATIVE = 256.0


def iter_recordings(limit=None):
    """(eeg_edf_path, events_tsv_path, subject_id) for every SeizeIT2 EEG run."""
    out = []
    for ev in sorted(glob.glob(os.path.join(ROOT, "sub-*", "ses-*", "eeg", "*_events.tsv"))):
        edf = ev.replace("_events.tsv", "_eeg.edf")
        if not os.path.exists(edf):
            continue
        subj = os.path.basename(ev).split("_")[0]  # sub-XXX
        out.append((edf, ev, subj))
    return out[:limit] if limit else out


def load_seiz_events(events_tsv):
    """events.tsv -> [(start_s, stop_s)] for rows whose eventType starts with 'sz_'."""
    df = pd.read_csv(events_tsv, sep="\t")
    sz = df[df["eventType"].astype(str).str.startswith("sz_")]
    return [(float(r.onset), float(r.onset) + float(r.duration)) for r in sz.itertuples()]


def load_windows(edf, win_s=WIN_S, sfreq_out=100.0):
    """Continuous behind-the-ear EEG -> non-overlapping windows on both channels."""
    import mne
    mne.set_log_level("error")
    try:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
    except Exception:
        return None, None
    if raw.info["sfreq"] != sfreq_out:
        raw.resample(sfreq_out, verbose="error")
    x = raw.get_data()
    w = int(win_s * sfreq_out)
    n = x.shape[1] // w
    if n == 0:
        return None, None, None
    X = np.stack([x[:, i * w:(i + 1) * w] for i in range(n)]).astype(np.float32)
    starts = np.arange(n) * win_s
    return X, starts, list(raw.ch_names)


def label_windows(starts, win_s, seiz_events):
    y = np.zeros(len(starts), int)
    for i, s in enumerate(starts):
        e = s + win_s
        if any(s < b and a < e for a, b in seiz_events):
            y[i] = 1
    return y


def make_embed_fn(model, sfreq, dev, ch_names):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        # SeizeIT2's electrodes ('BTEleft SD', 'CROSStop SD', ...) are proprietary
        # Byteflies SensorDot wearable labels -- behind-the-ear / cross-head
        # references with NO honest single-electrode correspondence to REVE's
        # standard_1005 name vocabulary (unlike Sleep-EDF's bipolar derivations,
        # which at least have genuine 10-20 endpoints). Forcing an approximate
        # mapping here would be undocumented guesswork feeding a real result, so
        # classical-only for this dataset rather than a shaky FM number.
        raise SystemExit(
            "SeizeIT2 REVE unsupported: electrode names "
            f"{ch_names!r} are proprietary wearable labels with no defensible "
            "anatomical mapping to REVE's standard_1005 vocabulary. "
            "Use --model logbandpower.")
    raise SystemExit(f"unknown model {model!r} (SeizeIT2 supports logbandpower only -- see reve note)")


def subject_splits(limit=None, seed_frac=(0.7, 0.15, 0.15)):
    """Subject-disjoint 70/15/15. Floors dev/eval at >=1 subject (else a small
    --limit smoke test rounds dev to 0 -> degenerate, untuned operating point)."""
    subs = sorted({s for _, _, s in iter_recordings(limit)})
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def build_split(subjects, sfreq_out, limit=None, embed_fn_factory=None):
    """Streaming build+embed (like atlas_epilepsy_section) -- embed_fn built lazily
    from the FIRST recording's channel names (constant across SeizeIT2)."""
    recs, embed_fn = [], None
    for edf, ev, subj in iter_recordings(limit):
        if subj not in subjects:
            continue
        X, starts, chs = load_windows(edf, WIN_S, sfreq_out)
        if X is None:
            continue
        if embed_fn is None and embed_fn_factory is not None:
            embed_fn = embed_fn_factory(chs)
        seiz = load_seiz_events(ev)
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
    print(f"[seizeit2] model={a.model} limit={a.limit} subjects train/dev/eval="
          f"{len(tr_s)}/{len(dv_s)}/{len(ev_s)}", flush=True)
    embed_factory = lambda chs: make_embed_fn(a.model, a.sfreq, dev_, chs)
    splits = {}
    for name, subs in (("train", tr_s), ("dev", dv_s), ("eval", ev_s)):
        recs = build_split(subs, a.sfreq, a.limit, embed_fn_factory=embed_factory)
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
        print(f"\n[seizeit2] {a.model}{tag} SzCORE:")
        print(f"  eval AUC[.1-10 FP/day] = {out['szcore_auc']:.3f}  sens@1/d={out['szcore_sens1']*100:.1f}%"
              f"  sens@10/d={out['szcore_sens10']*100:.1f}%  F1={out['op_f1']:.3f} @ {out['op_faday']:.1f} FP/day")
    if False in results and True in results:
        d = results[False]["szcore_auc"] - results[True]["szcore_auc"]
        print(f"\n[seizeit2] {a.model} identity-free Δ(eval AUC) = {d:+.3f}")


if __name__ == "__main__":
    main()
