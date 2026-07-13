"""NeuroTechX-Atlas epilepsy section, dataset 5: Siena Scalp EEG (PhysioNet
1.0.0, Detti et al.). A 5th independent epilepsy population -- Italian adult
clinical scalp EEG (14 patients), standard 10-20 monopolar montage (unlike
CHB-MIT's bipolar derivations), so REVE gets the genuine channel positions
(no anchor-electrode approximation needed).

Per-patient ``Seizures-list-PNxx.txt`` gives seizure times as WALL-CLOCK
strings (``HH.MM.SS``), not seconds-from-file-start like CHB-MIT/TUSZ --
``_parse_seizure_list`` converts each seizure's start/end to an offset in
seconds from that block's own "Registration start time" (mod 86400h, so a
recording that crosses midnight is handled correctly). A single file can
appear across multiple seizure blocks (e.g. PN10-4.5.6.edf has 3 separate
seizure blocks with the same registration start) -- offsets are accumulated
per filename, not overwritten. Some filenames are themselves multi-numbered
(e.g. "PN10-4.5.6.edf") -- this is the real on-disk filename, not a parsing
artifact.

Real 10-20 monopolar channel names (same STD19 convention as TUSZ/Helsinki),
with the same "EEG "-prefix + inconsistent Fp/FP casing quirk as Helsinki.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epilepsy_scorer as es  # noqa: E402

ROOT = os.environ.get("SIENA_ROOT", "/data/datasets/eeg_fmri/siena_scalp_eeg/1.0.0")
WIN_S = 10.0
STD19 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
         "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]


def _clock_to_s(clock):
    h, m, s = (int(x) for x in clock.split(".")[:3])
    return h * 3600 + m * 60 + s


def _parse_seizure_list(text):
    """Seizures-list-PNxx.txt -> {filename: [(start_s, end_s), ...]}, offsets
    in seconds from that block's own registration start (mod 24h)."""
    out = defaultdict(list)
    blocks = re.split(r"(?=File name:)", text)
    for block in blocks:
        m = re.search(r"File name:\s*(\S+)", block)
        if not m:
            continue
        fname = m.group(1)
        reg = re.search(r"Registration start time:\s*(\d{1,2}\.\d{1,2}\.\d{1,2})", block)
        s0 = re.search(r"Seizure start time:\s*(\d{1,2}\.\d{1,2}\.\d{1,2})", block)
        s1 = re.search(r"Seizure end time:\s*(\d{1,2}\.\d{1,2}\.\d{1,2})", block)
        if not (reg and s0 and s1):
            continue
        reg_s = _clock_to_s(reg.group(1))
        start = (_clock_to_s(s0.group(1)) - reg_s) % 86400
        end = (_clock_to_s(s1.group(1)) - reg_s) % 86400
        out[fname].append((float(start), float(end)))
    return dict(out)


def iter_recordings(limit=None):
    """(edf_path, seizure_events, patient_id) for every Siena recording."""
    out = []
    for lst in sorted(glob.glob(os.path.join(ROOT, "PN*", "Seizures-list-*.txt"))):
        subj = os.path.basename(os.path.dirname(lst))
        with open(lst, errors="ignore") as f:
            events_by_file = _parse_seizure_list(f.read())
        seizure_edfs = set(events_by_file)
        edf_dir = os.path.dirname(lst)
        for edf in sorted(glob.glob(os.path.join(edf_dir, f"{subj}-*.edf"))):
            fname = os.path.basename(edf)
            out.append((edf, events_by_file.get(fname, []), subj))
    return out[:limit] if limit else out


def load_windows(edf, win_s=WIN_S, sfreq_out=100.0):
    import mne
    mne.set_log_level("error")
    try:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
    except Exception:
        return None, None
    ren = {c: re.sub(r"^EEG\s+", "", c, flags=re.IGNORECASE).strip().upper()
           for c in raw.ch_names}
    raw.rename_channels(ren)
    present = [c for c in STD19 if c in raw.ch_names]
    if len(present) < len(STD19):
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


def make_embed_fn(model, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        return lambda X: embed_reve(X, STD19, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (siena supports logbandpower, reve)")


def subject_splits(limit=None, seed_frac=(0.6, 0.2, 0.2)):
    subs = sorted({s for _, _, s in iter_recordings(limit)})
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def build_split(subjects, sfreq_out, limit=None, embed_fn=None):
    recs = []
    for edf, events, subj in iter_recordings(limit):
        if subj not in subjects:
            continue
        X, starts = load_windows(edf, WIN_S, sfreq_out)
        if X is None:
            continue
        y = label_windows(starts, WIN_S, events)
        dur_s = starts[-1] + WIN_S if len(starts) else 0.0
        rec = dict(y=y, starts=starts, seiz=events, patient=subj, dur_h=dur_s / 3600.0)
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
    print(f"[siena] model={a.model} limit={a.limit} patients train/dev/eval="
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
        print(f"\n[siena] {a.model}{tag} SzCORE:")
        print(f"  eval AUC[.1-10 FP/day] = {out['szcore_auc']:.3f}  sens@1/d={out['szcore_sens1']*100:.1f}%"
              f"  sens@10/d={out['szcore_sens10']*100:.1f}%  F1={out['op_f1']:.3f} @ {out['op_faday']:.1f} FP/day")
    if False in results and True in results:
        d = results[False]["szcore_auc"] - results[True]["szcore_auc"]
        print(f"\n[siena] {a.model} identity-free Δ(eval AUC) = {d:+.3f}")


if __name__ == "__main__":
    main()
