"""NeuroTechX-Atlas CogNeuro section: frozen EEG-FM / TS-FM on the ERP CORE battery
(Kappenman & Luck 2021) — the SAME ~40 subjects across 7 canonical cognitive ERP
components. A cognitive-process domain NeuroAtlas barely touches (it used only the
N170 dataset, and as a BCI task).

Why ERP CORE is the sharpest identity-trap testbed: subject identity is a CONSTANT
confound across all components, so per component we report three numbers:
  - LOSO normalized-BA         : cross-subject decode (identity-CONFOUNDED)
  - within-subject normalized-BA: per-subject CV, averaged (identity-FREE by design)
  - identity-free (LEACE) LOSO : erase subject axis, re-probe cross-subject
The within-vs-LOSO gap and the LEACE Δ both isolate how much of an FM's "cognitive"
decode is really subject re-identification. On the clean adult ERP CORE the gap
should be small (existence proof that FMs capture cognition); on HBN-task (the
developmental stress-test, added later) it should be large.

Reuses the BCI-section extractors (one braindecode-zoo / TS-FM / REVE path) + the
ERP-CORE loader. Runs in the NGC 26.06 container (scripts/eegfm_t9.sh).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_bci_section import (CLASSICAL, EEGFM, TSFM, _norm_ba, embed_classical,  # noqa: E402
                               embed_eegfm, embed_reve, embed_tsfm, loso)
from erpcore_luck_parity import COMPONENTS, _load_component_epochs  # noqa: E402


def extract(model, X, ch_names, sfreq, dev):
    if model in EEGFM:
        return embed_eegfm(model, X, ch_names, sfreq, dev)
    if model in TSFM:
        return embed_tsfm(model, X, sfreq)
    if model == "reve":
        return embed_reve(X, ch_names, sfreq, dev)
    if model in CLASSICAL:
        return embed_classical(X, sfreq)
    raise SystemExit(f"unknown model {model!r}")


def within_subject_ba(emb, y, subj, n_classes, seed=42):
    """Per-subject 5-fold CV (train+test within the SAME subject → identity-free by
    construction), balanced-accuracy averaged across subjects."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    accs = []
    for s in np.unique(subj):
        m = subj == s
        Xs, ys = emb[m], y[m]
        if len(np.unique(ys)) < 2 or np.min(np.bincount(ys)) < 5:
            continue
        pred = np.zeros(len(ys), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(Xs[tr]), ys[tr])
            pred[te] = clf.predict(sc.transform(Xs[te]))
        accs.append(balanced_accuracy_score(ys, pred))
    return _norm_ba(float(np.mean(accs)), n_classes) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="eeg-fm (biot/bendr/labram), ts-fm (moment/mantis/chronos-bolt), reve, or logbandpower")
    ap.add_argument("--components", nargs="+", default=list(COMPONENTS),
                    help="subset of ERP CORE components (default: all 7)")
    ap.add_argument("--sfreq", type=float, default=200.0)
    ap.add_argument("--fmax", type=float, default=45.0)
    ap.add_argument("--max-subjects", type=int, default=None)
    a = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import mne
    import torch
    mne.set_log_level("error")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[cogneuro] model={a.model}  components={a.components}", flush=True)
    print(f"{'component':8s} {'n':>5s} {'LOSO':>7s} {'within':>7s} {'idfree':>7s} {'id-Δ':>6s}")
    rows = []
    for comp in a.components:
        ds_cls, interval, _win = COMPONENTS[comp]
        raw, _times, y, subj, ch_names = _load_component_epochs(
            ds_cls, interval, a.fmax, a.sfreq, a.max_subjects)
        y = np.asarray(y)
        classes = sorted(set(y))
        yi = np.array([classes.index(v) for v in y])
        nc = len(classes)
        emb = extract(a.model, raw.astype(np.float32), ch_names, a.sfreq, dev)
        losoc = loso(emb, yi, subj, nc, erase_identity=False)
        idf = loso(emb, yi, subj, nc, erase_identity=True)
        wsub = within_subject_ba(emb, yi, subj, nc)
        rows.append((comp, len(yi), losoc, wsub, idf))
        print(f"{comp:8s} {len(yi):5d} {losoc*100:6.1f}% {wsub*100:6.1f}% {idf*100:6.1f}% "
              f"{(losoc-idf)*100:+5.1f}", flush=True)

    fin = [r for r in rows if np.isfinite(r[2])]
    if fin:
        print(f"\n[cogneuro] {a.model} mean over {len(fin)} components: "
              f"LOSO {np.mean([r[2] for r in fin])*100:.1f}%  "
              f"within {np.nanmean([r[3] for r in fin])*100:.1f}%  "
              f"idfree {np.mean([r[4] for r in fin])*100:.1f}%")


if __name__ == "__main__":
    main()
