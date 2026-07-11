"""T3A pilot on HMC (our cleanest existing identity-trap dataset: REVE wins
raw perf but collapses hardest under identity erasure -- docs/hmc_sleep_
results.md). Question: does T3A's per-subject online adaptation gain shrink
under identity-erasure? If the held-out subject's own test features are what
T3A leans on to adapt, LEACE-erasing subject identity from those same
features BEFORE T3A runs should shrink any T3A benefit that was actually
identity exploitation rather than genuine distribution-shift adaptation.

Only meaningful for REVE (LogisticRegression probe) -- classical's GBM has no
linear weight vectors for T3A to warm-start supports from (see tta_t3a.py
module docstring).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_sleep_hmc as hmc  # noqa: E402
import atlas_sleep_section as sl  # noqa: E402
import tta_t3a as t3a  # noqa: E402


def main():
    import warnings; warnings.filterwarnings("ignore")
    import torch
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = hmc.subject_splits(None)
    print(f"[tta-pilot-hmc] subjects train/dev/eval={len(tr_s)}/{len(dv_s)}/{len(ev_s)}",
          flush=True)
    embed_fn = hmc.make_embed_fn("reve", 100.0, dev)
    train = hmc.build_split(tr_s, None, embed_fn=embed_fn)
    evl = hmc.build_split(ev_s, None, embed_fn=embed_fn)
    print(f"  train: {len(train)} recs, {sum(len(r['y']) for r in train)} epochs", flush=True)
    print(f"  eval:  {len(evl)} recs, {sum(len(r['y']) for r in evl)} epochs", flush=True)

    Xtr = np.concatenate([r["emb"] for r in train])
    ytr = np.concatenate([r["y"] for r in train])

    for erase in (False, True):
        pid = None
        if erase:
            pmap = {p: i for i, p in enumerate(sorted({r["patient"] for r in train}))}
            pid = np.concatenate([[pmap[r["patient"]]] * len(r["y"]) for r in train])
        clf, sc, er = sl._fit_probe(Xtr, ytr, "reve", erase_patient=pid)

        base_y, base_yhat, t3a_yhat = [], [], []
        for rec in evl:
            Xs = sc.transform(rec["emb"])
            if er is not None:
                Xs = er.transform(Xs)
            base_yhat.append(clf.predict(Xs))
            t3a_yhat.append(t3a.t3a_adapt_sequence(clf, Xs, filter_k=20))
            base_y.append(rec["y"])
        y = np.concatenate(base_y)
        base_yhat = np.concatenate(base_yhat)
        t3a_yhat = np.concatenate(t3a_yhat)

        tag = "identity-free" if erase else "normal"
        base_bacc = balanced_accuracy_score(y, base_yhat)
        base_kappa = cohen_kappa_score(y, base_yhat)
        t3a_bacc = balanced_accuracy_score(y, t3a_yhat)
        t3a_kappa = cohen_kappa_score(y, t3a_yhat)
        print(f"\n[tta-pilot-hmc] reve {tag}:")
        print(f"  baseline   bacc={base_bacc*100:.1f}% kappa={base_kappa:.3f}")
        print(f"  T3A-adapt  bacc={t3a_bacc*100:.1f}% kappa={t3a_kappa:.3f}")
        print(f"  T3A gain (kappa) = {t3a_kappa - base_kappa:+.3f}")


if __name__ == "__main__":
    main()
