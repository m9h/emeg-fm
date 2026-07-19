"""NeuroTechX-Atlas depression-detection section: MDD (Mumtaz 2016, HF
braindecode/mdd_mumtaz2016). New TASK DOMAIN (clinical psychiatric
classification, not sleep/epilepsy/BCI/affect/workload/speech) and one of
OpenEEGBench's 12.

64 subjects (healthy "HS*" + MDD "MDDS*"), binary MDD-vs-healthy
classification, 19-channel standard monopolar 10-20 montage (same channel
SET as the epilepsy/sleep STD19 convention, different order), 200 Hz, 5s
windows (1000 samples). Windows are drawn from 3 mixed task conditions
(P300, eyesClosed, eyesOpen) under one binary label -- reproduces
OpenEEGBench's own task framing (they don't split by sub-task either)
rather than inventing a narrower resting-state-only protocol.

Pulled via braindecode's own `BaseConcatDataset.pull_from_hub` -- shares
atlas_hf_braindecode.py's grouping helper with Arithmetic and BCI2020-3.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_hf_braindecode as hb  # noqa: E402
import atlas_sleep_section as sl  # noqa: E402  (reuses _fit_probe/evaluate)
import epilepsy_scorer as es  # noqa: E402

HF_ID = "braindecode/mdd_mumtaz2016"
EEG_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
                "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
SFREQ = 200.0


def _reve_channel_names():
    return list(EEG_CHANNELS)


def make_embed_fn(model, sfreq, dev):
    if model == "logbandpower":
        return lambda X: np.stack([es.window_features(w, sfreq) for w in X])
    if model == "reve":
        from atlas_bci_section import embed_reve
        ch = _reve_channel_names()
        return lambda X: embed_reve(X, ch, sfreq, dev)
    raise SystemExit(f"unknown model {model!r} (mdd supports logbandpower, reve)")


def build_split(groups, subjects, embed_fn):
    recs = []
    for subj in subjects:
        if subj not in groups:
            continue
        X, y = groups[subj]
        rec = dict(y=y, patient=subj)
        rec["emb"] = embed_fn(X) if embed_fn is not None else None
        recs.append(rec)
    return recs


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="logbandpower")
    ap.add_argument("--sfreq", type=float, default=SFREQ)
    ap.add_argument("--identity-free", action="store_true")
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    ds = hb.pull(HF_ID)
    groups = hb.concat_dataset_to_subject_groups(ds)
    # subject id encodes label ("HS*"=healthy, "MDDS*"=MDD) -- a naive
    # lexicographic sort-and-slice puts the whole "MDDS*" class into eval
    # (verified: 100% single-class eval, degenerate kappa=0.000). Stratify
    # by each subject's (constant) label instead.
    subject_to_label = {s: int(y[0]) for s, (_, y) in groups.items()}
    tr_s, dv_s, ev_s = hb.stratified_subject_splits(subject_to_label)
    print(f"[mdd] model={a.model} subjects train/dev/eval="
          f"{len(tr_s)}/{len(dv_s)}/{len(ev_s)}", flush=True)
    embed_fn = make_embed_fn(a.model, a.sfreq, dev_)
    splits = {}
    for name, subs in (("train", tr_s), ("dev", dv_s), ("eval", ev_s)):
        recs = build_split(groups, subs, embed_fn)
        tot = sum(len(r["y"]) for r in recs)
        splits[name] = recs
        print(f"  {name}: {len(recs)} subjects, {tot} windows, "
              f"d={recs[0]['emb'].shape[1] if recs else 0}", flush=True)

    configs = [False, True] if (a.both or a.identity_free) else [False]
    results = {}
    for erase in configs:
        out = sl.evaluate(splits["train"], splits["dev"], splits["eval"], a.model, erase_patient=erase)
        results[erase] = out
        tag = " (identity-free)" if erase else ""
        print(f"\n[mdd] {a.model}{tag} MDD-vs-healthy:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        print(f"  (dev balanced-acc = {out['dev_bacc']*100:.1f}%  kappa = {out['dev_kappa']:.3f})")
    if False in results and True in results:
        d = results[False]["eval_kappa"] - results[True]["eval_kappa"]
        print(f"\n[mdd] {a.model} identity-free Δ(eval kappa) = {d:+.3f} "
              f"(normal {results[False]['eval_kappa']:.3f} - idfree {results[True]['eval_kappa']:.3f})")


if __name__ == "__main__":
    main()
