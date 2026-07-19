"""NeuroTechX-Atlas mental-workload section: Arithmetic (Zyma 2019, HF
braindecode/arithmetic_zyma2019). New TASK DOMAIN (mental workload / task
engagement, not sleep/epilepsy/BCI/affect) and one of OpenEEGBench's 12.

36 subjects, binary rest-vs-mental-arithmetic classification, 19-channel
standard monopolar 10-20 montage, 200 Hz, 5s windows (1000 samples). Pulled
via braindecode's own `BaseConcatDataset.pull_from_hub` (see
atlas_hf_braindecode.py for the shared grouping helper -- Arithmetic,
BCI Comp 2020-3, and MDD Mumtaz2016 all use this same schema).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_hf_braindecode as hb  # noqa: E402
import atlas_sleep_section as sl  # noqa: E402  (reuses _fit_probe/evaluate)
import epilepsy_scorer as es  # noqa: E402

HF_ID = "braindecode/arithmetic_zyma2019"
EEG_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4",
                "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"]
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
    raise SystemExit(f"unknown model {model!r} (arithmetic supports logbandpower, reve)")


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
    tr_s, dv_s, ev_s = hb.subject_splits(groups.keys())
    print(f"[arithmetic] model={a.model} subjects train/dev/eval="
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
        print(f"\n[arithmetic] {a.model}{tag} rest-vs-arithmetic:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        print(f"  (dev balanced-acc = {out['dev_bacc']*100:.1f}%  kappa = {out['dev_kappa']:.3f})")
    if False in results and True in results:
        d = results[False]["eval_kappa"] - results[True]["eval_kappa"]
        print(f"\n[arithmetic] {a.model} identity-free Δ(eval kappa) = {d:+.3f} "
              f"(normal {results[False]['eval_kappa']:.3f} - idfree {results[True]['eval_kappa']:.3f})")


if __name__ == "__main__":
    main()
