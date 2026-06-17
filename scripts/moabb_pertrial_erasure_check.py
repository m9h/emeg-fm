#!/usr/bin/env python
"""Per-trial vs recording-pooled subject-axis erasure on a MOABB cohort.

The FMScope MOABB leaderboard measures the identity trap with
``audit_cell`` → ``subject_axis_erasure`` on window-level features, but the
cohort builder emits **one recording per ``(subject, class)``** group
(``moabb_cohort.build_moabb_cohort``). So the erasure's recording-level CV
pools *all* of a subject's class-trials into a single pooled prediction — e.g.
BNCI2014_001 collapses to 9 subjects × 2 classes = 18 points. Averaging ~hundreds
of trials into each point crushes the within-class variance, so the pooled
decode saturates and the subject-offset that LEACE removes becomes the only
thing keeping the *raw* score down — manufacturing a large "identity-free lift"
that need not reflect any per-trial generalization gap (this is exactly the
artifact that vanished on ERP CORE once measured per trial).

This script extracts the cohort's per-trial REVE features **once** and runs
``subject_axis_erasure`` two ways on the *same* features:

  * ``pooled``    — default grouping (``_segment_recordings`` → one recording
                    per contiguous ``(subject, class)`` run): the leaderboard's
                    current metric.
  * ``per_trial`` — one recording per trial, grouped by subject
                    (StratifiedGroupKFold never shares a subject): single-trial,
                    ``n ≫ p``, the meaningful metric.

If the ``per_trial`` lift collapses toward zero while ``pooled`` stays large,
the leaderboard trap for that dataset was a pooling artifact. If both lift, the
trap is real on a defensible metric.

Runtime: Docker NGC PyTorch 26.05 + uv on /mnt/t9 (REVE needs torch+GPU and the
gated ``brain-bzh/reve-base`` checkpoint). See the accompanying ``.sbatch``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


def _append_moabb_libs() -> None:
    """Mirror the leaderboard: ensure the moabb fork + libs are importable."""
    for p in os.environ.get("PYTHONPATH", "").split(":"):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def _pick_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _erasure_both_ways(feats, sids, labels, cv):
    """Return (pooled, per_trial) ErasureResult on the same features."""
    from fmscope.diagnostics.erasure import subject_axis_erasure
    pooled = subject_axis_erasure(feats, sids, labels, cv=cv)
    n = len(labels)
    per_trial = subject_axis_erasure(
        feats, sids, labels,
        window_recording=np.arange(n), rec_labels=labels, rec_pids=sids,
        cv=cv,
    )
    return pooled, per_trial


def _row(tag, er):
    return {
        "grouping": tag,
        "raw": round(float(er.label_ba_raw), 4),
        "erased": round(float(er.label_ba_erased), 4),
        "lift": round(float(er.label_ba_delta), 4),
        "lift_std": round(float(er.label_ba_delta_std), 4),
        "subj_ba_pre": round(float(er.subj_ba_linear_pre), 4),
        "subj_ba_post": round(float(er.subj_ba_linear_post), 4),
        "subj_chance": round(float(er.chance), 4),
        "rank": int(er.rank_subject_axis),
        "degenerate": bool(er.degenerate),
        "interpretable": bool(er.interpretable),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--model", default="brain-bzh/reve-base")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cv", default="stratified-kfold",
                    choices=["stratified-kfold", "loso"])
    ap.add_argument("--out-dir",
                    default=os.path.expanduser("~/dev/emeg-fm/results/moabb_fmscope"))
    args = ap.parse_args()

    _append_moabb_libs()
    from emeg_fm.moabb_cohort import build_moabb_cohort
    from emeg_fm.fmscope_bridge import REVEExtractor
    from fmscope.verdict.audit import _extract_features

    os.makedirs(args.out_dir, exist_ok=True)
    device = _pick_device()

    # Default cohort is the BNCI2014_001 LeftRightImagery PoC (9 subj, 22 ch).
    cohort = build_moabb_cohort()
    ext = REVEExtractor(ch_names=cohort.ch_names, layer=args.layer,
                        model_id=args.model)
    feats, sids, labels, stats = _extract_features(
        ext, cohort, batch_size=args.batch_size, device=device)
    n_subj = int(np.unique(sids).size)
    print(f"[cohort] BNCI2014_001 LeftRightImagery: trials={len(labels)} "
          f"subj={n_subj} dim={feats.shape[1]} device={device}", flush=True)

    pooled, per_trial = _erasure_both_ways(feats, sids, labels, args.cv)
    rows = [_row("pooled_subject_class", pooled), _row("per_trial", per_trial)]

    for r in rows:
        print(f"[{r['grouping']:>20}] raw={r['raw']:.3f} erased={r['erased']:.3f} "
              f"lift={r['lift']:+.3f}±{r['lift_std']:.3f} "
              f"subj_ba {r['subj_ba_pre']:.2f}->{r['subj_ba_post']:.2f} "
              f"(chance {r['subj_chance']:.3f}) "
              f"interpretable={r['interpretable']} degenerate={r['degenerate']}",
              flush=True)

    verdict = ("ARTIFACT — pooled lift does not survive per-trial"
               if pooled.label_ba_delta - per_trial.label_ba_delta > 0.05
               and per_trial.label_ba_delta < 0.03
               else "TRAP SURVIVES per-trial"
               if per_trial.label_ba_delta >= 0.03
               else "inconclusive")
    print(f"[verdict] BNCI2014_001 ({args.cv}): {verdict}  "
          f"(pooled lift {pooled.label_ba_delta:+.3f} vs "
          f"per-trial lift {per_trial.label_ba_delta:+.3f})", flush=True)

    base = os.path.join(args.out_dir, f"pertrial_erasure_check_{args.cv}")
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(base + ".md", "w") as f:
        f.write(
            "# BNCI2014_001 — per-trial vs pooled subject-axis erasure\n\n"
            f"REVE block-{args.layer} ({args.model}), {n_subj} subjects, "
            f"{len(labels)} trials, CV={args.cv}. Same features both ways; only "
            "the recording grouping (and thus prediction pooling) differs.\n\n"
            "| grouping | raw | erased | lift | subj BA pre→post (chance) | "
            "interp | degenerate |\n"
            "|---|---:|---:|---:|:--|:--|:--|\n")
        for r in rows:
            f.write(f"| {r['grouping']} | {r['raw']:.3f} | {r['erased']:.3f} | "
                    f"{r['lift']:+.3f}±{r['lift_std']:.3f} | "
                    f"{r['subj_ba_pre']:.2f}→{r['subj_ba_post']:.2f} "
                    f"({r['subj_chance']:.3f}) | {r['interpretable']} | "
                    f"{r['degenerate']} |\n")
        f.write(f"\n**Verdict:** {verdict}\n")
    print(f"[done] wrote {base}.{{csv,md}}", flush=True)


if __name__ == "__main__":
    main()
