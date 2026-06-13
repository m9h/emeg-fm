#!/usr/bin/env python
"""Run the FMScope identity-trap audit on a MOABB BCI cohort against REVE.

Builds a MOABB motor-imagery cohort (BNCI2014_001, left/right hand), extracts
frozen REVE block-6 features, and runs ``fmscope.verdict.audit_cell`` to surface
the variance decomposition (``label_frac`` vs ``subject_frac``), null-calibrated
excess, c̄ direction-consistency, and subject-axis erasure. The numeric row is
written to ``results/moabb_fmscope/<cell>.json`` and the headline numbers are
printed so the verdict is visible in the Slurm log.

A high ``subject_frac`` with a low ``label_frac`` / collapsing ``Δ_erase`` is the
identity-trap signature: REVE's separability on this BCI cohort would be carried
by *who* the subject is rather than *what* they imagined.

moabb is installed to ``$MOABB_LIBS`` (a --target dir) and appended to the end of
sys.path so the SIF's numpy/scipy/torch keep precedence over the target copies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _append_moabb_libs() -> None:
    libs = os.environ.get("MOABB_LIBS")
    if libs and os.path.isdir(libs) and libs not in sys.path:
        sys.path.append(libs)  # APPEND: SIF packages win for shared names.


def _pick_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=6,
                    help="REVE block to read features from (default 6).")
    ap.add_argument("--model", default="brain-bzh/reve-base")
    ap.add_argument("--cell-name", default="BNCI2014_001-LR")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir",
                    default=os.path.expanduser("~/dev/emeg-fm/results/moabb_fmscope"))
    args = ap.parse_args()

    _append_moabb_libs()

    # Imports after sys.path is set up.
    from emeg_fm.moabb_cohort import build_moabb_cohort
    from emeg_fm.fmscope_bridge import REVEExtractor
    from fmscope.verdict import audit_cell, AuditConfig

    os.makedirs(args.out_dir, exist_ok=True)

    print("[moabb] building cohort (MOABB get_data downloads on first run)...",
          flush=True)
    cohort = build_moabb_cohort()
    n_rec = sum(1 for _ in cohort.iter_recordings())
    print(f"[moabb] cohort: {n_rec} recordings, {cohort.n_channels} ch "
          f"@ {cohort.sfreq} Hz", flush=True)
    print(f"[moabb] ch_names: {cohort.ch_names}", flush=True)

    print(f"[moabb] loading REVE {args.model} (layer {args.layer})...", flush=True)
    extractor = REVEExtractor(ch_names=cohort.ch_names, layer=args.layer,
                              model_id=args.model)
    device = _pick_device()
    print(f"[moabb] device={device}, embed_dim={extractor.embed_dim}", flush=True)

    # layout "W,C": within-subject (state) contrast — both classes present per
    # subject — so c̄ direction-consistency runs (not skipped as a trait cell).
    cfg = AuditConfig(cell_name=args.cell_name, cell_layout="W,C",
                      batch_size=args.batch_size, device=device)
    row = audit_cell(cohort, extractor, config=cfg)

    out = os.path.join(args.out_dir, f"{args.cell_name}.json")
    with open(out, "w") as f:
        json.dump(row, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else float(o))
    print(f"[moabb] wrote {out}", flush=True)

    def g(k):
        v = row.get(k)
        return float(v) if v is not None else float("nan")

    print("[moabb] === identity-trap verdict ===", flush=True)
    print(f"  label_frac    = {g('label_frac'):.4f}", flush=True)
    print(f"  subject_frac  = {g('subject_frac'):.4f}", flush=True)
    print(f"  residual_frac = {g('residual_frac'):.4f}", flush=True)
    print(f"  excess_label_ratio   = {g('excess_label_ratio'):.4f}", flush=True)
    print(f"  excess_subject_ratio = {g('excess_subject_ratio'):.4f}", flush=True)
    print(f"  c_bar_value          = {g('c_bar_value'):.4f}", flush=True)
    print(f"  erasure_label_ba_delta = {g('erasure_label_ba_delta'):.4f} "
          f"(interpretable={row.get('erasure_interpretable')})", flush=True)


if __name__ == "__main__":
    main()
