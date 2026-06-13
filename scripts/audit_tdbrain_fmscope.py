#!/usr/bin/env python
"""Run the FMScope identity-trap audit on TDBRAIN as a TRAIT cell against REVE.

TDBRAIN carries one psychiatric diagnosis per subject, so unlike the MOABB
motor-imagery cells (within-subject left/right state, ``cell_layout="W,C"``) this
is the *trait* regime (``cell_layout="T,C"``): erasing the subject-identity
subspace (LEACE) can genuinely destroy a per-subject trait label, which is
exactly why identity-free balanced accuracy is *diagnostic* here rather than
saturating at the within-subject ceiling that made BA degenerate on MI.

Default contrast: **MDD vs ADHD** — the best-powered pair in TDBRAIN (released,
labelled set ≈378 vs ≈253; ~30% of clinical/outcome labels are blinded so the
paper's ≈426/271 totals are higher). Healthy controls are marginal (≈47), so
MDD-vs-HC is underpowered for LOSO. Pass ``--classes MDD HC`` to run the
clinical-vs-healthy axis instead.

The diagnosis labels live in the DUA-gated ``participants.tsv`` (Synapse
syn26468893, downloadable only via the brainclinics.com ORCID portal — Synapse
does not yet grant download) — NOT in the EEG zips. Point ``--participants`` at
it once staged; the script reports clearly if it is absent.

Run (REVE needs a GPU; reuse the t9 Docker NGC pattern, never /data NFS)::

    python scripts/audit_tdbrain_fmscope.py \
        --bids-root /mnt/t9/tdbrain/bids \
        --participants /mnt/t9/tdbrain/participants.tsv

moabb is not needed here; the EEG stack (mne) loads the BrainVision files.
"""

from __future__ import annotations

import argparse
import json
import os


def _pick_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bids-root", default="/mnt/t9/tdbrain/bids")
    ap.add_argument("--participants", default="/mnt/t9/tdbrain/participants.tsv",
                    help="DUA-gated label table (Synapse syn26468893).")
    ap.add_argument("--classes", nargs=2, default=["MDD", "ADHD"],
                    metavar=("POS", "NEG"),
                    help="Two trait classes to contrast (default: MDD ADHD).")
    ap.add_argument("--label-col", default="indication",
                    choices=["indication", "formal Dx"])
    ap.add_argument("--task", default="restEC", choices=["restEC", "restEO"])
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="Cap recordings/class to balance the contrast.")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--model", default="brain-bzh/reve-base")
    ap.add_argument("--cell-name", default=None,
                    help="Default: TDBRAIN-<POS>v<NEG>.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cv", default="stratified-kfold",
                    choices=["stratified-kfold", "loso"],
                    help="Erasure label-probe CV. LOSO is underpowered for the "
                         "tiny HC arm; prefer stratified-kfold for trait cells.")
    ap.add_argument("--out-dir",
                    default=os.path.expanduser("~/dev/emeg-fm/results/tdbrain_fmscope"))
    args = ap.parse_args()

    if not os.path.exists(args.participants):
        raise SystemExit(
            f"[tdbrain] participants table not found: {args.participants}\n"
            "  The MDD/ADHD labels are DUA-gated (Synapse syn26468893) and are "
            "NOT in the EEG zips. Stage it (download_tdbrain.py --which "
            "participants, once DOWNLOAD access is granted) then re-run."
        )

    from emeg_fm.tdbrain_cohort import build_tdbrain_cohort
    from emeg_fm.fmscope_bridge import REVEExtractor
    from fmscope.verdict import audit_cell, AuditConfig

    pos, neg = args.classes
    cell_name = args.cell_name or f"TDBRAIN-{pos}v{neg}"
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[tdbrain] building trait cohort {pos} vs {neg} "
          f"(col={args.label_col}, task={args.task})...", flush=True)
    cohort = build_tdbrain_cohort(
        bids_root=args.bids_root, participants_tsv=args.participants,
        classes=(pos, neg), label_col=args.label_col, task=args.task,
        max_per_class=args.max_per_class,
    )
    recs = list(cohort.iter_recordings())
    n_pos = sum(1 for _, lbl, _ in recs if lbl == 0)
    n_neg = sum(1 for _, lbl, _ in recs if lbl == 1)
    print(f"[tdbrain] cohort: {len(recs)} subjects "
          f"({pos}={n_pos}, {neg}={n_neg}), {cohort.n_channels} ch "
          f"@ {cohort.sfreq} Hz", flush=True)

    print(f"[tdbrain] loading REVE {args.model} (layer {args.layer})...", flush=True)
    extractor = REVEExtractor(ch_names=cohort.ch_names, layer=args.layer,
                              model_id=args.model)
    device = _pick_device()
    print(f"[tdbrain] device={device}, embed_dim={extractor.embed_dim}", flush=True)

    # layout "T,C": trait (one label/subject) — c̄ direction-consistency is
    # skipped (it is a within-subject metric); identity-free BA Δ is the
    # diagnostic signal in this regime.
    cfg = AuditConfig(cell_name=cell_name, cell_layout="T,C",
                      batch_size=args.batch_size, device=device,
                      erasure_cv=args.cv)
    row = audit_cell(cohort, extractor, config=cfg)

    out = os.path.join(args.out_dir, f"{cell_name}.json")
    with open(out, "w") as f:
        json.dump(row, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else float(o))
    print(f"[tdbrain] wrote {out}", flush=True)

    def g(k):
        v = row.get(k)
        return float(v) if v is not None else float("nan")

    print("[tdbrain] === identity-trap verdict (trait cell) ===", flush=True)
    print(f"  label_frac    = {g('label_frac'):.4f}", flush=True)
    print(f"  subject_frac  = {g('subject_frac'):.4f}", flush=True)
    print(f"  residual_frac = {g('residual_frac'):.4f}", flush=True)
    print(f"  raw_label_ba          = {g('erasure_label_ba_raw'):.4f}", flush=True)
    print(f"  identity_free_label_ba= {g('erasure_label_ba_erased'):.4f}", flush=True)
    print(f"  erasure_label_ba_delta= {g('erasure_label_ba_delta'):.4f} "
          f"(interpretable={row.get('erasure_interpretable')})", flush=True)


if __name__ == "__main__":
    main()
