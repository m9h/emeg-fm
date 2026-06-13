#!/usr/bin/env python
"""Loop the FMScope identity-trap audit over the MOABB dataset set.

For every MOABB dataset compatible with the chosen paradigm (default:
``LeftRightImagery``), this builds the cohort, extracts frozen-REVE features, and
runs ``fmscope.verdict.audit_cell``. Each dataset becomes one row in a growing
leaderboard CSV whose headline addition is the **identity-free** score: the task
balanced-accuracy *after* the subject-identifying subspace is erased (LEACE),
reported next to the raw score. The gap is the subject-identity inflation that
MOABB's leave-one-subject-out (CrossSubject) evaluation cannot otherwise expose.

The loop is **resumable**: dataset codes already in the CSV are skipped, so
rerunning simply adds more datasets ("adding more and more"). Per-dataset errors
(download/format/paradigm-incompat) are caught and recorded, never aborting the
sweep. After each successful dataset the CSV and the rendered markdown
leaderboard are rewritten, so progress survives an interrupted run.

Run on the user's infra (Docker NGC 26.05 + uv + /mnt/t9) via
``scripts/moabb_identity_leaderboard.sbatch`` / the docker launcher — never /data
NFS or the NFS apptainer SIF.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback


# Persisted leaderboard schema (CSV header order).
FIELDS = [
    "dataset", "paradigm", "n_subjects", "n_recordings", "n_windows",
    "label_frac", "subject_frac", "residual_frac",
    "excess_label_ratio", "excess_subject_ratio", "c_bar_value",
    "raw_label_ba", "identity_free_label_ba", "erasure_delta",
    "erasure_interpretable", "verdict", "layer", "model", "status",
]


def _append_moabb_libs() -> None:
    libs = os.environ.get("MOABB_LIBS")
    if libs and os.path.isdir(libs) and libs not in sys.path:
        sys.path.append(libs)  # APPEND: container packages win for shared names.


def _pick_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _g(row, k):
    v = row.get(k)
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _verdict(label_frac, subject_frac, raw_ba, free_ba) -> str:
    """Coarse identity-trap flag for the leaderboard.

    TRAP: subject identity dominates variance and the raw separability does not
    survive as genuine task skill — i.e. identity carried the score. We call it
    when subject_frac exceeds label_frac and erasing identity does not lower the
    task score (free_ba >= raw_ba, the leakage was unhelpful-or-harmful, so the
    raw number was not real task skill).
    """
    import math

    if any(math.isnan(x) for x in (label_frac, subject_frac, raw_ba, free_ba)):
        return "n/a"
    if subject_frac > label_frac and free_ba >= raw_ba - 1e-9:
        return "TRAP"
    if subject_frac > label_frac:
        return "identity-reliant"
    return "task-carried"


def _load_done(csv_path) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            # Only successfully-audited datasets count as done; failed rows are
            # retried on the next run (transient timeouts, fixed adapter bugs).
            if r.get("dataset") and r.get("status") == "ok":
                done.add(r["dataset"])
    return done


def _write_csv(csv_path, rows) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_rows(csv_path) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _render_md(csv_path, md_path, paradigm_name) -> None:
    rows = [r for r in _read_rows(csv_path) if r.get("status") == "ok"]

    def fnum(r, k, p=3):
        try:
            return f"{float(r[k]):.{p}f}"
        except (KeyError, TypeError, ValueError):
            return "—"

    # Sort by erasure lift (identity-free - raw), largest identity-masking first.
    def lift(r):
        try:
            return float(r["identity_free_label_ba"]) - float(r["raw_label_ba"])
        except (KeyError, TypeError, ValueError):
            return float("-inf")

    rows.sort(key=lift, reverse=True)

    lines = [
        f"# MOABB identity-free leaderboard — {paradigm_name}",
        "",
        "Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647). "
        "**Identity-free BA** = task balanced-accuracy after the subject subspace "
        "is erased (LEACE); **Δ = Identity-free − Raw** is the task signal "
        "*recovered* by erasure. A large positive Δ means subject-identity variance "
        "was masking the left/right axis in MOABB's cross-subject evaluation — the "
        "identity trap. `subj_frac` = fraction of representation variance explained "
        "by subject identity; `c̄` = cross-subject direction-consistency of the task "
        "axis (≈0 ⇒ the left/right axis does not generalize across people).",
        "",
        "| Dataset | N subj | Raw BA | Identity-free BA | Δ (lift) | subj_frac | label_frac | c̄ | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r.get('n_subjects','—')} | "
            f"{fnum(r,'raw_label_ba')} | {fnum(r,'identity_free_label_ba')} | "
            f"{fnum(r,'erasure_delta')} | {fnum(r,'subject_frac')} | "
            f"{fnum(r,'label_frac')} | {fnum(r,'c_bar_value')} | "
            f"{r.get('verdict','—')} |"
        )
    failed = [r for r in _read_rows(csv_path) if r.get("status") != "ok"]
    if failed:
        lines += ["", "### Skipped / failed", ""]
        for r in failed:
            lines.append(f"- `{r['dataset']}` — {r.get('status','?')}")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def _resolve_datasets(paradigm, codes):
    """Return [(code, dataset_instance), ...] for the paradigm, optionally
    filtered to an explicit ``codes`` list (matched against ``dataset.code``)."""
    out = []
    for d in paradigm.datasets:
        code = getattr(d, "code", d.__class__.__name__)
        if codes and code not in codes:
            continue
        if "fake" in code.lower():
            continue
        # License-gated datasets (e.g. Shin2017A) refuse to download unless
        # accept=True; running this sweep is the acceptance, so set it here.
        if hasattr(d, "accept"):
            d.accept = True
        out.append((code, d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--model", default="brain-bzh/reve-base")
    ap.add_argument("--paradigm", default="leftright", choices=["leftright"],
                    help="MOABB paradigm to sweep (only leftright for now).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Explicit dataset codes to run (default: all compatible).")
    ap.add_argument("--min-subjects", type=int, default=4)
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="Skip datasets with more subjects than this (0 = no cap).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most this many NEW datasets this run (0 = all).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cv", default="stratified-kfold",
                    choices=["stratified-kfold", "loso"],
                    help="Erasure label-probe CV. 'loso' = leave-one-subject-out "
                         "(strictest cross-subject split); writes a separate "
                         "leaderboard_<paradigm>_loso.csv so it never clobbers the "
                         "main StratifiedGroupKFold leaderboard.")
    ap.add_argument("--out-dir",
                    default=os.path.expanduser("~/dev/emeg-fm/results/moabb_fmscope"))
    args = ap.parse_args()

    _append_moabb_libs()

    from moabb.paradigms import LeftRightImagery
    from emeg_fm.moabb_cohort import build_moabb_cohort
    from emeg_fm.fmscope_bridge import REVEExtractor
    from fmscope.verdict import audit_cell, AuditConfig

    os.makedirs(args.out_dir, exist_ok=True)
    cv_suffix = "_loso" if args.cv == "loso" else ""
    csv_path = os.path.join(args.out_dir, f"leaderboard_{args.paradigm}{cv_suffix}.csv")
    md_path = os.path.join(args.out_dir, f"leaderboard_{args.paradigm}{cv_suffix}.md")

    paradigm = LeftRightImagery(fmin=0.5, fmax=99.5, resample=200.0)
    paradigm_name = "LeftRightImagery"
    device = _pick_device()

    all_ds = _resolve_datasets(paradigm, set(args.datasets) if args.datasets else None)
    done = _load_done(csv_path)
    rows = _read_rows(csv_path)
    todo = [(c, d) for (c, d) in all_ds if c not in done]
    print(f"[loop] paradigm={paradigm_name} compatible={len(all_ds)} "
          f"done={len(done)} todo={len(todo)} device={device}", flush=True)

    processed = 0
    for code, dataset in todo:
        if args.limit and processed >= args.limit:
            print(f"[loop] --limit {args.limit} reached; stopping.", flush=True)
            break
        try:
            n_subj = len(dataset.subject_list)
        except Exception:
            n_subj = -1
        if n_subj >= 0 and n_subj < args.min_subjects:
            print(f"[skip] {code}: {n_subj} subj < min {args.min_subjects}", flush=True)
            continue
        if args.max_subjects and n_subj > args.max_subjects:
            print(f"[skip] {code}: {n_subj} subj > max {args.max_subjects}", flush=True)
            continue

        print(f"\n[run] {code} (n_subj={n_subj}) ...", flush=True)
        rec = {f: "" for f in FIELDS}
        rec.update(dataset=code, paradigm=paradigm_name, n_subjects=n_subj,
                   layer=args.layer, model=args.model)
        try:
            cohort = build_moabb_cohort(dataset=dataset, paradigm=paradigm)
            n_rec = sum(1 for _ in cohort.iter_recordings())
            extractor = REVEExtractor(ch_names=cohort.ch_names, layer=args.layer,
                                      model_id=args.model)
            cfg = AuditConfig(cell_name=f"{code}-LR", cell_layout="W,C",
                              batch_size=args.batch_size, device=device,
                              erasure_cv=args.cv)
            row = audit_cell(cohort, extractor, config=cfg)

            lf, sf = _g(row, "label_frac"), _g(row, "subject_frac")
            raw = _g(row, "erasure_label_ba_raw")
            free = _g(row, "erasure_label_ba_erased")
            rec.update(
                n_recordings=n_rec,
                n_windows=int(row.get("extraction", {}).get("n_windows", 0) or 0),
                label_frac=lf, subject_frac=sf,
                residual_frac=_g(row, "residual_frac"),
                excess_label_ratio=_g(row, "excess_label_ratio"),
                excess_subject_ratio=_g(row, "excess_subject_ratio"),
                c_bar_value=_g(row, "c_bar_value"),
                raw_label_ba=raw, identity_free_label_ba=free,
                erasure_delta=_g(row, "erasure_label_ba_delta"),
                erasure_interpretable=row.get("erasure_interpretable"),
                verdict=_verdict(lf, sf, raw, free), status="ok",
            )
            print(f"[ok] {code}: raw_BA={raw:.3f} identity_free_BA={free:.3f} "
                  f"subj_frac={sf:.3f} verdict={rec['verdict']}", flush=True)
        except Exception as exc:  # noqa: BLE001 — isolate per-dataset failures.
            rec["status"] = f"FAILED: {type(exc).__name__}: {exc}"[:200]
            print(f"[FAIL] {code}: {rec['status']}", flush=True)
            traceback.print_exc()

        rows = [r for r in rows if r.get("dataset") != code] + [rec]
        _write_csv(csv_path, rows)
        _render_md(csv_path, md_path, paradigm_name)
        processed += 1

    print(f"\n[loop] done. {processed} dataset(s) this run.", flush=True)
    print(f"[loop] CSV: {csv_path}", flush=True)
    print(f"[loop] MD : {md_path}", flush=True)


if __name__ == "__main__":
    main()
