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
    "erasure_interpretable",
    "layer_label_ba_first", "layer_label_ba_last", "layer_label_ba_max",
    "layer_argmax_depth", "layer_sign",
    "state_drop", "subject_drop", "oneoverf_role",
    "verdict", "layer", "model", "status",
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


def _verdict(raw_ba, free_ba, interpretable, *, lift_eps=0.02) -> str:
    """Identity-trap verdict, read from the cross-subject *decode* (the erasure).

    Coarse flag derived purely from the erasure label-probe, which MUST be the
    per-trial decode (``--per-trial``); a recording-pooled decode crushes
    within-class variance and fabricates a lift (the artifact that made every
    cell read TRAP). The reading:

    - ``no-transfer``  : the raw decode is not interpretable (raw BA below the
                         gate) — the FM has no above-chance cross-subject task
                         signal, so there is nothing for identity to trap.
    - ``TRAP``         : interpretable raw signal AND erasing the subject axis
                         lifts it by > ``lift_eps`` — identity was masking task
                         skill that erasure recovers.
    - ``task-carried`` : interpretable raw signal that erasure does NOT lift —
                         genuine cross-subject task skill, not identity-reliant.

    ``subject_frac`` (variance dominance) is reported as its own column; it is
    deliberately NOT part of the verdict, because identity dominating the
    *variance* does not by itself imply identity carried the *decode*.
    """
    import math

    if any(math.isnan(x) for x in (raw_ba, free_ba)):
        return "n/a"
    if interpretable not in (True, "True", "true", 1, "1"):
        return "no-transfer"
    if free_ba - raw_ba > lift_eps:
        return "TRAP"
    return "task-carried"


def _summarize_layer_probe(probe) -> dict | None:
    """Reduce a :func:`reve_layer_probe` result to the four AuditConfig keys.

    ``label_ba_{first,last,max}`` are the subject-grouped label balanced
    accuracies at the shallowest / deepest / best block; ``argmax_depth`` is the
    depth-fraction (``(k+1)/n_blocks``) of the best block.
    """
    import math

    per_depth = probe.get("per_depth", []) if probe else []
    if not per_depth:
        return None
    label_bas = [d["label_ba_mean"] for d in per_depth]
    finite = [(i, v) for i, v in enumerate(label_bas) if not math.isnan(v)]
    if not finite:
        return None
    argmax_i = max(finite, key=lambda t: t[1])[0]
    return {
        "label_ba_first": label_bas[0],
        "label_ba_last": label_bas[-1],
        "label_ba_max": max(v for _, v in finite),
        "argmax_depth": per_depth[argmax_i]["depth_fraction"],
    }


def _layer_sign(summary) -> str:
    """Reproduction-only rubric for *where* label info lives across depth.

    ``+early`` — label BA peaks in the shallow third of the network
    (argmax depth-fraction ≤ 0.35) then drops ≥0.04 to the final block: task
    signal is present early but the head trades it for identity. ``-deep`` —
    final-block label BA ≤0.45: the head barely separates the task at all.
    """
    import math

    if not summary:
        return ""
    last, mx = summary["label_ba_last"], summary["label_ba_max"]
    argf = summary["argmax_depth"]
    signs = []
    if not any(math.isnan(x) for x in (mx, last, argf)):
        if argf <= 0.35 and (mx - last) >= 0.04:
            signs.append("+early")
    if not math.isnan(last) and last <= 0.45:
        signs.append("-deep")
    return ",".join(signs)


def _oneoverf_role(summary) -> str:
    """Reproduction-only rubric for what the 1/f aperiodic component carries.

    ``state signal`` — ablating 1/f costs >0.03 label BA (the aperiodic slope
    helps the task). ``subject confound`` — ablating it costs >0.05 subject BA
    (1/f is an identity fingerprint, the trap mechanism).
    """
    import math

    if not summary:
        return ""
    sd = summary.get("state_drop_mean", float("nan"))
    subd = summary.get("subject_drop_mean", float("nan"))
    roles = []
    if not math.isnan(subd) and subd > 0.05:
        roles.append("subject confound")
    if not math.isnan(sd) and sd > 0.03:
        roles.append("state signal")
    return ";".join(roles)


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
        f"Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647) "
        f"on the {paradigm_name} task contrast. **Raw BA** / **Identity-free BA** "
        "are the cross-subject task balanced-accuracy from the **per-trial** "
        "erasure decode (each trial its own recording, StratifiedGroupKFold "
        "grouped by subject; `n ≫ p`) before / after the subject subspace is "
        "erased (LEACE); **Δ = Identity-free − Raw**. NB: a recording-pooled "
        "decode crushes within-class variance and fabricates a large Δ — the "
        "artifact that made every cell read TRAP — so these numbers are per-trial. "
        "`subj_frac` = fraction of representation variance explained by subject "
        "identity (reported, but NOT part of the verdict); `c̄` = cross-subject "
        "direction-consistency of the task axis (≈0 ⇒ no axis that generalizes "
        "across people). **Verdict:** `no-transfer` = raw BA below the "
        "interpretability gate (no above-chance cross-subject task signal — "
        "nothing to trap); `TRAP` = interpretable raw signal that erasure lifts "
        "(> 0.02); `task-carried` = interpretable signal erasure does not lift "
        "(genuine, identity-robust task skill).",
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
    # Optional depth + 1/f section — only when the layer-probe / FOOOF
    # diagnostics were run for at least one dataset.
    def _diag_present(r):
        for k in ("layer_label_ba_max", "state_drop"):
            try:
                if not __import__("math").isnan(float(r[k])):
                    return True
            except (KeyError, TypeError, ValueError):
                pass
        return bool(r.get("layer_sign") or r.get("oneoverf_role"))

    diag_rows = [r for r in rows if _diag_present(r)]
    if diag_rows:
        lines += [
            "",
            "## Depth & 1/f diagnostics",
            "",
            "`L first/last/max` = subject-grouped label balanced-accuracy at the "
            "shallowest / deepest / best REVE block; `argmax` = depth-fraction of "
            "the best block. `sign`: `+early` = task signal peaks shallow then the "
            "head trades it away (max−last ≥0.04, argmax ≤0.35); `−deep` = final "
            "block barely separates the task (≤0.45). `state_drop` / `subj_drop` = "
            "label / subject BA lost when the 1/f aperiodic slope is ablated "
            "(FOOOF); `role`: `state signal` (1/f helps the task, drop >0.03) vs "
            "`subject confound` (1/f is an identity fingerprint, drop >0.05).",
            "",
            "| Dataset | L first | L last | L max | argmax | sign | state_drop | subj_drop | 1/f role |",
            "|---|---:|---:|---:|---:|---|---:|---:|---|",
        ]
        for r in diag_rows:
            lines.append(
                f"| {r['dataset']} | {fnum(r,'layer_label_ba_first')} | "
                f"{fnum(r,'layer_label_ba_last')} | {fnum(r,'layer_label_ba_max')} | "
                f"{fnum(r,'layer_argmax_depth',2)} | {r.get('layer_sign','') or '—'} | "
                f"{fnum(r,'state_drop')} | {fnum(r,'subject_drop')} | "
                f"{r.get('oneoverf_role','') or '—'} |"
            )

    failed = [r for r in _read_rows(csv_path) if r.get("status") != "ok"]
    if failed:
        lines += ["", "### Skipped / failed", ""]
        for r in failed:
            lines.append(f"- `{r['dataset']}` — {r.get('status','?')}")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


# REVE's published input contract band-passes up to 99.5 Hz, but a dataset's
# native sampling rate caps the analyzable band: MNE refuses an h_freq at/above
# the native Nyquist, and the paradigm filter applies to *every* recording, so
# fmax must clear the LOWEST native Nyquist in the cohort. PhysionetMotorImagery
# is heterogeneous — mostly 160 Hz but a handful of subjects are 128 Hz
# (Nyquist 64) — so fmax must sit below 64; we use 60. resample=200 still
# upsamples to REVE's expected rate; only the pre-resample band is clipped.
DEFAULT_FMAX = 99.5
# fmax must clear the lowest native Nyquist in the cohort (MNE rejects an h_freq
# at/above Nyquist). PhysionetMotorImagery has 128 Hz subjects (Nyquist 64);
# MAMEM3 is sampled at 128 Hz (Nyquist 64). Both cap at 60.
FMAX_OVERRIDE = {"PhysionetMotorImagery": 60.0, "MAMEM3": 60.0}


# --- Paradigm registry ------------------------------------------------------ #
# Each paradigm is built lazily (moabb imported inside the factory) with REVE's
# broadband contract (0.5–99.5 Hz, 200 Hz) rather than the paradigm's narrowband
# BCI default, so the audited features match REVE's training distribution across
# motor-imagery, ERP and SSVEP alike. ``tag`` is the short cell-name suffix; the
# label semantics differ per paradigm (left/right vs Target/NonTarget vs flicker
# frequency) but the audit machinery is label-agnostic, so the same loop serves
# all three. SSVEP/ERP get the full event set (``n_classes=None``).
def _make_leftright(fmin, fmax, resample):
    from moabb.paradigms import LeftRightImagery

    return LeftRightImagery(fmin=fmin, fmax=fmax, resample=resample)


def _make_erp(fmin, fmax, resample):
    from moabb.paradigms import P300

    return P300(fmin=fmin, fmax=fmax, resample=resample)


def _make_ssvep(fmin, fmax, resample):
    from moabb.paradigms import SSVEP

    # n_classes=2: the FMScope erasure/LP machinery (LEACE + binary LogReg) is
    # binary by construction (pos/neg counts, predict_proba[:,1], 0.5 threshold).
    # SSVEP is natively multi-frequency, so restrict to MOABB's first two flicker
    # classes to get a well-posed binary "frequency A vs B" contrast the audit
    # can score — rather than diverging from the paper's binary method.
    return SSVEP(fmin=fmin, fmax=fmax, resample=resample, n_classes=2)


PARADIGMS = {
    "leftright": {"make": _make_leftright, "display": "LeftRightImagery", "tag": "LR"},
    "erp":       {"make": _make_erp,       "display": "P300",            "tag": "P3"},
    "ssvep":     {"make": _make_ssvep,     "display": "SSVEP",           "tag": "SSV"},
}


def _build_paradigm(code, paradigm_key="leftright", fmax=None):
    if fmax is None:
        fmax = FMAX_OVERRIDE.get(code, DEFAULT_FMAX)
    return PARADIGMS[paradigm_key]["make"](0.5, fmax, 200.0)


def _nyquist_from_error(err):
    """The native Nyquist MNE rejected our fmax against, or None if unrelated.
    The static FMAX_OVERRIDE can't enumerate every low-rate ERP/SSVEP cohort
    (e.g. BrainInvaders2012 @128 Hz); recover it from MNE's message and retry."""
    import re
    m = re.search(r"Nyquist frequency\s*([0-9.]+)", str(err))
    return float(m.group(1)) if m else None


def _clamp_fmax_below(nyquist, margin=4.0):
    """Largest broadband fmax that clears a native Nyquist (MNE needs strict <);
    margin=4 maps a 64 Hz Nyquist -> 60 Hz, matching the static override."""
    return float(max(1.0, nyquist - margin))


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
    ap.add_argument("--paradigm", default="leftright",
                    choices=["leftright", "erp", "ssvep"],
                    help="MOABB paradigm to sweep: leftright (motor imagery, "
                         "left/right), erp (P300 Target/NonTarget) or ssvep "
                         "(steady-state, flicker-frequency classes).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Explicit dataset codes to run (default: all compatible).")
    ap.add_argument("--min-subjects", type=int, default=4)
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="Skip datasets with more subjects than this (0 = no cap).")
    ap.add_argument("--subject-cap", type=int, default=0,
                    help="Cap subjects LOADED per dataset (0=all); bounds peak RAM "
                         "on giant cohorts (Lee2019_ERP OOM'd the TCM run at 54).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most this many NEW datasets this run (0 = all).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--layer-probe", action="store_true",
                    help="Run the depth-wise label+subject linear probe (FMScope "
                         "diag #4) over REVE's blocks; surfaces layer_* columns "
                         "(where the task signal lives vs where the head trades it "
                         "for identity).")
    ap.add_argument("--fooof", action="store_true",
                    help="Run the FOOOF aperiodic-ablation probe (FMScope diag #3); "
                         "surfaces state_drop/subject_drop — whether the 1/f slope "
                         "carries the task or the subject identity.")
    ap.add_argument("--probe-folds", type=int, default=3,
                    help="CV folds for the layer-probe / FOOOF probes.")
    ap.add_argument("--probe-max-windows", type=int, default=0,
                    help="Cap windows per recording in the layer/FOOOF probes "
                         "(0 = no cap) to bound their cost on large cohorts.")
    ap.add_argument("--cv", default="stratified-kfold",
                    choices=["stratified-kfold", "loso"],
                    help="Erasure label-probe CV. 'loso' = leave-one-subject-out "
                         "(strictest cross-subject split); writes a separate "
                         "leaderboard_<paradigm>_loso.csv so it never clobbers the "
                         "main StratifiedGroupKFold leaderboard.")
    ap.add_argument("--per-trial", action="store_true",
                    help="Decode the subject-axis erasure PER TRIAL (each window "
                         "its own recording, grouped by subject) instead of the "
                         "default per-(subject,class) recording pooling. Pooling "
                         "crushes within-class variance and fabricates an "
                         "identity-free lift on high-dim FM features (confirmed on "
                         "BNCI2014_001 + ERP CORE); per-trial is n>>p and honest. "
                         "Writes a separate leaderboard_<paradigm>_pertrial.csv.")
    ap.add_argument("--out-dir",
                    default=os.path.expanduser("~/dev/emeg-fm/results/moabb_fmscope"))
    args = ap.parse_args()

    _append_moabb_libs()

    from emeg_fm.moabb_cohort import build_moabb_cohort
    from emeg_fm.fmscope_bridge import REVEExtractor, reve_layer_probe, fooof_role
    from fmscope.verdict import audit_cell, AuditConfig

    os.makedirs(args.out_dir, exist_ok=True)
    cv_suffix = "_loso" if args.cv == "loso" else ""
    if args.per_trial:
        cv_suffix += "_pertrial"
    csv_path = os.path.join(args.out_dir, f"leaderboard_{args.paradigm}{cv_suffix}.csv")
    md_path = os.path.join(args.out_dir, f"leaderboard_{args.paradigm}{cv_suffix}.md")

    spec = PARADIGMS[args.paradigm]
    # Registry-level paradigm only enumerates compatible datasets; per-dataset
    # fmax overrides are applied in _build_paradigm at cohort-build time.
    paradigm = spec["make"](0.5, DEFAULT_FMAX, 200.0)
    paradigm_name = spec["display"]
    cell_tag = spec["tag"]
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
            # Subject cap (bounds peak RAM; loading every subject of a giant
            # cohort OOM-killed the TCM ERP run on Lee2019_ERP at 54 subjects;
            # an OOM SIGKILL is uncatchable, so it must be prevented, not caught).
            subjects = None
            if args.subject_cap and n_subj > args.subject_cap:
                subjects = list(dataset.subject_list)[:args.subject_cap]
                print(f"[{code}] subject-cap: loading {args.subject_cap}/{n_subj} "
                      f"subjects", flush=True)
            try:
                cohort = build_moabb_cohort(
                    dataset=dataset, paradigm=_build_paradigm(code, args.paradigm),
                    subjects=subjects)
            except ValueError as exc:
                nyq = _nyquist_from_error(exc)
                if nyq is None:
                    raise
                fmax = _clamp_fmax_below(nyq)
                print(f"[{code}] native Nyquist {nyq} < broadband fmax; retry at "
                      f"fmax={fmax}", flush=True)
                cohort = build_moabb_cohort(
                    dataset=dataset,
                    paradigm=_build_paradigm(code, args.paradigm, fmax=fmax),
                    subjects=subjects)
            n_rec = sum(1 for _ in cohort.iter_recordings())
            extractor = REVEExtractor(ch_names=cohort.ch_names, layer=args.layer,
                                      model_id=args.model)

            # Optional FMScope diagnostics #4 (depth probe) and #3 (1/f role).
            # Both are extra forward passes, so flag-gated; their summaries feed
            # audit_cell via AuditConfig so the row carries the layer_*/1-f cols.
            mw = args.probe_max_windows or None
            layer_summary = oneoverf_summary = None
            if args.layer_probe:
                lp = reve_layer_probe(extractor, cohort,
                                      batch_size=args.batch_size,
                                      n_folds=args.probe_folds,
                                      max_windows_per_recording=mw)
                layer_summary = _summarize_layer_probe(lp)
                print(f"[probe] {code}: layer label BA "
                      f"first={ (layer_summary or {}).get('label_ba_first', float('nan')):.3f} "
                      f"last={ (layer_summary or {}).get('label_ba_last', float('nan')):.3f} "
                      f"max={ (layer_summary or {}).get('label_ba_max', float('nan')):.3f}",
                      flush=True)
            if args.fooof:
                oneoverf_summary = fooof_role(
                    extractor, cohort,
                    sfreq=float(getattr(cohort, "sfreq", 200.0)),
                    batch_size=args.batch_size, n_folds=args.probe_folds,
                    max_windows_per_recording=mw)
                print(f"[fooof] {code}: state_drop="
                      f"{oneoverf_summary['state_drop_mean']:.3f} subject_drop="
                      f"{oneoverf_summary['subject_drop_mean']:.3f}", flush=True)

            cfg = AuditConfig(cell_name=f"{code}-{cell_tag}", cell_layout="W,C",
                              batch_size=args.batch_size, device=device,
                              erasure_cv=args.cv,
                              erasure_per_trial=args.per_trial,
                              layer_probe=layer_summary, oneoverf=oneoverf_summary)
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
                layer_label_ba_first=_g(row, "layer_label_ba_first"),
                layer_label_ba_last=_g(row, "layer_label_ba_last"),
                layer_label_ba_max=_g(row, "layer_label_ba_max"),
                layer_argmax_depth=_g(row, "layer_argmax_depth"),
                layer_sign=_layer_sign(layer_summary),
                state_drop=_g(row, "state_drop"),
                subject_drop=_g(row, "subject_drop"),
                oneoverf_role=_oneoverf_role(oneoverf_summary),
                verdict=_verdict(raw, free, row.get("erasure_interpretable")),
                status="ok",
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
