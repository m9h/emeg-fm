#!/usr/bin/env python
"""Full-parity ERP CORE decoding comparison against the Steve Luck lab.

Compares three decoders on the seven ERP CORE components (Kappenman et al.
2021) under the *exact* ERPLAB12 ``pop_decoding_regularization`` protocol
(OSF un6pq, ``m_1_decoding_ERPCORE_regularization.m``): linear SVM, pseudo-ERP
crossblock CV (3 folds), nIter resampling, EqualizeTrials='classes', and the
``Gamma_Value`` regularization grid swept as the SVM box constraint.

Columns per component:
  - ``scalp_within``    : ERPLAB DECODE classical baseline — linear SVM on raw
                          scalp voltage, decoded across the component
                          measurement window, **within participant** (Luck's
                          native protocol). Best over the Gamma grid.
  - ``reve_within``     : same within-participant protocol on frozen REVE
                          (block-6) trial embeddings. Best over the Gamma grid.
  - ``reve_cross_raw``  : cross-subject decode of the component label on REVE
                          embeddings, measured the way the rest of FMScope
                          measures it (``subject_axis_erasure``): one trial =
                          one recording, ``StratifiedGroupKFold`` grouped by
                          subject (train/test never share a subject), balanced
                          accuracy. Single-trial, ``n ≫ p`` — reflects real
                          generalization, not separability.
  - ``reve_cross_free`` : the same cross-subject decode after the subject-
                          identity subspace is erased (LEACE; Belrose et al.
                          2023). ``lift = reve_cross_free − reve_cross_raw`` is
                          the identity-trap lift FMScope surfaces.
  - ``degenerate`` / ``interpretable`` : validity flags from the erasure. The
                          lift is meaningful only when the subject subspace does
                          not fill the ambient space (``rank < 0.95·dim``) and
                          the raw decode clears the interpretability gate (≥
                          0.55). ``subj_ba_pre/post`` document that an actual
                          linear identity axis was present and removed.

``scalp_within`` vs ``reve_within`` answers "does the FM embedding beat raw
scalp voltage under Luck's own protocol?"; the cross trio answers "is the FM's
cross-subject ERP decoding riding on subject identity?".

The cross-subject decode deliberately does *not* use the pseudo-ERP block
protocol used within-subject: averaging each subject-block's trials into a
handful of near-noiseless points and decoding them in REVE's high-dim embedding
is ``p ≫ n`` separable and saturates to ~1.0 after erasure (the block-offset is
the only thing LEACE removes). Single-trial group-CV is the metric that is
meaningful for these features.

Component epoch windows follow Luck (epochTW), which differ from MOABB's
``ErpCore2021`` ``interval`` defaults for ERN/LRP (MOABB swaps the two) — we
override ``dataset.interval`` to Luck's values for parity.

Runtime: Docker NGC PyTorch 26.05 + uv on /mnt/t9 (REVE needs torch+GPU and the
gated ``brain-bzh/reve-base`` checkpoint). Never /data NFS / NFS apptainer SIF.
See ``scripts/erpcore_luck_parity.sbatch``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback

import numpy as np


# Luck ERP CORE protocol constants (m_1_decoding_ERPCORE_regularization.m + the
# per-component measurement windows in the ERP CORE decoding scripts).
# (epoch_tmin, epoch_tmax, win_lo, win_hi) in seconds. epoch_* = epochTW;
# win_* = the component measurement window decoding is summarised over.
COMPONENTS = {
    "N170": ("ErpCore2021_N170", (-0.2, 0.8), (0.110, 0.150)),
    "MMN":  ("ErpCore2021_MMN",  (-0.2, 0.8), (0.125, 0.225)),
    "N2pc": ("ErpCore2021_N2pc", (-0.2, 0.8), (0.200, 0.275)),
    "P3":   ("ErpCore2021_P3",   (-0.2, 0.8), (0.300, 0.600)),
    "N400": ("ErpCore2021_N400", (-0.2, 0.8), (0.300, 0.500)),
    "ERN":  ("ErpCore2021_ERN",  (-0.6, 0.4), (0.000, 0.100)),
    "LRP":  ("ErpCore2021_LRP",  (-0.8, 0.2), (-0.100, 0.000)),
}

GAMMA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

FIELDS = [
    "component", "n_subjects", "n_trials", "n_channels",
    "epoch_tmin", "epoch_tmax", "win_lo", "win_hi",
    "scalp_within", "scalp_within_bestC",
    "reve_within", "reve_within_bestC",
    "reve_cross_raw", "reve_cross_free", "identity_free_lift", "lift_std",
    "subj_ba_pre", "subj_ba_post", "subj_chance", "subj_axis_rank",
    "degenerate", "interpretable", "status",
]


def _append_moabb_libs() -> None:
    libs = os.environ.get("MOABB_LIBS")
    if libs and os.path.isdir(libs) and libs not in sys.path:
        sys.path.append(libs)


def _pick_device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_component_epochs(dataset_cls_name, interval, fmax, sfreq_out,
                           max_subjects):
    """Return (raw (n_trials,C,T) volts, times, y∈{0,1}, subj, ch_names).

    Overrides ``dataset.interval`` to Luck's epochTW, then epochs every subject
    with the broadband REVE contract (0.5–``fmax`` Hz, ``sfreq_out``).
    """
    import moabb.datasets as mds
    from moabb.paradigms import P300

    dataset = getattr(mds, dataset_cls_name)()
    dataset.interval = interval  # Luck epochTW (overrides MOABB default)
    subjects = list(dataset.subject_list)
    if max_subjects:
        subjects = subjects[:max_subjects]

    paradigm = P300(fmin=0.5, fmax=fmax, resample=sfreq_out)
    epochs, y, meta = paradigm.get_data(dataset, subjects=subjects,
                                        return_epochs=True)
    raw = epochs.get_data(copy=True)  # (n_trials, C, T) Volts
    times = np.asarray(epochs.times)
    classes = sorted({str(v) for v in y})
    label_map = {c: i for i, c in enumerate(classes)}
    labels = np.asarray([label_map[str(v)] for v in y])
    subj = np.asarray(meta["subject"])
    return raw, times, labels, subj, list(epochs.ch_names)


def _reve_features(raw, subj, ch_names, *, layer, model_id, sfreq_out, clamp,
                   batch_size, device):
    """Per-subject REVE-contract normalise then extract per-trial embeddings.

    Mirrors :func:`emeg_fm.moabb_cohort.build_moabb_cohort`'s per-subject frozen
    scaler so cross-trial amplitude survives and REVE sees its training input
    distribution; returns ``(n_trials, embed_dim)`` aligned with ``raw``.
    """
    from emeg_fm.alljoined import ReveInputNorm
    from emeg_fm.fmscope_bridge import REVEExtractor

    extractor = REVEExtractor(ch_names=ch_names, layer=layer, model_id=model_id)
    feats = np.zeros((raw.shape[0], extractor.embed_dim), dtype=np.float32)
    for s in np.unique(subj):
        m = np.where(subj == s)[0]
        norm = ReveInputNorm(sfreq_out=sfreq_out, clamp=clamp).fit(
            raw[m], sfreq_in=sfreq_out)
        win = norm.transform(raw[m], sfreq_in=sfreq_out).astype(np.float32)
        for i in range(0, len(win), batch_size):
            feats[m[i:i + batch_size]] = extractor(win[i:i + batch_size])
    return feats


def _within_subject_best(decode_fn, subj):
    """Grand-average over subjects of each subject's best-Gamma accuracy.

    ``decode_fn(C, subject_mask)`` returns one accuracy; we sweep the Gamma grid
    per subject, take that subject's best, then average across subjects. Returns
    ``(grand_acc, modal_best_C)``.
    """
    per_subj_best, per_subj_argC = [], []
    for s in np.unique(subj):
        m = subj == s
        accs = {C: decode_fn(C, m) for C in GAMMA_GRID}
        finite = {C: a for C, a in accs.items() if np.isfinite(a)}
        if not finite:
            continue
        bestC = max(finite, key=lambda C: finite[C])
        per_subj_best.append(finite[bestC])
        per_subj_argC.append(bestC)
    if not per_subj_best:
        return float("nan"), float("nan")
    # Modal best-C across subjects (most-often-selected regularization).
    vals, counts = np.unique(per_subj_argC, return_counts=True)
    return float(np.mean(per_subj_best)), float(vals[int(np.argmax(counts))])


def _run_component(name, cfg, args, device):
    from fmscope.training.svm_probe import (
        erplab_decode_scalp, luck_svm_decode,
    )

    dataset_cls_name, (etmin, etmax), (wlo, whi) = cfg
    raw, times, y, subj, ch_names = _load_component_epochs(
        dataset_cls_name, (etmin, etmax), args.fmax, args.sfreq_out,
        args.max_subjects)
    if args.max_channels and len(ch_names) > args.max_channels:
        keep = slice(0, args.max_channels)
        raw = raw[:, keep, :]
        ch_names = ch_names[: args.max_channels]

    n_trials, n_ch = raw.shape[0], raw.shape[1]
    print(f"[{name}] trials={n_trials} ch={n_ch} subj={np.unique(subj).size} "
          f"epoch=({etmin},{etmax}) win=({wlo},{whi})", flush=True)

    # --- Classical scalp baseline (within subject, ERPLAB DECODE) --- #
    # Crop to the measurement window so the time-resolved decode stays cheap.
    win_mask = (times >= wlo) & (times <= whi)
    raw_win, times_win = raw[:, :, win_mask], times[win_mask]

    def scalp_decode(C, m):
        return erplab_decode_scalp(
            raw_win[m], y[m], times_win, window=(wlo, whi),
            decode_every=args.decode_every, crossfold=args.crossfold,
            n_iter=args.n_iter, C=C, seed=args.seed)["window_acc"]

    scalp_acc, scalp_C = _within_subject_best(scalp_decode, subj)
    print(f"[{name}] scalp_within={scalp_acc:.3f} (C={scalp_C})", flush=True)

    # --- REVE embeddings (within + cross subject) --- #
    feats = _reve_features(
        raw, subj, ch_names, layer=args.layer, model_id=args.model,
        sfreq_out=args.sfreq_out, clamp=args.clamp,
        batch_size=args.batch_size, device=device)

    def reve_decode(C, m):
        return luck_svm_decode(feats[m], y[m], crossfold=args.crossfold,
                               n_iter=args.n_iter, C=C, seed=args.seed)

    reve_acc, reve_C = _within_subject_best(reve_decode, subj)
    print(f"[{name}] reve_within={reve_acc:.3f} (C={reve_C})", flush=True)

    # Cross-subject raw vs identity-free (LEACE), measured the way the rest of
    # FMScope measures the identity trap: single-trial decode, one recording
    # per trial, StratifiedGroupKFold grouped by subject (train/test never
    # share a subject), balanced accuracy. This is n >> p and carries its own
    # degeneracy / interpretability guards — unlike a pseudo-ERP block decode,
    # which saturates to ~1.0 on these high-dim embeddings.
    from fmscope.diagnostics.erasure import subject_axis_erasure
    er = subject_axis_erasure(
        feats, subj, y,
        window_recording=np.arange(n_trials),
        rec_labels=y, rec_pids=subj,
        cv="stratified-kfold",
    )
    lift = er.label_ba_delta
    print(f"[{name}] reve_cross_raw={er.label_ba_raw:.3f} "
          f"reve_cross_free={er.label_ba_erased:.3f} lift={lift:+.3f} "
          f"(subj_ba {er.subj_ba_linear_pre:.2f}->{er.subj_ba_linear_post:.2f}, "
          f"rank={er.rank_subject_axis}, degenerate={er.degenerate}, "
          f"interpretable={er.interpretable})", flush=True)

    def _r(v, n=4):
        return round(float(v), n) if np.isfinite(v) else float("nan")

    return {
        "component": name, "n_subjects": int(np.unique(subj).size),
        "n_trials": int(n_trials), "n_channels": int(n_ch),
        "epoch_tmin": etmin, "epoch_tmax": etmax, "win_lo": wlo, "win_hi": whi,
        "scalp_within": round(scalp_acc, 4), "scalp_within_bestC": scalp_C,
        "reve_within": round(reve_acc, 4), "reve_within_bestC": reve_C,
        "reve_cross_raw": _r(er.label_ba_raw),
        "reve_cross_free": _r(er.label_ba_erased),
        "identity_free_lift": _r(lift), "lift_std": _r(er.label_ba_delta_std),
        "subj_ba_pre": _r(er.subj_ba_linear_pre),
        "subj_ba_post": _r(er.subj_ba_linear_post),
        "subj_chance": _r(er.chance),
        "subj_axis_rank": int(er.rank_subject_axis),
        "degenerate": bool(er.degenerate),
        "interpretable": bool(er.interpretable),
        "status": "ok",
    }


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _render_md(path, rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    lines = [
        "# ERP CORE decoding — Luck-lab parity comparison",
        "",
        "**Within-participant** (Luck's native regime): linear-SVM pseudo-ERP "
        "decoding under the ERPLAB12 `pop_decoding_regularization` protocol "
        "(3-fold crossblock, Gamma grid, EqualizeTrials), best over the grid. "
        "`scalp_within` is raw scalp voltage (the ERPLAB DECODE baseline); "
        "`reve_within` is frozen REVE block-6 embeddings under the same "
        "protocol.",
        "",
        "**Cross-subject** (`reve_cross_*`): single-trial decode of the "
        "component label, one trial = one recording, StratifiedGroupKFold "
        "grouped by subject (train/test never share a subject), balanced "
        "accuracy — `n ≫ p`, so the score reflects generalization, not "
        "separability. `raw` → before, `free` → after LEACE subject-axis "
        "erasure; `lift = free − raw` (± across-seed SD) is the identity-trap "
        "lift. It is meaningful only when **interpretable** (raw ≥ 0.55) and "
        "not **degenerate** (subject subspace < 0.95·dim); `subj BA` shows the "
        "linear identity axis (pre → post erasure, vs chance) that was removed.",
        "",
        "| Component | N | Scalp (within) | REVE (within) | REVE cross raw "
        "| REVE cross id-free | Lift | subj BA (pre→post / chance) | valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:--|",
    ]

    def f(r, k):
        v = r.get(k)
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return "—"

    for r in ok:
        lift = f(r, "identity_free_lift")
        sd = r.get("lift_std")
        if isinstance(sd, (int, float)) and np.isfinite(sd):
            lift = f"{lift} ±{float(sd):.3f}"
        deg = r.get("degenerate")
        interp = r.get("interpretable")
        valid = "✓" if (interp and not deg) else (
            "degenerate" if deg else "not interp.")
        subj = (f"{f(r,'subj_ba_pre')}→{f(r,'subj_ba_post')} "
                f"/ {f(r,'subj_chance')}")
        lines.append(
            f"| {r['component']} | {r['n_subjects']} | {f(r,'scalp_within')} | "
            f"{f(r,'reve_within')} | {f(r,'reve_cross_raw')} | "
            f"{f(r,'reve_cross_free')} | {lift} | {subj} | {valid} |")
    failed = [r for r in rows if r.get("status") != "ok"]
    if failed:
        lines += ["", "### Failed", ""]
        lines += [f"- `{r['component']}` — {r.get('status')}" for r in failed]
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--components", nargs="*", default=None,
                    help="Subset of ERP CORE components (default: all 7).")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--model", default="brain-bzh/reve-base")
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="Cap subjects per component (0 = all 40).")
    ap.add_argument("--max-channels", type=int, default=28,
                    help="Use first N EEG channels (Luck decoded 28; 0 = all).")
    ap.add_argument("--n-iter", type=int, default=50,
                    help="Pseudo-ERP resamples (ERPLAB nIter=100; 50 default "
                         "for tractable grid×subject×component cost).")
    ap.add_argument("--crossfold", type=int, default=3)
    ap.add_argument("--decode-every", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--sfreq-out", type=float, default=200.0)
    ap.add_argument("--fmax", type=float, default=99.5)
    ap.add_argument("--clamp", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir",
                    default=os.path.expanduser("~/dev/emeg-fm/results/erpcore_luck_parity"))
    args = ap.parse_args()

    _append_moabb_libs()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "erpcore_luck_parity.csv")
    md_path = os.path.join(args.out_dir, "erpcore_luck_parity.md")
    device = _pick_device()

    comps = args.components or list(COMPONENTS)
    print(f"[parity] components={comps} device={device} "
          f"n_iter={args.n_iter} crossfold={args.crossfold}", flush=True)

    rows = []
    for name in comps:
        if name not in COMPONENTS:
            print(f"[skip] unknown component {name}", flush=True)
            continue
        try:
            rows.append(_run_component(name, COMPONENTS[name], args, device))
        except Exception as exc:  # noqa: BLE001 — isolate per-component failures
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            rows.append({**{k: "" for k in FIELDS}, "component": name,
                         "status": f"FAILED: {type(exc).__name__}: {exc}"[:200]})
        _write_csv(csv_path, rows)
        _render_md(md_path, rows)

    print(f"\n[parity] done. CSV: {csv_path}\n[parity] MD : {md_path}",
          flush=True)


if __name__ == "__main__":
    main()
