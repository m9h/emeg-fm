#!/usr/bin/env python
"""Wang TCM (Topographic Component Model) in the per-trial MOABB identity audit.

Adds Wang/Begleiter/Porjesz (2000) TCM — a per-mode trilinear decomposition whose
per-trial loadings L = B^T x C are compact features — as a feature family in the
SAME per-trial identity-trap audit used for the FMs (REVE, LuMamba). For each
MOABB dataset: raw windows -> fit per-mode TCM bases over all trials (unsupervised)
-> project each trial to its loading -> subject_axis_erasure both pooled
(subject*class) and per-trial (n>>p), grouped by subject. Directly comparable to
the REVE/LuMamba per-trial leaderboard and to the classical-Riemannian control.

TCM math vendored (MIT) from smni-cmi `smni_cmi.multiway` (Ingber/SMNI program);
pure NumPy SVD, no JAX/GPU. CMI/PATHINT (the other smni-cmi pieces) are
within-subject tools that fail cross-subject by their own validation, so TCM is
the piece that belongs in a cross-subject decode sweep.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


# --- vendored Wang TCM (smni_cmi.multiway, MIT) -------------------------------
def _unfold(X, mode):
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def _rank_by_variance(sv, frac):
    var = sv ** 2
    var = var / var.sum()
    return int(np.searchsorted(np.cumsum(var), frac) + 1)


def tcm_components(tensor, spatial_var=0.90, temporal_var=0.95):
    """Per-mode SVD bases with separate variance-threshold ranks (Wang-faithful).
    tensor (C, T, N) -> (B (C,Ks), C (Kt,T), Ks, Kt). Unsupervised."""
    Ds = _unfold(tensor, 0)
    Us, ss, _ = np.linalg.svd(Ds, full_matrices=False)
    Ks = min(_rank_by_variance(ss, spatial_var), Us.shape[1])
    Dt = _unfold(tensor, 1)
    Ut, st, _ = np.linalg.svd(Dt, full_matrices=False)
    Kt = min(_rank_by_variance(st, temporal_var), Ut.shape[1])
    return Us[:, :Ks], Ut[:, :Kt].T, Ks, Kt


def tcm_loading(slab, B, C):
    return (B.T @ slab @ C.T).ravel()
# -----------------------------------------------------------------------------


FIELDS = ["dataset", "n_subjects", "n_trials", "n_channels", "Ks", "Kt", "n_feat",
          "pooled_raw", "pooled_erased", "pooled_delta",
          "pertrial_raw", "pertrial_erased", "pertrial_delta",
          "subj_ba_pre", "subj_ba_post", "subj_chance",
          "interpretable", "degenerate", "verdict", "status"]


def _verdict(raw, free, interp, lift_eps=0.02):
    import math
    if any(math.isnan(x) for x in (raw, free)):
        return "n/a"
    if interp not in (True, "True", 1):
        return "no-transfer"
    return "TRAP" if (free - raw) > lift_eps else "task-carried"


def _flatten_cohort(cohort):
    Xs, subj, lab = [], [], []
    for sid, label, windows in cohort.iter_recordings():
        w = np.asarray(windows)
        Xs.append(w)
        subj.extend([sid] * len(w))
        lab.extend([label] * len(w))
    return np.concatenate(Xs, 0), np.asarray(subj), np.asarray(lab)


def _run_dataset(code, dataset, args):
    from emeg_fm.moabb_cohort import build_moabb_cohort
    from moabb.paradigms import LeftRightImagery
    from fmscope.diagnostics.erasure import subject_axis_erasure

    paradigm = LeftRightImagery(fmin=0.5, fmax=99.5, resample=200.0)
    cohort = build_moabb_cohort(dataset=dataset, paradigm=paradigm, normalize=False)
    X, subj, labels = _flatten_cohort(cohort)            # X (N, C, T)
    n, C, T = X.shape
    tensor = np.moveaxis(X, 0, 2)                        # (C, T, N)
    # Cap trials for the per-mode SVD basis (the time-unfold is (T, C*N) and
    # OOMs on large cohorts, e.g. Stieger2021); all trials are still projected.
    if n > args.max_basis_trials:
        sub = np.random.RandomState(0).choice(n, args.max_basis_trials, replace=False)
        B, Cb, Ks, Kt = tcm_components(tensor[:, :, sub], args.spatial_var,
                                       args.temporal_var)
    else:
        B, Cb, Ks, Kt = tcm_components(tensor, args.spatial_var, args.temporal_var)
    feats = np.stack([tcm_loading(X[i], B, Cb) for i in range(n)]).astype(np.float64)
    print(f"[{code}] N={n} C={C} T={T} -> TCM Ks={Ks} Kt={Kt} feat={feats.shape[1]} "
          f"subj={np.unique(subj).size}", flush=True)

    pooled = subject_axis_erasure(feats, subj, labels, cv="stratified-kfold")
    per = subject_axis_erasure(feats, subj, labels, window_recording=np.arange(n),
                               rec_labels=labels, rec_pids=subj, cv="stratified-kfold")
    v = _verdict(per.label_ba_raw, per.label_ba_erased, per.interpretable)
    print(f"[{code}] pooled {pooled.label_ba_raw:.3f}->{pooled.label_ba_erased:.3f} "
          f"(Δ{pooled.label_ba_delta:+.3f}) | per-trial {per.label_ba_raw:.3f}->"
          f"{per.label_ba_erased:.3f} (Δ{per.label_ba_delta:+.3f}) "
          f"subj {per.subj_ba_linear_pre:.2f}->{per.subj_ba_linear_post:.2f} "
          f"=> {v}", flush=True)

    def r(x):
        return round(float(x), 4) if np.isfinite(x) else float("nan")
    return {
        "dataset": code, "n_subjects": int(np.unique(subj).size), "n_trials": n,
        "n_channels": C, "Ks": Ks, "Kt": Kt, "n_feat": feats.shape[1],
        "pooled_raw": r(pooled.label_ba_raw), "pooled_erased": r(pooled.label_ba_erased),
        "pooled_delta": r(pooled.label_ba_delta),
        "pertrial_raw": r(per.label_ba_raw), "pertrial_erased": r(per.label_ba_erased),
        "pertrial_delta": r(per.label_ba_delta),
        "subj_ba_pre": r(per.subj_ba_linear_pre), "subj_ba_post": r(per.subj_ba_linear_post),
        "subj_chance": r(per.chance), "interpretable": bool(per.interpretable),
        "degenerate": bool(per.degenerate), "verdict": v, "status": "ok",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="dataset codes; default = all LeftRightImagery")
    ap.add_argument("--min-subjects", type=int, default=4)
    ap.add_argument("--max-subjects", type=int, default=120)
    ap.add_argument("--spatial-var", type=float, default=0.90)
    ap.add_argument("--temporal-var", type=float, default=0.95)
    ap.add_argument("--max-basis-trials", type=int, default=4000,
                    help="cap trials used for the per-mode SVD basis (bounds "
                         "memory on large cohorts; all trials are still projected)")
    # Repo-relative (NOT ~, which is the container home under -w /emeg-fm).
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results",
        "moabb_fmscope", "tcm_pertrial.csv"))
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import mne
    mne.set_log_level("error")
    from moabb.paradigms import LeftRightImagery

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {r["dataset"] for r in csv.DictReader(f)}
    rows = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            rows = list(csv.DictReader(f))

    paradigm = LeftRightImagery(fmin=0.5, fmax=99.5, resample=200.0)
    want = set(args.datasets) if args.datasets else None
    todo = []
    for d in paradigm.datasets:
        code = getattr(d, "code", d.__class__.__name__)
        if want and code not in want:
            continue
        if "fake" in code.lower() or code in done:
            continue
        if hasattr(d, "accept"):
            d.accept = True
        todo.append((code, d))
    print(f"[sweep] TCM per-trial: {len(todo)} datasets to run", flush=True)

    for code, dataset in todo:
        try:
            ns = len(dataset.subject_list)
            if ns < args.min_subjects or ns > args.max_subjects:
                print(f"[skip] {code}: {ns} subjects out of range", flush=True)
                continue
            rows.append(_run_dataset(code, dataset, args))
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {code}: {type(exc).__name__}: {exc}", flush=True)
            rows.append({f: "" for f in FIELDS} | {"dataset": code,
                        "status": f"FAILED: {type(exc).__name__}"})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
