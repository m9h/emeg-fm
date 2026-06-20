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
import re

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


def _tcm_loadings_batch(X, B, C):
    """Project a whole (N, C, T) trial stack to (N, Ks*Kt) loadings in two
    tensordots (best practice adopted from smni-cmi ``moabb_shootout.tcm_feat``);
    numerically identical to looping ``tcm_loading`` per trial, but far faster on
    the large cross-subject cohorts this audit runs (e.g. Lee2019, Stieger2021)."""
    tmp = np.tensordot(X, B, axes=([1], [0]))        # (N, T, Ks)
    return np.tensordot(tmp, C, axes=([1], [1])).reshape(len(X), -1)  # (N, Ks*Kt)
# -----------------------------------------------------------------------------


# --- Paradigm registry ------------------------------------------------------ #
# Mirrors moabb_identity_leaderboard.py: every paradigm is built lazily under the
# SAME broadband contract the FM audit uses (0.5–99.5 Hz, 200 Hz) rather than its
# narrowband BCI default, so the TCM loadings the audit sees match the FM/control
# feature distribution across motor-imagery, ERP and SSVEP alike. The erasure is
# label-agnostic, so one loop serves all three; only SSVEP needs the binary
# n_classes=2 restriction the LEACE step assumes.
DEFAULT_FMAX = 99.5
# fmax must clear the lowest native Nyquist in the cohort (MNE rejects h_freq at/
# above Nyquist). 128 Hz cohorts (Nyquist 64) cap at 60.
FMAX_OVERRIDE = {"PhysionetMotorImagery": 60.0, "MAMEM3": 60.0}


def _make_leftright(fmin, fmax, resample):
    from moabb.paradigms import LeftRightImagery
    return LeftRightImagery(fmin=fmin, fmax=fmax, resample=resample)


def _make_erp(fmin, fmax, resample):
    from moabb.paradigms import P300
    return P300(fmin=fmin, fmax=fmax, resample=resample)


def _make_ssvep(fmin, fmax, resample):
    from moabb.paradigms import SSVEP
    # n_classes=2: the LEACE erasure / binary LogReg probe is binary by
    # construction; SSVEP is natively multi-frequency, so restrict to the first
    # two flicker classes for a well-posed "frequency A vs B" contrast.
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
    """The Nyquist freq MNE rejected our fmax against, or None if unrelated.

    The static FMAX_OVERRIDE can't enumerate every low-rate cohort (ERP/SSVEP add
    many 128/256 Hz sets), so we recover the native Nyquist from MNE's own message
    and retry below it. None => re-raise (don't mask non-Nyquist failures)."""
    m = re.search(r"Nyquist frequency\s*([0-9.]+)", str(err))
    return float(m.group(1)) if m else None


def _clamp_fmax_below(nyquist, margin=4.0):
    """Largest broadband fmax that clears a native Nyquist (MNE needs strict <).
    margin=4 maps a 64 Hz Nyquist -> 60 Hz, matching the static override."""
    return float(max(1.0, nyquist - margin))
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
    from fmscope.diagnostics.erasure import subject_axis_erasure

    paradigm = _build_paradigm(code, args.paradigm)
    # Subject cap (smni-cmi SUBJ_CAP best practice): loading every subject of a
    # giant cohort at once OOM-killed Lee2019_ERP (54 subjects). Bound peak RAM by
    # loading at most --subject-cap subjects; 0 = all. The cross-subject erasure
    # stays well-posed with a couple dozen subjects.
    subjects = None
    if args.subject_cap and len(dataset.subject_list) > args.subject_cap:
        subjects = list(dataset.subject_list)[:args.subject_cap]
        print(f"[{code}] subject-cap: loading {args.subject_cap}/"
              f"{len(dataset.subject_list)} subjects", flush=True)
    try:
        cohort = build_moabb_cohort(dataset=dataset, paradigm=paradigm,
                                    normalize=False, subjects=subjects)
    except ValueError as exc:
        nyq = _nyquist_from_error(exc)
        if nyq is None:
            raise
        fmax = _clamp_fmax_below(nyq)
        print(f"[{code}] native Nyquist {nyq} < broadband fmax; retry at "
              f"fmax={fmax}", flush=True)
        paradigm = _build_paradigm(code, args.paradigm, fmax=fmax)
        cohort = build_moabb_cohort(dataset=dataset, paradigm=paradigm,
                                    normalize=False, subjects=subjects)
    X, subj, labels = _flatten_cohort(cohort)            # X (N, C, T)
    n, C, T = X.shape
    # Cap trials for the per-mode SVD basis (the time-unfold is (T, C*N) and OOMs
    # on large cohorts, e.g. Stieger2021); all trials are still projected. Build
    # the (C, T, K) basis tensor from the subset DIRECTLY — never moveaxis the
    # full (N, C, T) stack into a second (C, T, N) copy (that doubling helped
    # OOM-kill Lee2019_ERP).
    if n > args.max_basis_trials:
        sub = np.random.RandomState(0).choice(n, args.max_basis_trials, replace=False)
        basis_tensor = np.moveaxis(X[sub], 0, 2)
    else:
        basis_tensor = np.moveaxis(X, 0, 2)
    B, Cb, Ks, Kt = tcm_components(basis_tensor, args.spatial_var, args.temporal_var)
    del basis_tensor
    feats = _tcm_loadings_batch(X, B, Cb).astype(np.float64)
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
    ap.add_argument("--paradigm", default="leftright",
                    choices=["leftright", "erp", "ssvep"],
                    help="MOABB paradigm to sweep: leftright (motor imagery), "
                         "erp (P300 Target/NonTarget) or ssvep (binary "
                         "flicker-frequency contrast).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="dataset codes; default = all of the chosen paradigm")
    ap.add_argument("--min-subjects", type=int, default=4)
    ap.add_argument("--max-subjects", type=int, default=120)
    ap.add_argument("--spatial-var", type=float, default=0.90)
    ap.add_argument("--temporal-var", type=float, default=0.95)
    ap.add_argument("--max-basis-trials", type=int, default=4000,
                    help="cap trials used for the per-mode SVD basis (bounds "
                         "memory on large cohorts; all trials are still projected)")
    ap.add_argument("--subject-cap", type=int, default=0,
                    help="cap subjects LOADED per dataset (0=all); bounds peak "
                         "RAM on giant cohorts (Lee2019_ERP OOM'd at 54 subjects)")
    # Repo-relative (NOT ~, which is the container home under -w /emeg-fm).
    # Default resolved per-paradigm below so ERP/SSVEP runs never clobber the MI
    # CSV: leftright -> tcm_pertrial.csv (back-compat); else tcm_pertrial_<p>.csv.
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        stem = "tcm_pertrial" if args.paradigm == "leftright" \
            else f"tcm_pertrial_{args.paradigm}"
        args.out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "results",
            "moabb_fmscope", f"{stem}.csv")

    import warnings
    warnings.filterwarnings("ignore")
    import mne
    mne.set_log_level("error")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Resume on completed (status == "ok") rows only: a prior FAILED row (e.g. a
    # transient download error or a Nyquist clip now auto-handled) should be
    # retried, not permanently skipped. Keep only ok rows so the retry doesn't
    # leave a duplicate behind.
    done = set()
    rows = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    done.add(r["dataset"])
                    rows.append(r)

    paradigm = _build_paradigm("__listing__", args.paradigm)
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
