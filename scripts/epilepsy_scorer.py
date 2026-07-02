"""Event-level seizure-detection scoring + a reproducible gradient-boosting baseline
for the NeuroTechX-Atlas epilepsy section (TUSZ).

Follows the Sci-Rep 2026 "transparent AI assurance" TUSZ protocol + NeuroAtlas's
Event-Sens@FA metric:
  - continuous-EEG, patient-disjoint official Train/Dev/Eval split (never mix patients);
  - per-window seizure scores -> post-process (threshold, min-duration, merge-gap) into
    predicted EVENTS -> any-overlap event scoring vs the .csv_bi ground truth;
  - report event sensitivity across a false-alarm budget (0.1-100 FA/h) and its AUC
    over log10(FA/h) — the clinically meaningful metric, not window AUROC.
Operating point (threshold + post-processing) is chosen on Dev, applied to Eval.

The SAME scorer serves the classical anchor (expert features -> gradient boosting) AND
any frozen-FM window embeddings (braindecode zoo -> linear probe -> per-window score),
so the FM-vs-classical epilepsy comparison is apples-to-apples. Our added axis: fit the
probe with subject/patient identity LEACE-erased -> identity-free Event-Sens@FA (seizure
EEG is highly patient-specific -> prime identity-trap territory).
"""
from __future__ import annotations

import numpy as np


# ---------- ground-truth annotations ----------
def load_csv_bi(path):
    """Parse a TUSZ ``.csv_bi`` file -> list of (start_s, stop_s) SEIZ events.

    Format: header lines starting '#', then rows
    ``channel,start_time,stop_time,label,confidence`` where label in {seiz,bckg}
    (term-based files use TERM as channel). Only seizure ('seiz') rows kept."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("channel"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            _ch, start, stop, label = parts[0], parts[1], parts[2], parts[3].strip().lower()
            if label.startswith("seiz") or label in ("fnsz", "gnsz", "cpsz", "spsz",
                                                      "absz", "tnsz", "tcsz", "mysz"):
                events.append((float(start), float(stop)))
    return _merge_overlaps(sorted(events))


def _merge_overlaps(events, gap=0.0):
    """Merge overlapping/near (<= gap s) intervals."""
    out = []
    for s, e in sorted(events):
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


# ---------- window scores -> predicted events ----------
def scores_to_events(scores, win_starts, win_dur, threshold,
                     min_event_s=1.0, merge_gap_s=1.0):
    """Threshold per-window scores, merge consecutive above-threshold windows into
    events, drop events shorter than ``min_event_s``. Returns [(start_s, stop_s)]."""
    above = np.asarray(scores) >= threshold
    raw = [(float(win_starts[i]), float(win_starts[i] + win_dur))
           for i in np.where(above)[0]]
    merged = _merge_overlaps(raw, gap=merge_gap_s)
    return [(s, e) for s, e in merged if (e - s) >= min_event_s]


# ---------- any-overlap event scoring ----------
def event_confusion(pred_events, true_events):
    """Any-overlap scoring (SzCORE-style):
      TP = true events overlapped by >=1 prediction; FN = missed true events;
      FP = predicted events overlapping NO true event.
    Returns (tp, fn, fp)."""
    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]
    tp = sum(any(overlaps(t, p) for p in pred_events) for t in true_events)
    fn = len(true_events) - tp
    fp = sum(not any(overlaps(p, t) for t in true_events) for p in pred_events)
    return tp, fn, fp


def sens_fa_curve(scores, win_starts, win_dur, true_events, total_hours,
                  thresholds=None, min_event_s=1.0, merge_gap_s=1.0):
    """Sweep thresholds -> (sensitivity, FA-per-hour) operating points, aggregated
    across recordings. Pass per-recording lists (scores/win_starts/true_events are
    lists-of-arrays; total_hours a scalar sum). Returns dict of arrays."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 40)
    recs = list(zip(scores, win_starts, true_events))
    sens, fa = [], []
    for th in thresholds:
        TP = FN = FP = 0
        for sc, ws, te in recs:
            pe = scores_to_events(sc, ws, win_dur, th, min_event_s, merge_gap_s)
            tp, fn, fp = event_confusion(pe, te)
            TP += tp; FN += fn; FP += fp
        sens.append(TP / (TP + FN) if (TP + FN) else np.nan)
        fa.append(FP / total_hours if total_hours else np.nan)
    return {"thresholds": np.asarray(thresholds),
            "sensitivity": np.asarray(sens), "fa_per_h": np.asarray(fa)}


def event_sens_at_fa_auc(curve, fa_lo=0.1, fa_hi=100.0):
    """AUC of sensitivity vs log10(FA/h) over [fa_lo, fa_hi] (NeuroAtlas metric,
    bounded [0,1]). Curve is monotonised (best sensitivity achievable at <= each FA)."""
    fa = curve["fa_per_h"]
    sens = curve["sensitivity"]
    ok = np.isfinite(fa) & np.isfinite(sens) & (fa > 0)
    fa, sens = fa[ok], sens[ok]
    if len(fa) < 2:
        return float("nan")
    order = np.argsort(fa)
    fa, sens = fa[order], sens[order]
    sens = np.maximum.accumulate(sens)          # best sensitivity at <= this FA
    grid = np.logspace(np.log10(fa_lo), np.log10(fa_hi), 200)
    s = np.interp(np.log10(grid), np.log10(fa), sens, left=sens[0], right=sens[-1])
    return float((np.trapezoid if hasattr(np,"trapezoid") else np.trapz)(s, np.log10(grid)) / (np.log10(fa_hi) - np.log10(fa_lo)))


def sensitivity_at(curve, fa_target):
    """Best sensitivity achievable at <= fa_target FA/h."""
    fa, sens = curve["fa_per_h"], curve["sensitivity"]
    m = np.isfinite(fa) & np.isfinite(sens) & (fa <= fa_target)
    return float(np.nanmax(sens[m])) if m.any() else float("nan")


# ---------- classical baseline (reproducible gradient boosting) ----------
def train_gbm(X, y):
    """CatBoost if available (the paper's choice), else sklearn HistGradientBoosting
    — both strong, reproducible gradient-boosting anchors."""
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                               loss_function="Logloss", verbose=False,
                               auto_class_weights="Balanced", random_seed=42)
        m.fit(X, y)
        return m, "catboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                           class_weight="balanced", random_state=42)
        m.fit(X, y)
        return m, "hist-gbm"


def window_features(x, sfreq):
    """Expert per-channel features for one window (C, T) -> 1-D vector: log band-power
    (delta/theta/alpha/beta/gamma) + line-length + variance — a compact seizure-
    relevant set for the gradient-boosting anchor."""
    from scipy.signal import welch
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    f, pxx = welch(x, fs=sfreq, nperseg=min(int(2 * sfreq), x.shape[-1]), axis=-1)
    bp = [np.log(pxx[:, (f >= lo) & (f < hi)].mean(-1) + 1e-30) for lo, hi in bands]
    line_length = np.log(np.abs(np.diff(x, axis=-1)).sum(-1) + 1e-30)
    var = np.log(x.var(-1) + 1e-30)
    return np.concatenate(bp + [line_length, var])


if __name__ == "__main__":
    # tiny self-test of the event scorer (no data): a 100 s recording @ 10 s windows,
    # a true seizure 30-50 s; a probe that fires 32-46 s -> should be 1 TP, 0 FP.
    win_dur, ws = 10.0, np.arange(0, 100, 10.0)
    sc = np.zeros(len(ws)); sc[3:5] = 0.9              # windows starting 30,40 s
    true = [(30.0, 50.0)]
    pe = scores_to_events(sc, ws, win_dur, 0.5)
    tp, fn, fp = event_confusion(pe, true)
    print(f"self-test: pred_events={pe} -> TP={tp} FN={fn} FP={fp}")
    assert (tp, fn, fp) == (1, 0, 0), "event scorer self-test failed"
    print("epilepsy_scorer self-test OK")
