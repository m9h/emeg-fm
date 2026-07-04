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


# ---------- SzCORE reference scorer (EpilepsyBench / Sci-Rep 2026 standard) ----------
# Primary scorer: the validated `timescoring` EventScoring (SzCORE tolerances:
# 30 s pre / 60 s post, 90 s refractory, 5 min max event). Our earlier hand-rolled
# any-overlap scorer stays available as a lightweight fallback + a NEDC cross-check
# is run separately. timescoring lives in the container lib stack (/mnt/t9).
_SZCORE_DEFAULT = dict(toleranceStart=30.0, toleranceEnd=60.0, minOverlap=0.0,
                       maxEventDuration=5 * 60.0, minDurationBetweenEvents=90.0)


def _szcore_imports():
    from timescoring import scoring
    from timescoring.annotations import Annotation
    return scoring, Annotation


def szcore_score_recording(pred_events, true_events, dur_s, fs=10, params=None):
    """SzCORE EventScoring for ONE recording. pred/true events = [(start_s, stop_s)].
    Returns (tp, fp, refTrue, dur_days) for cross-recording aggregation."""
    scoring, Annotation = _szcore_imports()
    N = max(1, int(round(dur_s * fs)))

    def mask(evs):
        m = np.zeros(N, bool)
        for a, b in evs:
            m[max(0, int(round(a * fs))):min(N, int(round(b * fs)))] = True
        return m

    ref = Annotation(mask(true_events), fs)
    hyp = Annotation(mask(pred_events), fs)
    p = scoring.EventScoring.Parameters(**(params or _SZCORE_DEFAULT))
    s = scoring.EventScoring(ref, hyp, p)
    return int(s.tp), int(s.fp), int(s.refTrue), N / fs / 86400.0


def szcore_curve(scores, win_starts, win_dur, true_events, durations_s,
                 thresholds=None, fs=10, params=None, min_event_s=1.0, merge_gap_s=1.0):
    """Sweep thresholds -> (sensitivity, FA-per-DAY) via the SzCORE reference scorer,
    aggregated across recordings (sens = ΣTP/ΣrefTrue, FA/day = ΣFP/Σdays)."""
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.98, 40)
    recs = list(zip(scores, win_starts, true_events, durations_s))
    sens, fa_day, f1, tps, fps, rts = [], [], [], [], [], []
    for th in thresholds:
        TP = FP = RT = 0
        days = 0.0
        for sc, ws, te, dur in recs:
            pe = scores_to_events(sc, ws, win_dur, th, min_event_s, merge_gap_s)
            tp, fp, rt, d = szcore_score_recording(pe, te, dur, fs, params)
            TP += tp; FP += fp; RT += rt; days += d
        se = TP / RT if RT else np.nan
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        sens.append(se); fa_day.append(FP / days if days else np.nan)
        f1.append(2 * prec * se / (prec + se) if (prec + se) and np.isfinite(se) else 0.0)
        tps.append(TP); fps.append(FP); rts.append(RT)
    fa_day = np.asarray(fa_day)
    return {"thresholds": np.asarray(thresholds), "sensitivity": np.asarray(sens),
            "fa_per_day": fa_day, "fa_per_h": fa_day / 24.0, "f1": np.asarray(f1),
            "tp": np.asarray(tps), "fp": np.asarray(fps), "ref_true": np.asarray(rts)}


def szcore_operating_point(dev_curve, eval_curve):
    """Pick the F1-optimal threshold on DEV, report EVAL metrics at that threshold
    (SzCORE/EpilepsyBench convention: tune operating point on dev, apply to eval)."""
    i = int(np.nanargmax(dev_curve["f1"]))
    th = float(dev_curve["thresholds"][i])
    j = int(np.argmin(np.abs(eval_curve["thresholds"] - th)))
    return dict(threshold=th,
                eval_sensitivity=float(eval_curve["sensitivity"][j]),
                eval_fa_per_day=float(eval_curve["fa_per_day"][j]),
                eval_f1=float(eval_curve["f1"][j]),
                dev_f1=float(dev_curve["f1"][i]))


def sensitivity_at_fa_day(curve, fa_day_target):
    """Best sensitivity achievable at <= fa_day_target FP/24h (SzCORE operating point)."""
    fa, sens = curve["fa_per_day"], curve["sensitivity"]
    m = np.isfinite(fa) & np.isfinite(sens) & (fa <= fa_day_target)
    return float(np.nanmax(sens[m])) if m.any() else 0.0


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


def event_sens_at_fa_auc(curve, fa_lo=0.1, fa_hi=100.0, fa_key="fa_per_h"):
    """AUC of sensitivity vs log10(FA) over [fa_lo, fa_hi] (NeuroAtlas metric, [0,1]).
    Monotonised (best sensitivity achievable at <= each FA). ``fa_key`` selects the
    FA axis ('fa_per_h' NeuroAtlas, or 'fa_per_day' SzCORE).

    IMPORTANT: at FA targets STRICTER than the minimum achievable FA (the classifier
    can't operate that quietly), sensitivity is 0 — you cannot detect at that budget.
    Extrapolating the max sensitivity down (the old ``left=sens[0]``) let a degenerate
    all-alarm classifier — which only operates at huge FA — score a perfect 1.0."""
    fa = np.asarray(curve[fa_key], float)
    sens = np.asarray(curve["sensitivity"], float)
    ok = np.isfinite(fa) & np.isfinite(sens)
    fa, sens = fa[ok], sens[ok]
    if len(fa) < 1:
        return float("nan")
    # FA==0 (a no-false-alarm operating point) is achievable at ANY budget -> anchor it
    # just left of the grid so its sensitivity carries across the whole range.
    fa = np.maximum(fa, fa_lo * 0.1)
    order = np.argsort(fa)
    fa, sens = fa[order], sens[order]
    sens = np.maximum.accumulate(sens)          # best sensitivity at <= this FA
    grid = np.logspace(np.log10(fa_lo), np.log10(fa_hi), 200)
    # STEP interpolation (right-continuous), NOT linear: sensitivity at a FA budget g
    # is the best sensitivity ACHIEVABLE at some operating point with fa' <= g, else 0.
    # Linear np.interp would draw a line across empty FA gaps and FABRICATE sensitivity
    # where no operating point exists (e.g. a bimodal degenerate probe whose points are
    # only at fa~0/sens0 and fa~250/sens1 -> linear interp invents ~0.5 at fa=10).
    idx = np.searchsorted(fa, grid, side="right") - 1
    s = np.where(idx >= 0, sens[np.clip(idx, 0, len(sens) - 1)], 0.0)
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
