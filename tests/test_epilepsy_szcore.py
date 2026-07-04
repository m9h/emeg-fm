"""SzCORE scorer integration + the all-alarm AUC-degeneracy fix.
Run in the 26.06 container (timescoring lives in /mnt/t9): scripts/eegfm_t9.sh
python -m pytest tests/test_epilepsy_szcore.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import epilepsy_scorer as es  # noqa: E402


def _synthetic(n_rec=8, dur_s=600.0, win=10.0, seed=0):
    """n_rec recordings, one 40 s seizure each. Return per-rec (starts, true_events, dur)."""
    rng = np.random.default_rng(seed)
    starts, trues, durs = [], [], []
    for _ in range(n_rec):
        nwin = int(dur_s // win)
        st = np.arange(nwin) * win
        on = rng.uniform(60, dur_s - 100)
        starts.append(st); trues.append([(on, on + 40.0)]); durs.append(dur_s)
    return starts, trues, durs


def _scores(starts, trues, win, kind):
    """Per-rec window scores. kind: 'perfect' (high on seizure), 'all_alarm' (all high)."""
    out = []
    for st, te in zip(starts, trues):
        s = np.full(len(st), 0.02)
        if kind == "all_alarm":
            s[:] = 0.99
        else:  # perfect: high only on the seizure windows
            for a, b in te:
                s[(st + win > a) & (st < b)] = 0.99
        out.append(s)
    return out


def test_szcore_score_recording_detects_overlap():
    # pred overlapping the true seizure -> tp=1, fp=0
    tp, fp, rt, days = es.szcore_score_recording([(100, 140)], [(105, 135)], 600.0)
    assert tp == 1 and fp == 0 and rt == 1
    assert days == 600.0 / 86400.0


def test_szcore_curve_perfect_beats_allalarm():
    starts, trues, durs = _synthetic()
    win = 10.0
    perfect = es.szcore_curve(_scores(starts, trues, win, "perfect"),
                              starts, win, trues, durs)
    allalarm = es.szcore_curve(_scores(starts, trues, win, "all_alarm"),
                               starts, win, trues, durs)
    auc_perfect = es.event_sens_at_fa_auc(perfect, 0.1, 100.0, fa_key="fa_per_day")
    auc_allalarm = es.event_sens_at_fa_auc(allalarm, 0.1, 100.0, fa_key="fa_per_day")
    # THE FIX: an all-alarm classifier can only operate at a huge FA/day, so within a
    # sane [0.1,100] FP/day budget its usable sensitivity is ~0 -> low AUC, NOT 1.0.
    assert auc_allalarm < 0.2, f"all-alarm should score ~0, got {auc_allalarm:.3f}"
    assert auc_perfect > auc_allalarm + 0.3, (auc_perfect, auc_allalarm)


def test_bimodal_degenerate_auc_is_zero():
    # A collapsed probe whose curve has ONLY (fa~0, sens0) and (fa~big, sens1) points
    # and nothing between — the REVE identity-free failure mode. The Event-Sens@FA AUC
    # must be ~0 (no operating point achieves usable sensitivity at a clinical FA budget),
    # NOT the ~0.5 that linear interpolation across the empty gap would fabricate.
    curve = {"fa_per_day": np.array([0.0, 0.0, 300.0, 300.0]),
             "sensitivity": np.array([0.0, 0.0, 1.0, 1.0])}
    auc = es.event_sens_at_fa_auc(curve, 0.1, 10.0, fa_key="fa_per_day")
    assert auc < 0.05, f"bimodal degenerate should score ~0, got {auc:.3f}"
    assert es.sensitivity_at_fa_day(curve, 10.0) == 0.0


def test_allalarm_fa_per_day_is_huge():
    starts, trues, durs = _synthetic()
    c = es.szcore_curve(_scores(starts, trues, 10.0, "all_alarm"), starts, 10.0, trues, durs)
    # all-alarm at every threshold -> FP/day far above any clinical budget
    assert np.nanmin(c["fa_per_day"]) > 100.0
    # and sensitivity at a 1/day budget is 0 (can't operate that quietly)
    assert es.sensitivity_at_fa_day(c, 1.0) == 0.0
