"""Tests for the Luck-Lab ERPLAB-parity linear-SVM decode probe.

The contract mirrors ERPLAB12 ``pop_decoding_regularization`` (Steve Luck lab,
OSF un6pq): linear SVM, pseudo-ERP crossblock CV (default 3 folds), nIter
resampling, EqualizeTrials='classes', and a regularization (Gamma→C) grid. We
verify the protocol's *decision behaviour* on synthetic data — separable
classes decode near 1.0, identical distributions sit at chance — plus the
shape/finiteness invariants the driver relies on.
"""
from __future__ import annotations

import numpy as np
import pytest

from fmscope.training.svm_probe import (
    DEFAULT_GAMMA_GRID,
    erplab_decode_scalp,
    luck_svm_decode,
    luck_svm_grid,
)


def _two_class(n_per=60, n_feat=8, sep=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n_per, n_feat))
    X1 = rng.standard_normal((n_per, n_feat)) + sep
    X = np.concatenate([X0, X1], 0)
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)
    return X, y


def test_separable_classes_decode_high():
    X, y = _two_class(sep=3.0)
    acc = luck_svm_decode(X, y, n_iter=20, crossfold=3, seed=1)
    assert acc > 0.85


def test_identical_distributions_at_chance():
    X, y = _two_class(sep=0.0)
    acc = luck_svm_decode(X, y, n_iter=40, crossfold=3, seed=1)
    assert 0.35 < acc < 0.65


def test_accuracy_is_finite_and_bounded():
    X, y = _two_class(sep=1.0)
    acc = luck_svm_decode(X, y, n_iter=10, crossfold=3, seed=2)
    assert np.isfinite(acc)
    assert 0.0 <= acc <= 1.0


def test_equalize_handles_class_imbalance():
    # 80 vs 20 trials — EqualizeTrials must subsample, not crash.
    X0, y0 = _two_class(n_per=80, sep=2.5)
    Xa = X0[y0 == 0]
    Xb = X0[y0 == 1][:20]
    X = np.concatenate([Xa, Xb], 0)
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))]).astype(int)
    acc = luck_svm_decode(X, y, n_iter=15, crossfold=3, seed=3, equalize=True)
    assert acc > 0.7


def test_grid_returns_curve_and_best():
    X, y = _two_class(sep=2.0)
    res = luck_svm_grid(X, y, n_iter=10, crossfold=3, seed=4)
    assert set(res["per_C"]) == set(DEFAULT_GAMMA_GRID)
    assert all(np.isfinite(v) for v in res["per_C"].values())
    assert res["best_C"] in DEFAULT_GAMMA_GRID
    assert res["best_acc"] == max(res["per_C"].values())


def test_too_few_trials_returns_nan():
    # Fewer trials per class than crossfold blocks → undecodable, not a crash.
    X, y = _two_class(n_per=2, sep=3.0)
    acc = luck_svm_decode(X, y, n_iter=10, crossfold=3, seed=5)
    assert np.isnan(acc)


def test_grouped_cross_subject_decode():
    # 9 subjects, separable classes with a per-subject (identity) offset added
    # to both classes. Cross-subject grouped decode must still recover the
    # class axis despite the subject shifts.
    rng = np.random.default_rng(11)
    n_subj, n_per, n_feat = 9, 20, 8
    Xs, ys, gs = [], [], []
    class_axis = rng.standard_normal(n_feat)
    for s in range(n_subj):
        subj_shift = rng.standard_normal(n_feat) * 2.0
        for c in (0, 1):
            base = rng.standard_normal((n_per, n_feat)) + subj_shift
            base += (c * 2.0) * class_axis
            Xs.append(base)
            ys.append(np.full(n_per, c))
            gs.append(np.full(n_per, s))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    g = np.concatenate(gs, 0)
    acc = luck_svm_decode(X, y, groups=g, n_iter=20, crossfold=3, seed=12)
    assert np.isfinite(acc)
    assert acc > 0.7


def test_grouped_too_few_groups_returns_nan():
    X, y = _two_class(n_per=30, sep=3.0)
    g = np.zeros(len(y), dtype=int)  # one group < crossfold
    acc = luck_svm_decode(X, y, groups=g, n_iter=5, crossfold=3, seed=1)
    assert np.isnan(acc)


def test_scalp_decode_window_summary():
    # (n_trials, n_chan, n_times): inject a class difference only inside the
    # measurement window so windowed decoding must exceed chance there.
    rng = np.random.default_rng(7)
    n, C, T = 80, 6, 100
    times = np.linspace(-0.2, 0.8, T)
    X = rng.standard_normal((n, C, T))
    y = np.array([0, 1] * (n // 2))
    win = (times >= 0.3) & (times <= 0.5)
    X[y == 1][:, :, win] += 2.0  # in-place via fancy index won't stick; redo
    # Apply additive bump to class-1 trials within the window explicitly.
    idx1 = np.where(y == 1)[0]
    X[np.ix_(idx1, np.arange(C), np.where(win)[0])] += 2.0

    out = erplab_decode_scalp(
        X, y, times, window=(0.3, 0.5), decode_every=5,
        n_iter=10, crossfold=3, seed=8,
    )
    assert np.isfinite(out["window_acc"])
    assert out["window_acc"] > 0.7
    assert out["curve"].shape[0] == out["curve_times"].shape[0]
    assert np.all(np.isfinite(out["curve"]))
