"""T3A test-time classifier adjustment: sanity checks on toy separable data
and the binary-classifier virtual-2-class construction.
scripts/eegfm_t9.sh python -m pytest tests/test_tta_t3a.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import tta_t3a as t3a  # noqa: E402


def _make_separable(n_classes, n_per_class=30, d=8, seed=0, sep=6.0, centers=None):
    rng = np.random.RandomState(seed)
    if centers is None:
        centers = rng.randn(n_classes, d) * sep
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.randn(n_per_class, d) * 0.5)
        y.append(np.full(n_per_class, c))
    X = np.concatenate(X)
    y = np.concatenate(y)
    idx = rng.permutation(len(y))
    return X[idx], y[idx], centers


def test_t3a_matches_baseline_accuracy_on_easy_separable_multiclass():
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, centers = _make_separable(3, seed=1)
    Xte, yte, _ = _make_separable(3, seed=2, centers=centers)
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    base_acc = (clf.predict(Xte) == yte).mean()
    yhat = t3a.t3a_adapt_sequence(clf, Xte, filter_k=20)
    t3a_acc = (yhat == yte).mean()
    # on a trivially separable problem both should be near-perfect
    assert base_acc > 0.95
    assert t3a_acc > 0.90


def test_t3a_binary_virtual_2class_construction():
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, _ = _make_separable(2, seed=3)
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    assert clf.coef_.shape[0] == 1
    W, b = t3a._binary_to_virtual_2class(clf.coef_, clf.intercept_)
    assert W.shape == (2, Xtr.shape[1])
    assert b.shape == (2,)
    # the two virtual class weight vectors should be exact opposites (by construction)
    np.testing.assert_allclose(W[0], -W[1])


def test_t3a_predictions_are_causal_not_using_future_samples():
    """Changing sample i+1..n must not change the prediction for sample i."""
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, centers = _make_separable(2, seed=4)
    Xte, _, _ = _make_separable(2, seed=5, n_per_class=10, centers=centers)
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    yhat_full = t3a.t3a_adapt_sequence(clf, Xte, filter_k=5)
    yhat_prefix = t3a.t3a_adapt_sequence(clf, Xte[:5], filter_k=5)
    assert list(yhat_full[:5]) == list(yhat_prefix)


def test_t3a_filter_k_minus_one_keeps_all_supports_growing():
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, centers = _make_separable(2, seed=6)
    Xte, yte, _ = _make_separable(2, seed=7, n_per_class=15, centers=centers)
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    yhat = t3a.t3a_adapt_sequence(clf, Xte, filter_k=-1)
    assert len(yhat) == len(yte)
