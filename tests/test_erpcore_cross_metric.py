"""Tests for the *meaningful* cross-subject ERP CORE identity-trap metric.

The first cut of ``scripts/erpcore_luck_parity.py`` measured the cross-subject
identity trap with a pseudo-ERP leave-subject-block-out SVM. On frozen-FM
embeddings that metric saturates to ~1.0 after subject-axis erasure: only
``crossfold`` super-averaged points per class (each averaging many subjects ×
hundreds of trials → variance ≈ 0) are decoded in a >>-dimensional embedding
(p ≫ n ⇒ always linearly separable), and the only thing keeping the *raw*
column below 1.0 is the between-subject offset that LEACE removes by design.

The fix is to measure the trap with the canonical single-trial decode
(:func:`fmscope.diagnostics.erasure.subject_axis_erasure`): each trial is its
own recording, ``StratifiedGroupKFold`` grouped by subject (train/test never
share a subject), balanced accuracy. ``n ≫ p`` ⇒ the score reflects real
generalization, not separability, and the result carries ``degenerate`` /
``interpretable`` flags so a saturated/degenerate case is *flagged* rather than
reported as a triumphant 1.0.

These tests pin: (1) the canonical metric stays bounded below saturation on
data where the pseudo-ERP metric blows up; (2) erasure does not *hurt* the
label decode (the trap lift is ≥ ~0); (3) the degeneracy flag fires when the
subject subspace fills the ambient feature space.
"""
from __future__ import annotations

import numpy as np
import pytest

from fmscope.diagnostics.erasure import subject_axis_erasure


def _multi_subject(n_subj=14, n_per_class=50, dim=48,
                   subj_scale=6.0, class_scale=1.0, seed=0):
    """Synthetic frozen-FM-like features with a strong subject confound.

    Each subject gets a large random identity offset (``subj_scale``); a single
    shared direction carries a modest class signal (``class_scale``). Single
    trials are noisy (unit Gaussian), so a *single-trial* cross-subject decode
    is bounded well below 1.0 — but block-averaged pseudo-ERPs of the same data
    are near-noiseless and trivially separable.

    Returns ``(feats, y, subj)``; labels interleave within subject, as in ERP
    CORE (target/non-target trials are not contiguous).
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(dim)
    w /= np.linalg.norm(w)
    feats, y, subj = [], [], []
    for s in range(n_subj):
        off = rng.standard_normal(dim) * subj_scale
        for cls in (0, 1):
            base = rng.standard_normal((n_per_class, dim))
            signal = (class_scale if cls == 1 else -class_scale) * w
            feats.append(base + off + signal)
            y += [cls] * n_per_class
            subj += [s] * n_per_class
    feats = np.concatenate(feats, 0)
    y = np.asarray(y)
    subj = np.asarray(subj)
    # Interleave trials within each subject so labels are not contiguous.
    order = rng.permutation(len(y))
    return feats[order], y[order], subj[order]


def _erasure(feats, y, subj, **kw):
    """Drive subject_axis_erasure the way the parity driver does: one
    recording per trial, grouped by subject."""
    n = len(y)
    return subject_axis_erasure(
        feats, subj, y,
        window_recording=np.arange(n),
        rec_labels=y,
        rec_pids=subj,
        **kw,
    )


def test_cross_subject_decode_is_bounded_not_saturated():
    feats, y, subj = _multi_subject(seed=1)
    er = _erasure(feats, y, subj)
    # Single-trial cross-subject BA must be a real, sub-saturation number on
    # both sides of erasure — never the ~1.0 the pseudo-ERP metric produced.
    assert np.isfinite(er.label_ba_raw) and np.isfinite(er.label_ba_erased)
    assert 0.5 <= er.label_ba_raw < 0.97
    assert 0.5 <= er.label_ba_erased < 0.97
    assert not er.degenerate


def test_identity_trap_lift_non_negative_and_interpretable():
    # A genuinely decodable cross-subject signal under a subject confound: raw
    # clears the gate, erasing the subject axis lifts (does not hurt) the decode.
    feats, y, subj = _multi_subject(subj_scale=4.0, class_scale=2.0, seed=2)
    er = _erasure(feats, y, subj)
    assert er.interpretable
    assert er.label_ba_delta >= -0.05
    # The linear subject axis is genuinely there pre-erasure and collapses to
    # ~chance after — i.e. the metric is measuring an actual identity subspace.
    assert er.subj_ba_linear_pre > er.subj_ba_linear_post


def test_single_trial_metric_does_not_saturate_where_pseudo_erp_does():
    """Regression for the bug this fix addresses.

    On the same strong-confound / weak-signal data, the abandoned pseudo-ERP
    leave-subject-block-out decode saturates after erasure (≈1.0) because it
    decodes a handful of near-noiseless block averages in a high-dim space; the
    canonical single-trial group-CV decode stays a real, sub-saturation number.
    """
    from fmscope.diagnostics.erasure import (
        apply_eraser, subject_eraser, whiten,
    )
    from fmscope.training.svm_probe import luck_svm_decode

    feats, y, subj = _multi_subject(seed=1)  # subj_scale=6, class_scale=1
    mu, Xc, W, W_plus, _ = whiten(feats, shrinkage=True)
    _, P_perp, _ = subject_eraser(Xc, W, W_plus, subj)
    feats_free = apply_eraser(feats, mu, P_perp)

    grid = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    pseudo_erp_erased = max(
        luck_svm_decode(feats_free, y, groups=subj, crossfold=3, n_iter=50,
                        C=C, seed=0)
        for C in grid
    )
    single_trial_erased = _erasure(feats, y, subj).label_ba_erased

    assert pseudo_erp_erased > 0.95          # the saturation we're fixing
    assert single_trial_erased < 0.90        # the meaningful replacement
    assert pseudo_erp_erased - single_trial_erased > 0.10


def test_degenerate_flag_fires_when_subspace_fills_ambient():
    # More subjects than ambient dims ⇒ the subject-mean subspace fills the
    # feature space and erasure is degenerate; the flag must catch it.
    feats, y, subj = _multi_subject(n_subj=20, dim=8, seed=3)
    er = _erasure(feats, y, subj)
    assert er.degenerate


def test_chance_and_shapes_reported():
    feats, y, subj = _multi_subject(n_subj=10, seed=4)
    er = _erasure(feats, y, subj)
    assert er.n_subjects == 10
    assert er.chance == pytest.approx(0.1)
    assert er.rank_subject_axis >= 1
    assert er.embed_dim == feats.shape[1]
