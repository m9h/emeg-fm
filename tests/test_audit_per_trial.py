"""Tests for the per-trial (de-pooled) subject-axis erasure mode in audit_cell.

The MOABB / ERP CORE identity-trap leaderboards measured the trap with
``audit_cell`` → ``subject_axis_erasure`` at the *recording* level, where the
cohort makes each ``(subject, class)`` one recording. Pooling all of a subject's
class-trials into one prediction crushes within-class variance and fabricates an
"identity-free lift" on high-dim FM features. ``AuditConfig.erasure_per_trial``
switches the erasure to per-trial decoding (each window its own recording,
grouped by subject) — ``n ≫ p``, honest. These tests pin the grouping helper and
(when torch is available) that the flag actually de-pools inside ``audit_cell``.
"""
from __future__ import annotations

import numpy as np
import pytest

from fmscope.verdict.audit import _per_trial_grouping


def test_per_trial_grouping_one_recording_per_window():
    sids = np.array([0, 0, 1, 1, 2, 2])
    labels = np.array([0, 1, 0, 1, 0, 1])
    window_recording, rec_labels, rec_pids = _per_trial_grouping(sids, labels)
    # Each window is its own recording, in order.
    assert np.array_equal(window_recording, np.arange(6))
    # Labels/pids pass through unchanged, grouped (for CV) by subject.
    assert np.array_equal(rec_labels, labels)
    assert np.array_equal(rec_pids, sids)


def test_per_trial_grouping_lengths_match_any_order():
    sids = np.array([3, 1, 3, 1, 2, 2, 2])
    labels = np.array([1, 0, 0, 1, 1, 0, 0])
    wr, rl, rp = _per_trial_grouping(sids, labels)
    assert len(wr) == len(rl) == len(rp) == len(labels)
    assert len(np.unique(wr)) == len(labels)  # all distinct recordings


def test_audit_cell_per_trial_depools_the_erasure():
    """The flag must change the erasure from saturating-pooled to honest per-trial.

    Builds a tiny cohort with a strong subject confound and a weak class signal —
    the regime where pooled erasure saturates. With ``erasure_per_trial=True`` the
    erased BA must come back well below the pooled ceiling.
    """
    torch = pytest.importorskip("torch")
    from fmscope.verdict import audit_cell, AuditConfig
    from fmscope.data.adapters import InMemoryCohort

    rng = np.random.default_rng(0)
    dim, n_subj, n_per_class = 64, 8, 60
    w = rng.standard_normal(dim)
    w /= np.linalg.norm(w)
    recordings = []
    for s in range(n_subj):
        off = rng.standard_normal(dim) * 6.0  # large per-subject identity offset
        for cls in (0, 1):
            sig = (1.0 if cls else -1.0) * w  # weak shared class signal
            X = (rng.standard_normal((n_per_class, dim)) + off + sig).astype(np.float32)
            recordings.append((s, cls, X[:, None, :]))  # (n, 1, dim) "windows"
    cohort = InMemoryCohort(recordings, ch_names=[f"c{i}" for i in range(1)])

    # Identity extractor: collapse the (1, dim) window to its dim-vector.
    class _IdentityExtractor(torch.nn.Module):
        def forward(self, x):  # x: (B, 1, dim)
            return x.reshape(x.shape[0], -1)

    ext = _IdentityExtractor()
    common = dict(cell_name="synthetic", cell_layout="W,C", batch_size=64,
                  device="cpu", n_null_seeds=2)
    pooled = audit_cell(cohort, ext, config=AuditConfig(**common,
                        erasure_per_trial=False))
    per_trial = audit_cell(cohort, ext, config=AuditConfig(**common,
                           erasure_per_trial=True))

    pooled_erased = pooled["erasure_label_ba_erased"]
    per_trial_erased = per_trial["erasure_label_ba_erased"]
    # Pooling saturates; per-trial must be markedly lower (de-pooled).
    assert pooled_erased > 0.9
    assert per_trial_erased < 0.85
    assert pooled_erased - per_trial_erased > 0.1
