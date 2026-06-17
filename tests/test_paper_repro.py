"""Pure-logic tests for the FMScope paper from-raw reproduction (task #81).

The risky, silent-bug-prone parts of the reproduction are (1) mapping eegbci
annotations to the MI-vs-rest contrast and (2) ordering windows so that
``subject_axis_erasure``'s default ``_segment_recordings`` pools exactly one
recording per ``(subject, condition)`` — matching the paper's grouping. Both are
pure functions in ``emeg_fm.paper_repro`` and tested here on CPU; the download /
ICA / REVE-extraction glue is orchestrated by the script and run in-container.
"""
from __future__ import annotations

import numpy as np
import pytest

from emeg_fm.paper_repro import (
    ds002893_tone_label,
    ds004148_task_label,
    eegbci_mi_vs_rest_label,
    order_for_pooled_grouping,
)


def test_eegbci_label_mapping():
    assert eegbci_mi_vs_rest_label("T0") == 0      # interleaved rest
    assert eegbci_mi_vs_rest_label("T1") == 1      # fist motor imagery
    assert eegbci_mi_vs_rest_label("T2") == 1      # fist motor imagery
    assert eegbci_mi_vs_rest_label("rest") is None  # unknown -> dropped


def test_ds002893_attended_auditory_target_vs_standard():
    base = dict(focus_modality="auditory", attention_status="attended",
                event_type="high_tone", task_role="infrequent_stimulus")
    assert ds002893_tone_label(base) == 1                                   # target
    assert ds002893_tone_label({**base, "event_type": "low_tone",
                                "task_role": "frequent_stimulus"}) == 0     # standard
    # unattended / visual / non-tone roles are dropped (None), not mislabelled
    assert ds002893_tone_label({**base, "attention_status": "unattended"}) is None
    assert ds002893_tone_label({**base, "focus_modality": "visual"}) is None
    assert ds002893_tone_label({**base, "event_type": "button_press",
                                "task_role": "target_detected"}) is None


def test_ds004148_task_label():
    assert ds004148_task_label("eyesclosed") == 0
    assert ds004148_task_label("mathematic") == 1
    assert ds004148_task_label("eyesopen") is None      # other tasks dropped
    assert ds004148_task_label("memory") is None


def test_order_is_a_permutation():
    subj = np.array([2, 2, 1, 1, 3])
    label = np.array([0, 1, 1, 0, 0])
    perm = order_for_pooled_grouping(subj, label)
    assert sorted(perm.tolist()) == list(range(5))


def test_order_makes_subject_condition_blocks_contiguous():
    # Interleaved subjects and labels (as trials actually arrive).
    subj = np.array([1, 0, 1, 0, 1, 0])
    label = np.array([1, 0, 0, 1, 0, 1])
    perm = order_for_pooled_grouping(subj, label)
    s, l = subj[perm], label[perm]
    # subjects non-decreasing; labels non-decreasing within a subject
    assert np.all(np.diff(s) >= 0)
    for sid in np.unique(s):
        assert np.all(np.diff(l[s == sid]) >= 0)
    # every (subject,label) is a single contiguous run -> one pooled recording
    keys = list(zip(s.tolist(), l.tolist()))
    seen, prev = set(), None
    for k in keys:
        if k != prev:
            assert k not in seen, f"{k} is not contiguous"
            seen.add(k)
            prev = k
