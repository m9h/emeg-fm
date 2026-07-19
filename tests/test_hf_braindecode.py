"""Shared braindecode-HF-hub loader: subject grouping, deterministic splits.
scripts/eegfm_t9.sh python -m pytest tests/test_hf_braindecode.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_hf_braindecode as hb  # noqa: E402


def test_group_by_subject_concatenates_multiple_recordings_same_subject():
    w1 = [(np.zeros((2, 3)), 0), (np.ones((2, 3)), 1)]
    w2 = [(np.full((2, 3), 5.0), 0)]
    per_recording = [("sub-01", w1), ("sub-01", w2), ("sub-02", [(np.zeros((2, 3)), 1)])]
    grouped = hb.group_by_subject(per_recording)
    assert set(grouped.keys()) == {"sub-01", "sub-02"}
    X1, y1 = grouped["sub-01"]
    assert X1.shape == (3, 2, 3)
    assert list(y1) == [0, 1, 0]
    X2, y2 = grouped["sub-02"]
    assert X2.shape == (1, 2, 3)
    assert list(y2) == [1]


def test_group_by_subject_skips_empty_recordings():
    per_recording = [("sub-01", []), ("sub-02", [(np.zeros((1, 1)), 0)])]
    grouped = hb.group_by_subject(per_recording)
    assert "sub-01" not in grouped
    assert "sub-02" in grouped


def test_subject_splits_deterministic_and_disjoint():
    subs = {f"sub-{i:02d}" for i in range(20)}
    tr, dv, ev = hb.subject_splits(subs)
    assert tr | dv | ev == subs
    assert not (tr & dv) and not (dv & ev) and not (tr & ev)
    assert len(tr) > len(dv) and len(tr) > len(ev)


def test_stratified_subject_splits_keeps_both_classes_in_eval():
    # mirrors MDD's real bug: subject id prefix correlates with label, so a
    # naive lexicographic sort-and-slice puts the whole "MDDS*" class in eval
    subject_to_label = {f"HS{i}": 0 for i in range(1, 21)}
    subject_to_label.update({f"MDDS{i}": 1 for i in range(1, 21)})
    tr, dv, ev = hb.stratified_subject_splits(subject_to_label)
    ev_labels = {subject_to_label[s] for s in ev}
    assert ev_labels == {0, 1}  # both classes present, not degenerate single-class
    dv_labels = {subject_to_label[s] for s in dv}
    assert dv_labels == {0, 1}
    assert tr | dv | ev == set(subject_to_label)
    assert not (tr & dv) and not (dv & ev) and not (tr & ev)
