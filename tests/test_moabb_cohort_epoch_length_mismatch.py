"""MOABB cohort builder: per-subject epoch-length mismatch fallback.

Excluding subject 88 (test_moabb_cohort_incompatible_subjects.py) was not
sufficient -- the SAME top-level error (operands could not be broadcast
together with shapes (602,) (601,)) recurred on the full ~108-subject
PhysionetMotorImagery run, this time from ``mne.concatenate_epochs``
comparing ``epochs.times`` across TWO NORMAL (160 Hz) subjects whose actual
per-trial epoch length differs by one sample -- real per-recording timing
jitter in event-marker-based epoching, unrelated to the documented 128 Hz
outlier. ``_get_data_per_subject`` (the existing nchan-mismatch fallback)
needs to also crop each subject's epochs to a common minimum length before
concatenating, and the except-clause that routes into it needs to recognize
this failure mode too (not just "nchan" errors).

scripts/eegfm_t9.sh python -m pytest tests/test_moabb_cohort_epoch_length_mismatch.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import emeg_fm.moabb_cohort as mc  # noqa: E402


def test_is_epoch_concat_mismatch_recognizes_nchan_errors():
    assert mc._is_epoch_concat_mismatch(ValueError("nchan mismatch: 22 vs 20"))


def test_is_epoch_concat_mismatch_recognizes_broadcast_errors():
    assert mc._is_epoch_concat_mismatch(
        ValueError("operands could not be broadcast together with shapes (602,) (601,)"))


def test_is_epoch_concat_mismatch_false_for_unrelated_errors():
    assert not mc._is_epoch_concat_mismatch(ValueError("something else entirely"))


def test_crop_to_common_length_trims_to_shortest():
    arrays = [
        np.zeros((5, 3, 602)),
        np.zeros((4, 3, 601)),
        np.zeros((6, 3, 602)),
    ]
    out = mc._crop_to_common_length(arrays)
    assert all(a.shape[-1] == 601 for a in out)
    assert [a.shape[0] for a in out] == [5, 4, 6]  # trial counts untouched


def test_crop_to_common_length_is_noop_when_already_uniform():
    arrays = [np.zeros((3, 2, 100)), np.zeros((7, 2, 100))]
    out = mc._crop_to_common_length(arrays)
    assert out[0].shape == (3, 2, 100)
    assert out[1].shape == (7, 2, 100)
