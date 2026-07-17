"""LoRA-REVE CHB-MIT pilot: pure-math LoRA delta + bounded-recording selection.
scripts/eegfm_t9.sh python -m pytest tests/test_lora_reve_chbmit_pilot.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lora_reve_chbmit_pilot as lp  # noqa: E402


def test_lora_delta_matches_low_rank_matrix_multiply():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6))
    A = rng.standard_normal((2, 6))   # rank=2, d_in=6
    B = rng.standard_normal((3, 2))   # d_out=3, rank=2
    scaling = 16.0 / 2
    got = lp._lora_delta(x, A, B, scaling)
    expected = (x @ A.T @ B.T) * scaling
    assert got.shape == (4, 3)
    np.testing.assert_allclose(got, expected)


def test_lora_delta_zero_when_B_is_zero():
    # mirrors LoRALinear's init: B starts at zero -> delta is zero at init
    x = np.random.default_rng(1).standard_normal((3, 5))
    A = np.random.default_rng(2).standard_normal((4, 5))
    B = np.zeros((7, 4))
    got = lp._lora_delta(x, A, B, scaling=2.0)
    np.testing.assert_allclose(got, np.zeros((3, 7)))


def test_iter_bounded_recordings_includes_all_seizure_and_capped_clean(monkeypatch):
    fake_recs = [
        ("a.edf", [(10.0, 20.0)], "chb01"),
        ("b.edf", [], "chb01"),
        ("c.edf", [], "chb01"),
        ("d.edf", [], "chb01"),
        ("e.edf", [(5.0, 8.0)], "chb02"),
        ("f.edf", [], "chb02"),
    ]
    monkeypatch.setattr(lp.cb, "iter_recordings", lambda: fake_recs)
    out = lp.iter_bounded_recordings({"chb01", "chb02"}, max_nonseizure_per_patient=1)
    names = sorted(edf for edf, _, _ in out)
    # chb01: seizure "a" + 1 clean (first of b/c/d) = 2; chb02: seizure "e" + clean "f" = 2
    assert names == ["a.edf", "b.edf", "e.edf", "f.edf"]


def test_iter_bounded_recordings_merges_chb21_into_chb01(monkeypatch):
    fake_recs = [
        ("a.edf", [(1.0, 2.0)], "chb01"),
        ("b.edf", [(3.0, 4.0)], "chb21"),
    ]
    monkeypatch.setattr(lp.cb, "iter_recordings", lambda: fake_recs)
    out = lp.iter_bounded_recordings({"chb01"}, max_nonseizure_per_patient=1)
    assert len(out) == 2  # both attributed to the merged chb01 patient
