"""Tests for emeg_fm.alljoined — pure data-prep helpers for the
Alljoined-1.6M EEG→image smoke.

The dataset ships per-subject epochs as a *pickled dict* npy
(`preprocessed_eeg_data` (n,32,250) @ 250 Hz + `ch_names` + `configs`).
These helpers turn that into REVE-ready batches: load the dict, average
repeated trials per stimulus image, and resample/normalize into REVE's
200 Hz, z-scored, clamped input distribution. All pure-numpy so they can
be unit-tested without torch or the foundation model.
"""

import numpy as np
import pytest

from emeg_fm.alljoined import (
    load_subject_npy,
    average_by_image,
    preprocess_for_reve,
)


def _write_fake_npy(path, n=6, c=32, t=250, sfreq=250):
    rng = np.random.RandomState(0)
    d = {
        "preprocessed_eeg_data": rng.randn(n, c, t).astype(np.float64),
        "ch_names": [f"E{i}" for i in range(c)],
        "configs": {"sfreq": sfreq, "tmin": -0.2, "tmax": 1.0},
        "times": np.linspace(-0.2, 1.0, t),
    }
    np.save(path, d, allow_pickle=True)


# --- load_subject_npy --------------------------------------------------------

def test_load_subject_npy_unwraps_pickled_dict(tmp_path):
    p = tmp_path / "sub.npy"
    _write_fake_npy(p, n=5, c=32, t=250, sfreq=250)
    rec = load_subject_npy(str(p))
    assert rec["eeg"].shape == (5, 32, 250)
    assert len(rec["ch_names"]) == 32
    assert rec["sfreq"] == 250


# --- average_by_image --------------------------------------------------------

def test_average_by_image_groups_and_means():
    # Two 'a' trials, two 'b', one 'c' — distinct constant payloads + noise.
    base = {"a": 1.0, "b": 2.0, "c": 3.0}
    ids = ["b", "a", "b", "a", "c"]
    eeg = np.stack([np.full((4, 10), base[k]) for k in ids]).astype(np.float64)
    avg, uids, counts = average_by_image(eeg, ids)
    assert list(uids) == ["a", "b", "c"]        # sorted unique order
    assert list(counts) == [2, 2, 1]
    assert avg.shape == (3, 4, 10)
    # Constant payloads → average equals the per-image constant.
    np.testing.assert_allclose(avg[0], 1.0)
    np.testing.assert_allclose(avg[1], 2.0)
    np.testing.assert_allclose(avg[2], 3.0)


def test_average_by_image_actually_averages_noise():
    rng = np.random.RandomState(1)
    # Image 'x' true signal = ones; 20 noisy repeats average toward it.
    reps = np.ones((20, 2, 5)) + 0.5 * rng.randn(20, 2, 5)
    eeg = reps
    ids = ["x"] * 20
    avg, uids, counts = average_by_image(eeg, ids)
    assert avg.shape == (1, 2, 5)
    assert counts[0] == 20
    # Averaged estimate is closer to the true signal than a single trial.
    assert np.abs(avg[0] - 1.0).mean() < np.abs(reps[0] - 1.0).mean()


# --- preprocess_for_reve -----------------------------------------------------

def test_preprocess_resamples_250_to_200():
    rng = np.random.RandomState(2)
    eeg = rng.randn(4, 32, 250).astype(np.float64)
    out = preprocess_for_reve(eeg, sfreq_in=250, sfreq_out=200)
    assert out.shape == (4, 32, 200)


def test_preprocess_zscores_per_channel():
    rng = np.random.RandomState(3)
    # Channels with wildly different scale/offset; no extreme outliers so
    # the ±15 clamp does not bite. Keep sfreq_in==sfreq_out to isolate z-score.
    eeg = (rng.randn(3, 4, 200) * np.array([1, 50, 0.1, 1000])[None, :, None]
           + np.array([0, 100, -5, 3000])[None, :, None]).astype(np.float64)
    out = preprocess_for_reve(eeg, sfreq_in=200, sfreq_out=200, clamp=15.0)
    # Each (trial, channel) row is ~zero-mean, ~unit-std over time.
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-2)


def test_preprocess_clamps_outliers():
    eeg = np.zeros((1, 1, 100))
    eeg[0, 0, 0] = 1e6  # one giant spike → z-score blows up, must clamp
    out = preprocess_for_reve(eeg, sfreq_in=200, sfreq_out=200, clamp=15.0)
    assert np.isfinite(out).all()
    assert out.max() <= 15.0 + 1e-6
    assert out.min() >= -15.0 - 1e-6
