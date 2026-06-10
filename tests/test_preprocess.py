"""Tests for the REVE Layer-2 input contract (emeg_fm.alljoined.ReveInputNorm).

The canonical reve.yaml recipe fits one per-channel StandardScaler over the
whole recording, then z-scores every epoch with those frozen stats and clamps
to ±15. These tests pin the three properties that make that contract correct
for streaming: (1) the scaler is per-channel over all samples, (2) it is frozen
after fit so online epochs reuse calibration stats, (3) transform is per-epoch
independent so batch == stacked singles (stream==batch parity), and (4) unlike
per-epoch z-score, the clamp actually fires on loud excursions.
"""
import numpy as np
import pytest

from emeg_fm.alljoined import ReveInputNorm, preprocess_for_reve


SFREQ = 200.0   # == sfreq_out → no resampling, keeps assertions exact


def _epochs(n=8, c=4, t=50, seed=0):
    rng = np.random.default_rng(seed)
    # distinct per-channel scale so a per-channel scaler has work to do
    scale = np.array([1.0, 5.0, 20.0, 0.5])[:c]
    return rng.standard_normal((n, c, t)) * scale[None, :, None]


def test_fit_computes_per_channel_stats():
    eeg = _epochs()
    norm = ReveInputNorm(sfreq_out=SFREQ).fit(eeg, SFREQ)
    # stats pool over trials AND time → one mean/std per channel
    assert norm.mean_.shape == (eeg.shape[1],)
    np.testing.assert_allclose(norm.mean_, eeg.mean(axis=(0, 2)))
    np.testing.assert_allclose(norm.std_, eeg.std(axis=(0, 2)) + 1e-8)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        ReveInputNorm().transform(_epochs()[0], SFREQ)


def test_stream_equals_batch_parity():
    eeg = _epochs()
    norm = ReveInputNorm(sfreq_out=SFREQ).fit(eeg, SFREQ)
    batch = norm.transform(eeg, SFREQ)
    singles = np.stack([norm.transform(eeg[i], SFREQ) for i in range(len(eeg))])
    np.testing.assert_allclose(batch, singles)


def test_frozen_scaler_uses_calibration_stats_not_epoch():
    # Online epoch from a DIFFERENT amplitude regime must be scaled by the
    # calibration stats, not re-normalized to its own unit variance.
    calib = _epochs(seed=1)
    norm = ReveInputNorm(sfreq_out=SFREQ).fit(calib, SFREQ)
    loud = calib[0] * 4.0                       # 4× louder than calibration
    out = norm.transform(loud, SFREQ)
    expected = (loud - norm.mean_[:, None]) / norm.std_[:, None]
    np.testing.assert_allclose(out, np.clip(expected, -norm.clamp, norm.clamp))
    # std of the scaled loud epoch should be well above 1 (NOT re-standardized)
    assert out.std() > 1.5


def test_clamp_fires_unlike_per_epoch_zscore():
    calib = _epochs(seed=2)
    norm = ReveInputNorm(sfreq_out=SFREQ, clamp=15.0).fit(calib, SFREQ)
    spike = calib[0].copy()
    spike[0, 0] = calib[:, 0].std() * 1000.0    # a single huge sample
    out = norm.transform(spike, SFREQ)
    assert np.isclose(out[0, 0], 15.0)          # clamp engaged
    # the same spike under per-epoch z-score lands far from the clamp bound
    per_epoch = preprocess_for_reve(spike[None], sfreq_in=SFREQ,
                                    sfreq_out=SFREQ, clamp=15.0)[0]
    assert abs(per_epoch[0, 0]) < 15.0


def test_channel_count_mismatch_raises():
    norm = ReveInputNorm(sfreq_out=SFREQ).fit(_epochs(c=4), SFREQ)
    with pytest.raises(ValueError, match="channel count"):
        norm.transform(_epochs(c=3)[0], SFREQ)


def test_resample_changes_length():
    eeg = _epochs(t=100)
    norm = ReveInputNorm(sfreq_out=100.0).fit(eeg, sfreq_in=200.0)
    out = norm.transform(eeg, sfreq_in=200.0)
    assert out.shape[-1] == 50          # 200→100 Hz halves the samples
