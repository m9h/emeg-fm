"""Tests for the Layer-3 DeviceBridge (emeg_fm.device).

Synthetic single-tone signals exercise the filter behaviour without a headset:
the mains notch must kill 60 Hz while passing 10 Hz, the high-pass must remove
DC drift, the low-pass must auto-drop below Nyquist (the 128 Hz EPOC case), and
the whole transform must be per-epoch independent (batch == stacked singles).
"""
import numpy as np
import pytest

from emeg_fm.device import DeviceBridge


def _tone(freq, sfreq, secs, n_ch=4, amp=1.0, phase=0.0):
    t = np.arange(int(secs * sfreq)) / sfreq
    sig = amp * np.sin(2 * np.pi * freq * t + phase)
    return np.tile(sig, (n_ch, 1))           # (C, T), identical across channels


def _mid_rms(x):
    # central half, away from filtfilt edge transients
    n = x.shape[-1]
    return float(np.sqrt(np.mean(x[..., n // 4: 3 * n // 4] ** 2)))


def test_notch_kills_mains_passes_signal():
    sfreq = 256.0
    br = DeviceBridge(sfreq, highpass=0.5, lowpass=99.5, notch=60.0)
    keep = br.process(_tone(10.0, sfreq, 4.0))
    mains = br.process(_tone(60.0, sfreq, 4.0))
    # 10 Hz survives the band-pass, 60 Hz is notched out
    assert _mid_rms(keep) > 0.6
    assert _mid_rms(mains) < 0.15


def test_highpass_removes_dc_and_drift():
    sfreq = 256.0
    br = DeviceBridge(sfreq, highpass=0.5, lowpass=99.5, notch=None)
    drift = np.linspace(-50, 50, int(4 * sfreq))[None, :] * np.ones((3, 1))
    out = br.process(drift + 5.0)            # ramp + large DC offset
    assert _mid_rms(out) < 1.0               # near-flat after high-pass


def test_lowpass_dropped_below_nyquist():
    # 128 Hz device → Nyquist 64 → 99.5 lowpass is impossible, must be dropped
    br = DeviceBridge(128.0, highpass=0.5, lowpass=99.5, notch=60.0)
    assert br.lowpass is None
    # high-pass band still designed; a 20 Hz tone passes
    out = br.process(_tone(20.0, 128.0, 4.0))
    assert _mid_rms(out) > 0.6


def test_emotiv_preset():
    br = DeviceBridge.for_emotiv(sfreq=128.0, mains=60.0)
    assert br.notch == 60.0
    assert br.lowpass is None                # 99.5 above 64 Hz Nyquist
    out = br.process(_tone(60.0, 128.0, 4.0))
    assert _mid_rms(out) < 0.2               # mains still notched


def test_stream_equals_batch_parity():
    sfreq = 256.0
    br = DeviceBridge(sfreq, notch=60.0)
    rng = np.random.default_rng(0)
    batch_in = rng.standard_normal((5, 4, int(2 * sfreq)))
    batch = br.process(batch_in)
    singles = np.stack([br.process(batch_in[i]) for i in range(len(batch_in))])
    np.testing.assert_allclose(batch, singles, atol=1e-10)


def test_shape_preserved():
    sfreq = 256.0
    br = DeviceBridge(sfreq)
    assert br.process(np.zeros((4, 400))).shape == (4, 400)
    assert br.process(np.zeros((3, 4, 400))).shape == (3, 4, 400)


def test_common_average_reference_zeros_cross_channel_mean():
    sfreq = 256.0
    br = DeviceBridge(sfreq, highpass=None, lowpass=None, notch=None,
                      reref="average")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 300)) + np.array([1, 2, 3, 4, 5, 6])[:, None]
    out = br.process(x)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)


def test_label_reref_subtracts_named_channels():
    sfreq = 256.0
    br = DeviceBridge(sfreq, highpass=None, lowpass=None, notch=None,
                      reref=["M1", "M2"])
    x = np.arange(4 * 10, dtype=float).reshape(4, 10)
    ch = ["Fz", "Cz", "M1", "M2"]
    out = br.process(x, ch_names=ch)
    expected = x - x[[2, 3]].mean(axis=0, keepdims=True)
    np.testing.assert_allclose(out, expected)


def test_notch_harmonics_designed():
    # 60 Hz + harmonics below Nyquist (256 → 60, 120; 180 > 128 dropped)
    br = DeviceBridge(256.0, notch=60.0, notch_harmonics=True)
    assert len(br._notches) == 2
    # a higher-rate device admits more harmonics (512 → 60,120,180,240)
    assert len(DeviceBridge(512.0, notch=60.0, notch_harmonics=True)._notches) == 4
