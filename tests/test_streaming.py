"""Tests for emeg_fm.streaming — the pure epoching logic of the acquisition
bridge. No LSL / pylsl / headset: RingBuffer, channel selection, and the
marker→epoch assembler are exercised on synthetic uniform streams.
"""
import numpy as np
import pytest

from emeg_fm.streaming import (
    RingBuffer, select_channels, _EpochAssembler, Trial,
)


def _uniform_chunk(t0, n, sfreq, n_ch, value_base=0.0):
    ts = t0 + np.arange(n) / sfreq
    # each sample = its global index, broadcast across channels (easy to assert)
    samples = (value_base + np.arange(n))[:, None] * np.ones((1, n_ch))
    return ts, samples.astype(np.float32)


# --- RingBuffer --------------------------------------------------------------

def test_ringbuffer_epoch_slices_window():
    buf = RingBuffer(n_channels=4, sfreq=100.0, maxlen_s=5.0)
    ts, s = _uniform_chunk(10.0, 300, 100.0, 4)   # 3 s of data from t=10
    buf.push(ts, s)
    # window [10.0, 11.0): samples index 0..99
    ep = buf.epoch(onset_ts=10.0, tmin=0.0, tmax=1.0)
    assert ep.shape == (4, 100)
    # first column should be sample index 0, last should be 99
    assert ep[0, 0] == pytest.approx(0.0)
    assert ep[0, -1] == pytest.approx(99.0)


def test_ringbuffer_negative_tmin_baseline():
    buf = RingBuffer(4, 100.0, 5.0)
    ts, s = _uniform_chunk(0.0, 400, 100.0, 4)
    buf.push(ts, s)
    ep = buf.epoch(onset_ts=2.0, tmin=-0.2, tmax=1.0)   # 1.2 s → 120 samples
    assert ep.shape == (4, 120)


def test_ringbuffer_not_ready_raises():
    buf = RingBuffer(4, 100.0, 5.0)
    ts, s = _uniform_chunk(0.0, 50, 100.0, 4)           # only 0.5 s
    buf.push(ts, s)
    assert not buf.ready_for(0.0, tmax=1.0)
    with pytest.raises(ValueError, match="not ready"):
        buf.epoch(0.0, 0.0, 1.0)


def test_ringbuffer_aged_out_raises():
    buf = RingBuffer(4, 100.0, maxlen_s=1.0)            # holds only 100 samples
    ts, s = _uniform_chunk(0.0, 300, 100.0, 4)          # 3 s pushed → oldest gone
    buf.push(ts, s)
    with pytest.raises(ValueError, match="aged out"):
        buf.epoch(onset_ts=0.0, tmin=0.0, tmax=1.0)


def test_ringbuffer_dropout_detected():
    buf = RingBuffer(2, 100.0, 5.0)
    ts1, s1 = _uniform_chunk(0.0, 50, 100.0, 2)
    # gap: next chunk starts 0.5 s later than continuous would
    ts2, s2 = _uniform_chunk(1.0, 60, 100.0, 2, value_base=50)
    buf.push(ts1, s1); buf.push(ts2, s2)
    with pytest.raises(ValueError, match="dropout"):
        buf.epoch(onset_ts=0.0, tmin=0.0, tmax=1.1, max_gap_s=0.1)


def test_ringbuffer_channel_count_guard():
    buf = RingBuffer(8, 100.0, 5.0)
    ts, s = _uniform_chunk(0.0, 10, 100.0, 4)
    with pytest.raises(ValueError, match="channels"):
        buf.push(ts, s)


# --- select_channels ---------------------------------------------------------

def test_select_channels_reorders_and_subsets():
    epoch = np.arange(4 * 3).reshape(4, 3).astype(float)   # 4 ch × 3 samples
    src = ["Fp1", "Fp2", "Cz", "Oz"]
    out = select_channels(epoch, src, ["Oz", "Fp1"])
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out[0], epoch[3])        # Oz
    np.testing.assert_array_equal(out[1], epoch[0])        # Fp1


def test_select_channels_case_insensitive():
    epoch = np.zeros((2, 5))
    out = select_channels(epoch, ["FP1", "cz"], ["Fp1", "Cz"])
    assert out.shape == (2, 5)


def test_select_channels_missing_raises():
    epoch = np.zeros((2, 5))
    with pytest.raises(KeyError, match="T8"):
        select_channels(epoch, ["Fp1", "Cz"], ["Fp1", "T8"])


# --- _EpochAssembler (marker→epoch state machine) ---------------------------

def test_assembler_emits_trial_when_window_complete():
    asm = _EpochAssembler(ch_names=["a", "b"], sfreq=100.0, tmin=0.0, tmax=1.0)
    asm.push_marker(code=7, onset_ts=0.5)
    # not enough data yet
    ts, s = _uniform_chunk(0.0, 100, 100.0, 2)             # 0..1 s
    asm.push_eeg(ts, s)
    assert list(asm.drain()) == []                         # window end 1.5 not reached
    ts2, s2 = _uniform_chunk(1.0, 100, 100.0, 2, value_base=100)
    asm.push_eeg(ts2, s2)                                   # now through 2 s
    trials = list(asm.drain())
    assert len(trials) == 1
    assert isinstance(trials[0], Trial)
    assert trials[0].code == 7
    assert trials[0].epoch.shape == (2, 100)


def test_assembler_applies_montage():
    asm = _EpochAssembler(ch_names=["Fp1", "Fp2", "Cz"], sfreq=100.0,
                          tmin=0.0, tmax=1.0, montage=["Cz", "Fp1"])
    asm.push_marker(3, 0.0)
    ts, s = _uniform_chunk(0.0, 200, 100.0, 3)
    asm.push_eeg(ts, s)
    trial = next(iter(asm.drain()))
    assert trial.ch_names == ["Cz", "Fp1"]
    assert trial.epoch.shape == (2, 100)
