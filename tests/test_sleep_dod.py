"""Dreem DOD loader: hypnogram epoch-alignment + NOT-SCORED filtering + REVE
anchor-electrode naming.
scripts/eegfm_t9.sh python -m pytest tests/test_sleep_dod.py -q
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_sleep_dod as dod  # noqa: E402


def _make_fake_dodh_h5(path, n_epochs=5, fs=250, epoch_s=30.0):
    n_samp = int(n_epochs * epoch_s * fs)
    with h5py.File(path, "w") as f:
        hyp = np.array([0, 1, -1, 2, 4], dtype=np.int64)
        assert len(hyp) == n_epochs
        f.create_dataset("hypnogram", data=hyp)
        grp = f.create_group("signals/eeg")
        grp.attrs["fs"] = fs
        for c in dod.EEG_CHANNELS["dodh"]:
            key = c.replace("-", "_")
            grp.create_dataset(key, data=np.arange(n_samp, dtype=np.float32))
        f.create_group("events")


def test_load_epochs_drops_not_scored_and_aligns_to_hypnogram(tmp_path, monkeypatch):
    h5_path = tmp_path / "fake.h5"
    _make_fake_dodh_h5(str(h5_path))
    X, y = dod.load_epochs(str(h5_path), "dodh")
    # 5 epochs in hypnogram, 1 is NOT SCORED (-1) -> 4 kept
    assert len(y) == 4
    assert list(y) == [0, 1, 2, 4]
    assert X.shape == (4, len(dod.EEG_CHANNELS["dodh"]), int(30.0 * 250))


def test_reve_channel_names_uses_first_anchor_electrode():
    names = dod._reve_channel_names("dodh")
    assert names[dod.EEG_CHANNELS["dodh"].index("C3-M2")] == "C3"
    assert names[dod.EEG_CHANNELS["dodh"].index("F3-F4")] == "F3"
    assert len(names) == len(dod.EEG_CHANNELS["dodh"])


def test_dodo_montage_is_8ch_and_distinct_from_dodh():
    assert len(dod.EEG_CHANNELS["dodo"]) == 8
    assert len(dod.EEG_CHANNELS["dodh"]) == 12
    assert dod.EEG_CHANNELS["dodo"] != dod.EEG_CHANNELS["dodh"]


def test_iter_recordings_prefixes_subject_with_cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(dod, "ROOT", str(tmp_path))
    (tmp_path / "dodh").mkdir()
    (tmp_path / "dodh" / "abc123.h5").write_bytes(b"")
    recs = dod.iter_recordings("dodh")
    assert len(recs) == 1
    assert recs[0][1] == "dodh_abc123"
