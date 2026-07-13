"""Ear-EEG loader: scoring-code mapping (Artefact dropped) + REVE alias names.
scripts/eegfm_t9.sh python -m pytest tests/test_sleep_eareeg.py -q
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_sleep_eareeg as ee  # noqa: E402


def test_scoring_map_drops_artefact_and_matches_stage_convention():
    # 1=Wake, 2=REM, 3=N1, 4=N2, 5=N3, 6=Artefact
    assert ee.SCORING_MAP[1] == 0  # Wake -> W
    assert ee.SCORING_MAP[3] == 1  # N1
    assert ee.SCORING_MAP[4] == 2  # N2
    assert ee.SCORING_MAP[5] == 3  # N3
    assert ee.SCORING_MAP[2] == 4  # REM
    assert 6 not in ee.SCORING_MAP  # Artefact must be absent (dropped)


def test_reve_channel_names_use_published_ear_electrode_aliases():
    names = ee._reve_channel_names()
    assert names == ["A2", "T8", "A1", "T7"]


def test_load_epochs_drops_artefact_rows(tmp_path, monkeypatch):
    import mne
    sfreq = 250.0
    n_samp = int(4 * 30 * sfreq)  # 4 epochs worth
    data = np.random.randn(4, n_samp).astype(np.float32) * 1e-5
    info = mne.create_info(ee.EEG_CHANNELS, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    set_path = tmp_path / "sub-001_ses-001_task-sleep_acq-earEEG_eeg.set"
    mne.export.export_raw(str(set_path), raw, fmt="eeglab", overwrite=True, verbose="error")

    tsv_path = tmp_path / "sub-001_ses-001_task-sleep_acq-scoring_events.tsv"
    df = pd.DataFrame({
        "onset": [0.0, 30.0, 60.0, 90.0],
        "duration": [30, 30, 30, 30],
        "scoring_idx": [1, 6, 4, 2],  # Wake, Artefact, N2, REM
        "scoring": ["Wake", "Artefact", "N2", "REM"],
    })
    df.to_csv(tsv_path, sep="\t", index=False)

    X, y = ee.load_epochs(str(set_path), str(tsv_path))
    # 4 epochs in the tsv, 1 is Artefact -> 3 kept
    assert len(y) == 3
    assert list(y) == [0, 2, 4]
    assert X.shape == (3, 4, int(30 * sfreq))


def test_drop_nan_rows_filters_embeddings_and_labels_together():
    emb = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    y = np.array([0, 1, 2])
    emb_clean, y_clean, n_dropped = ee._drop_nan_rows(emb, y)
    assert n_dropped == 1
    assert emb_clean.shape == (2, 2)
    assert list(y_clean) == [0, 2]


def test_iter_recordings_only_includes_sessions_with_both_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ee, "ROOT", str(tmp_path))
    d = tmp_path / "sub-001" / "ses-001" / "eeg"
    d.mkdir(parents=True)
    (d / "sub-001_ses-001_task-sleep_acq-scoring_events.tsv").write_text("onset\tduration\tscoring_idx\tscoring\n")
    # no matching _eeg.set -> should be excluded
    assert ee.iter_recordings() == []
    (d / "sub-001_ses-001_task-sleep_acq-earEEG_eeg.set").write_text("")
    recs = ee.iter_recordings()
    assert len(recs) == 1
    assert recs[0][2] == "sub-001"
