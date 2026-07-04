"""Sleep-staging epoch extraction: synthetic Raw+Annotations (no real EDF needed).
Run in the 26.06 container (mne lives in /mnt/t9/moabblibs):
scripts/eegfm_t9.sh python -m pytest tests/test_sleep_section.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_sleep_section as sl  # noqa: E402


def _synthetic_raw(sfreq=100.0, n_epochs_per_stage=3):
    import mne
    mne.set_log_level("error")
    stages = ["Sleep stage W", "Sleep stage 1", "Sleep stage 2",
              "Sleep stage 3", "Sleep stage 4", "Sleep stage R"]
    onsets, durations, descs = [], [], []
    t = 0.0
    for s in stages:
        for _ in range(n_epochs_per_stage):
            onsets.append(t); durations.append(sl.EPOCH_S); descs.append(s)
            t += sl.EPOCH_S
    n_samples = int(t * sfreq)
    data = np.random.randn(2, n_samples) * 1e-5
    info = mne.create_info(sl.EEG_CHANNELS, sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_annotations(mne.Annotations(onsets, durations, descs), emit_warning=False)
    return raw


def test_stage_map_covers_all_aasm_labels():
    assert set(sl.STAGE_MAP) == {
        "Sleep stage W", "Sleep stage 1", "Sleep stage 2",
        "Sleep stage 3", "Sleep stage 4", "Sleep stage R"}
    # stage 3 and 4 both map to N3 (index 3) per AASM merging
    assert sl.STAGE_MAP["Sleep stage 3"] == sl.STAGE_MAP["Sleep stage 4"] == 3
    assert len(sl.STAGE_NAMES) == 5


def test_events_from_annotations_gives_expected_epoch_count_and_labels():
    import mne
    raw = _synthetic_raw(n_epochs_per_stage=3)
    events, _ = mne.events_from_annotations(
        raw, event_id=sl.STAGE_MAP, chunk_duration=sl.EPOCH_S, verbose="error")
    # 6 stages x 3 epochs = 18 total epochs, labels 0..4 with N3(3) appearing 2x as often
    assert len(events) == 18
    labels = events[:, 2]
    counts = {i: int((labels == i).sum()) for i in range(5)}
    assert counts[3] == 6            # stage 3 + stage 4 both -> label 3
    assert counts[0] == counts[1] == counts[2] == counts[4] == 3


def test_load_epochs_shapes_and_labels(tmp_path=None):
    import mne
    raw = _synthetic_raw(n_epochs_per_stage=2)
    psg = "/mnt/t9/_synth_test-PSG.edf"
    # mne.Annotations.save only supports FIF/CSV/TXT (not EDF); real Sleep-EDF
    # Hypnograms ARE genuine EDF+ files, but mne.read_annotations dispatches
    # purely on extension, so a .csv round-trip exercises load_epochs' actual
    # logic exactly.
    hyp = "/mnt/t9/_synth_test-Hypnogram.csv"
    mne.export.export_raw(psg, raw, fmt="edf", overwrite=True, verbose="error")
    raw.annotations.save(hyp, overwrite=True)
    X, y = sl.load_epochs(psg, hyp, crop_wake_min=0.0)
    assert X is not None
    assert X.shape[1] == 2                       # 2 EEG channels
    assert X.shape[2] == int(sl.EPOCH_S * raw.info["sfreq"])
    assert set(np.unique(y)) <= {0, 1, 2, 3, 4}
    assert len(y) == len(X)


def test_load_epochs_with_nonzero_crop_offset_recovers_correct_length():
    """Regression test for the MNE crop/events_from_annotations first_samp bug:
    after raw.crop(t0>0), events_from_annotations returns onset samples relative
    to the (now nonzero) raw.first_samp, but raw.get_data() is 0-indexed. Without
    subtracting first_samp, every epoch's slice lands out-of-bounds (empty) ->
    load_epochs silently returns (None, None) even though valid epochs exist."""
    import mne
    raw = _synthetic_raw(n_epochs_per_stage=4)
    psg = "/mnt/t9/_synth_crop_test-PSG.edf"
    hyp = "/mnt/t9/_synth_crop_test-Hypnogram.csv"
    mne.export.export_raw(psg, raw, fmt="edf", overwrite=True, verbose="error")
    raw.annotations.save(hyp, overwrite=True)
    # crop_wake_min small enough that the crop actually trims (nonzero t0)
    X, y = sl.load_epochs(psg, hyp, crop_wake_min=0.5)
    assert X is not None, "load_epochs must not silently drop all epochs after crop"
    assert len(X) > 0
    assert X.shape[2] == int(sl.EPOCH_S * raw.info["sfreq"])


def test_reve_channel_names_map_bipolar_derivations_to_anchor_electrode():
    ch = sl._reve_channel_names()
    assert ch == ["Cz", "Oz"]   # Fpz-Cz -> Cz, Pz-Oz -> Oz (anchor/second electrode)
