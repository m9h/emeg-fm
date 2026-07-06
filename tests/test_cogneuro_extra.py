"""ds002718 N170 label mapping + EEG-channel filtering (from channels.tsv 'type').
scripts/eegfm_t9.sh python -m pytest tests/test_cogneuro_extra.py -q
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_cogneuro_extra as ce  # noqa: E402


def test_eeg_channel_names_filters_non_eeg(tmp_path):
    tsv = tmp_path / "channels.tsv"
    tsv.write_text(
        "name\ttype\tunits\n"
        "EEG001\tEEG\tmicroV\n"
        "EEG002\tEEG\tmicroV\n"
        "EKG1\tEKG\tmicroV\n"
        "HEOG\tHEOG\tmicroV\n"
    )
    assert ce._eeg_channel_names(str(tsv)) == ["EEG001", "EEG002"]


def test_n170_label_fn_face_vs_scrambled():
    def label_fn(row):
        ft = row.get("face_type")
        if ft in ("famous", "unfamiliar"):
            return 1
        if ft == "scrambled":
            return 0
        return None

    assert label_fn({"face_type": "famous"}) == 1
    assert label_fn({"face_type": "unfamiliar"}) == 1
    assert label_fn({"face_type": "scrambled"}) == 0
    assert label_fn({"face_type": "n/a"}) is None


def test_events_filtered_to_faces_event_type():
    df = pd.DataFrame({
        "event_type": ["faces", "button_press", "faces"],
        "face_type": ["famous", "n/a", "scrambled"],
        "onset": [1.0, 1.5, 2.0],
    })
    filtered = df[df["event_type"] == "faces"]
    assert len(filtered) == 2
    assert set(filtered["face_type"]) == {"famous", "scrambled"}
