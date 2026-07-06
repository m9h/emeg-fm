"""ds002718 N170 label mapping + EEG-channel filtering (from channels.tsv 'type').
scripts/eegfm_t9.sh python -m pytest tests/test_cogneuro_extra.py -q
"""
import os
import sys

import numpy as np
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


def test_reve_names_from_electrode_positions_no_collisions_and_drops_nan(tmp_path):
    """Regression-style check on the real bug hit: some electrodes have NaN
    positions for a given subject (must not crash / must not silently produce
    an all-same-name degenerate mapping)."""
    ch = ["EEG001", "EEG002", "EEG003", "EEG004"]
    tsv = tmp_path / "sub-004_electrodes.tsv"
    tsv.write_text(
        "name\tx\ty\tz\n"
        "EEG001\t-2.94\t8.39\t-0.70\n"   # near Fp1
        "EEG002\t2.99\t8.49\t-0.71\n"    # near Fp2
        "EEG003\tnan\tnan\tnan\n"        # missing position (real bug case)
        "EEG004\t0.01\t8.82\t-0.17\n"    # near Fpz
    )
    mapped = ce._reve_names_from_electrode_positions(str(tsv), ch)
    assert len(mapped) == 4
    assert mapped[2] == "EEG003"  # no position -> falls back to its own name
    assert len(set(mapped[:2] + [mapped[3]])) == 3  # the 3 valid ones are distinct


def test_p3_aud_label_fn_oddball_vs_standard_only():
    def label_fn(row):
        v = row.get("value")
        if v in ("oddball", "oddball_with_reponse"):
            return 1
        if v in ("standard", "standard_with_reponse"):
            return 0
        return None

    assert label_fn({"value": "oddball"}) == 1
    assert label_fn({"value": "oddball_with_reponse"}) == 1
    assert label_fn({"value": "standard"}) == 0
    assert label_fn({"value": "standard_with_reponse"}) == 0
    assert label_fn({"value": "noise"}) is None
    assert label_fn({"value": "noise_with_reponse"}) is None
    assert label_fn({"value": "ignore"}) is None


def test_p3_vis_label_fn_target_vs_nontarget():
    def label_fn(row):
        s = str(row.get("value", "")).strip()
        if s.startswith("S"):
            s = s[1:].strip()
        if len(s) != 2 or not s.isdigit():
            return None
        return 1 if s[0] == s[1] else 0

    assert label_fn({"value": "S 11"}) == 1   # block target A, stimulus A -> target
    assert label_fn({"value": "S22"}) == 1
    assert label_fn({"value": "S 21"}) == 0   # block target B, stimulus A -> non-target
    assert label_fn({"value": "S 25"}) == 0
    assert label_fn({"value": "S201"}) is None  # 3-digit response code
    assert label_fn({"value": "S202"}) is None
    assert label_fn({"value": "boundary"}) is None


def test_ern_label_fn_err_vs_cor_only():
    df = pd.DataFrame({
        "trial_type": ["boundary", "con", "err", "inc", "cor", "err"],
        "onset": [1.0, 2.0, 2.5, 3.0, 3.5, 4.0],
    })
    filtered = df[df["trial_type"].isin(["err", "cor"])]
    labels = (filtered["trial_type"] == "err").astype(int).tolist()
    assert len(filtered) == 3
    assert labels == [1, 0, 1]
