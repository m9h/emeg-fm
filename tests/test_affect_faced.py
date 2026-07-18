"""FACED loader: video-index -> emotion label mapping, Neutral placeholder
handling, channel-count exclusion of EOG.
scripts/eegfm_t9.sh python -m pytest tests/test_affect_faced.py -q
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_affect_faced as fc  # noqa: E402


def test_build_label_map_uses_valence_for_neutral_placeholder(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "Video index": [1, 2, 13, 17],
        "Valence": ["Negative", "Negative", "Neutral", "Positive"],
        "Targeted Emotion": ["Anger", "Anger", "\\", "Amusement"],
    })
    path = tmp_path / "Stimuli_info.xlsx"
    df.to_excel(path, index=False)
    monkeypatch.setattr(fc, "STIMULI_INFO", str(path))
    label_map = fc._build_label_map()
    assert label_map == {1: "Anger", 2: "Anger", 13: "Neutral", 17: "Amusement"}


def test_label_to_int_map_is_sorted_and_stable():
    label_map = {1: "Anger", 2: "Joy", 3: "Anger", 4: "Neutral"}
    class_map = fc._label_to_int_map(label_map)
    assert class_map == {"Anger": 0, "Joy": 1, "Neutral": 2}


def test_load_epochs_excludes_eog_channels_and_maps_trial_order(tmp_path):
    import pickle
    n_ch_total = 32
    arr = np.random.default_rng(0).standard_normal((28, n_ch_total, 100)).astype(np.float64)
    pkl_path = tmp_path / "sub999.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(arr, f)
    label_map = {i + 1: "Anger" if i < 14 else "Joy" for i in range(28)}
    class_map = {"Anger": 0, "Joy": 1}
    X, y = fc.load_epochs(str(pkl_path), label_map, class_map)
    assert X.shape == (28, 30, 100)  # 32 -> 30 EEG channels (EOG dropped)
    assert list(y[:14]) == [0] * 14
    assert list(y[14:]) == [1] * 14


def test_reve_channel_names_are_real_monopolar_names_no_eog():
    names = fc._reve_channel_names()
    assert len(names) == 30
    assert "HEOL" not in names and "HEOR" not in names
    assert names[0] == "Fp1" and names[-1] == "O2"
