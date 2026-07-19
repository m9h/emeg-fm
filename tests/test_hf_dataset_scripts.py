"""Arithmetic/BCI2020-3/MDD thin wrapper scripts: channel-list config sanity
(the actual grouping/split logic is shared and tested in
test_hf_braindecode.py -- these scripts are mostly config wiring).
scripts/eegfm_t9.sh python -m pytest tests/test_hf_dataset_scripts.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_workload_arithmetic as ar  # noqa: E402
import atlas_depression_mdd as mdd  # noqa: E402


def test_arithmetic_channels_are_std19_monopolar():
    names = ar._reve_channel_names()
    assert len(names) == 19
    assert "Fp1" in names and "O2" in names
    assert "HEOL" not in names and "HEOR" not in names


def test_mdd_channels_are_std19_monopolar():
    names = mdd._reve_channel_names()
    assert len(names) == 19
    assert set(names) == {
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
        "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
    }
