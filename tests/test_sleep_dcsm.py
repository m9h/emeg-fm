"""DCSM loader: variable-duration interval hypnogram -> fixed 30s epoch
expansion, REVE anchor names.
scripts/eegfm_t9.sh python -m pytest tests/test_sleep_dcsm.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_sleep_dcsm as dc  # noqa: E402


def test_parse_hypnogram_expands_intervals_to_30s_epochs(tmp_path):
    hyp = tmp_path / "hypnogram.ids"
    hyp.write_text("0,60,W\n60,30,N1\n90,90,N2\n")
    y = dc._parse_hypnogram(str(hyp))
    # 60/30=2 W epochs, 30/30=1 N1 epoch, 90/30=3 N2 epochs
    assert list(y) == [0, 0, 1, 2, 2, 2]


def test_parse_hypnogram_flags_unmapped_stage_as_dropped(tmp_path):
    hyp = tmp_path / "hypnogram.ids"
    hyp.write_text("0,30,W\n30,30,UNKNOWN\n")
    y = dc._parse_hypnogram(str(hyp))
    assert list(y) == [0, -1]


def test_reve_channel_names_use_anchor_electrode_of_each_bipolar_pair():
    names = dc._reve_channel_names()
    assert names == ["F3", "F4", "C3", "C4", "O1", "O2"]


def test_iter_recordings_pairs_edf_with_hypnogram(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "ROOT", str(tmp_path))
    d = tmp_path / "tp0001"
    d.mkdir()
    (d / "hypnogram.ids").write_text("0,30,W\n")
    # no matching psg.edf -> excluded
    assert dc.iter_recordings() == []
    (d / "psg.edf").write_text("")
    recs = dc.iter_recordings()
    assert len(recs) == 1
    assert recs[0][2] == "tp0001"
