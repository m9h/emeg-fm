"""ISRUC loader: label remap {0,1,2,3,5}->0-4, REVE anchor names, subgroup-2
nested-night directory layout vs subgroup 1/3's flat layout.
scripts/eegfm_t9.sh python -m pytest tests/test_sleep_isruc.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_sleep_isruc as ir  # noqa: E402


def test_label_map_covers_isruc_convention_no_stage_4():
    assert ir.LABEL_MAP == {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
    assert 4 not in ir.LABEL_MAP


def test_reve_channel_names_use_anchor_electrode_of_each_bipolar_pair():
    names = ir._reve_channel_names()
    assert names == ["F3", "C3", "O1", "F4", "C4", "O2"]


def test_iter_recordings_handles_flat_and_nested_subgroup_layouts(tmp_path, monkeypatch):
    monkeypatch.setattr(ir, "ROOT", str(tmp_path))
    # subgroup 1: flat, subject "1"
    g1 = tmp_path / "1" / "1"
    g1.mkdir(parents=True)
    (g1 / "1.rec").write_text("")
    (g1 / "1_1.txt").write_text("0\n1\n")
    # subgroup 2: nested night dirs, subject "1", nights "1" and "2"
    g2n1 = tmp_path / "2" / "1" / "1"
    g2n1.mkdir(parents=True)
    (g2n1 / "1.rec").write_text("")
    (g2n1 / "1_1.txt").write_text("0\n1\n")
    g2n2 = tmp_path / "2" / "1" / "2"
    g2n2.mkdir(parents=True)
    (g2n2 / "2.rec").write_text("")
    (g2n2 / "2_1.txt").write_text("0\n1\n")
    # subgroup 3: flat, subject "1"
    g3 = tmp_path / "3" / "1"
    g3.mkdir(parents=True)
    (g3 / "1.rec").write_text("")
    (g3 / "1_1.txt").write_text("0\n1\n")

    recs = ir.iter_recordings()
    patients = sorted({p for _, _, p in recs})
    assert patients == ["g1_1", "g2_1", "g3_1"]
    assert len(recs) == 4  # g2_1 contributes 2 recordings (2 nights)
