"""Helsinki neonatal seizure EEG: consensus voting + mask-to-events conversion.
scripts/eegfm_t9.sh python -m pytest tests/test_helsinki_section.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_epilepsy_helsinki as hk  # noqa: E402


def test_mask_to_events_basic():
    mask = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    events = hk._mask_to_events(mask, sfreq_mask=1.0)
    assert events == [(2.0, 5.0), (7.0, 8.0)]


def test_mask_to_events_trailing_seizure():
    mask = np.array([0, 1, 1], dtype=bool)
    assert hk._mask_to_events(mask, sfreq_mask=1.0) == [(1.0, 3.0)]


def test_mask_to_events_no_seizure():
    assert hk._mask_to_events(np.zeros(5, dtype=bool)) == []


def test_channel_rename_uppercases_mixed_case_names():
    """Regression: Helsinki's raw channels are mixed-case ('Fp1'), but STD19 is
    all-caps ('FP1'). Without upper(), every recording's channel match silently
    fails (0 recordings loaded, no error) -- exact bug hit on the real corpus."""
    import re
    raw_names = ["EEG Fp1-REF", "EEG Cz-REF", "EEG T3-REF"]
    ren = {c: re.sub(r"^EEG\s+|-REF$", "", c).strip().upper() for c in raw_names}
    assert set(ren.values()) == {"FP1", "CZ", "T3"}
    assert all(v in hk.STD19 for v in ren.values())


def test_load_consensus_majority_vote(tmp_path, monkeypatch):
    # 3 annotators, recording "7": A=[1,1,0], B=[1,0,0], C=[0,0,0]
    # majority (>=2): [1,0,0]
    root = tmp_path
    header = "1,7,9\n"
    (root / "annotations_2017_A.csv").write_text(header + "0,1,0\n0,1,0\n0,0,0\n")
    (root / "annotations_2017_B.csv").write_text(header + "0,1,0\n0,0,0\n0,0,0\n")
    (root / "annotations_2017_C.csv").write_text(header + "0,0,0\n0,0,0\n0,0,0\n")
    monkeypatch.setattr(hk, "ROOT", str(root))
    consensus = hk._load_consensus(7)
    assert consensus.tolist() == [True, False, False]


def test_load_consensus_handles_ragged_shorter_recording(tmp_path, monkeypatch):
    """Regression: the real corpus's matrix is ragged -- each column has values
    only for its own recording's duration; later rows are blank ("") for shorter
    recordings even while OTHER (longer) columns in the same row still have data.
    Recording "1" here is shorter (2 rows) than recording "9" (3 rows)."""
    root = tmp_path
    header = "1,9\n"
    (root / "annotations_2017_A.csv").write_text(header + "0,1\n1,0\n,0\n")
    (root / "annotations_2017_B.csv").write_text(header + "0,1\n0,0\n,0\n")
    (root / "annotations_2017_C.csv").write_text(header + "0,0\n1,0\n,0\n")
    monkeypatch.setattr(hk, "ROOT", str(root))
    # recording 1 (idx 0) truncates at the blank 3rd row -> only 2 valid samples
    consensus = hk._load_consensus(1)
    assert consensus.tolist() == [False, True]   # majority: row0 all-0, row1 A&C=1
