"""HMC sleep-scoring txt parser (drops non-stage annotations like 'Lights off').
scripts/eegfm_t9.sh python -m pytest tests/test_hmc_section.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_sleep_hmc as hmc  # noqa: E402


def test_parse_scoring_txt_drops_non_stage_rows(tmp_path):
    txt = tmp_path / "SN001_sleepscoring.txt"
    txt.write_text(
        "Date, Time, Recording onset, Duration, Annotation, Linked channel\n"
        "01.01.01, 23.30.00, 0, 30, Sleep stage W, \n"
        "01.01.01, 23.30.30, 30, 30, Sleep stage W, \n"
        "01.01.01, 23.30.46.280, 46.28, 0, Lights off, Body position\n"
        "01.01.01, 23.31.00, 60, 30, Sleep stage 2, \n"
    )
    events = hmc._parse_scoring_txt(str(txt))
    assert events == [
        (0.0, 30.0, "Sleep stage W"),
        (30.0, 30.0, "Sleep stage W"),
        (60.0, 30.0, "Sleep stage 2"),
    ]
    assert not any(d == "Lights off" for _, _, d in events)


def test_reve_channel_names_use_active_electrode():
    assert hmc._reve_channel_names() == ["F4", "C4", "O2", "C3"]
