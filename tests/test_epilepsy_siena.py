"""Siena loader: wall-clock seizure-time parsing (registration-relative
offsets, midnight wraparound, repeated-filename accumulation).
scripts/eegfm_t9.sh python -m pytest tests/test_epilepsy_siena.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_epilepsy_siena as sn  # noqa: E402

SINGLE_SEIZURE = """\
Seizure n 1
File name: PN00-1.edf
Registration start time: 19.39.33
Registration end time:  20.22.58
Seizure start time: 19.58.36
Seizure end time: 19.59.46
"""

MIDNIGHT_CROSSING = """\
Seizure n 2
File name: PN00-2.edf
Registration start time:  23.50.00
Registration end time:  02.56.19
Seizure start time: 00.10.00
Seizure end time: 00.11.00
"""

REPEATED_FILENAME_MULTI_SEIZURE = """\
Seizure n 4 (seizure B3):
File name: PN10-4.5.6.edf
Registration start time: 12.11.21
Registration end time: 16.48.49
Seizure start time: 12.49.50
Seizure end time: 12.49.55

Seizure n 5 (B4):
File name: PN10-4.5.6.edf
Registration start time: 12.11.21
Registration end time: 16.48.49
Seizure start time: 14.00.25
Seizure end time: 14.00.44
"""


def test_single_seizure_offset_from_registration_start():
    out = sn._parse_seizure_list(SINGLE_SEIZURE)
    # 19.58.36 - 19.39.33 = 19*60+3 = 1143s; end = 19.59.46-19.39.33=1213s
    assert out["PN00-1.edf"] == [(1143.0, 1213.0)]


def test_midnight_crossing_wraps_correctly():
    out = sn._parse_seizure_list(MIDNIGHT_CROSSING)
    # 23.50.00 -> 00.10.00 next day = 20 min = 1200s later
    assert out["PN00-2.edf"] == [(1200.0, 1260.0)]


def test_repeated_filename_accumulates_multiple_seizures():
    out = sn._parse_seizure_list(REPEATED_FILENAME_MULTI_SEIZURE)
    assert len(out["PN10-4.5.6.edf"]) == 2
    assert out["PN10-4.5.6.edf"][0][0] < out["PN10-4.5.6.edf"][1][0]


def test_label_windows_flags_overlapping_seizure_window():
    starts = [0.0, 10.0, 20.0, 30.0]
    y = sn.label_windows(starts, 10.0, [(15.0, 25.0)])
    assert list(y) == [0, 1, 1, 0]
