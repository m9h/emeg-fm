"""CHB-MIT loader: summary.txt seizure-time parsing (both numbering styles),
chb21==chb01 patient merge, REVE anchor-electrode naming.
scripts/eegfm_t9.sh python -m pytest tests/test_epilepsy_chbmit.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_epilepsy_chbmit as cb  # noqa: E402

SINGLE_SEIZURE_SUMMARY = """\
File Name: chb01_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
"""

MULTI_SEIZURE_SUMMARY = """\
File Name: chb04_08.edf
File Start Time: 12:34:56
File End Time: 13:34:56
Number of Seizures in File: 2
Seizure 1 Start Time: 1679 seconds
Seizure 1 End Time: 1781 seconds
Seizure 2 Start Time: 3782 seconds
Seizure 2 End Time: 3898 seconds
"""

CHB24_STYLE_SUMMARY = """\
File Name: chb24_01.edf
Number of Seizures in File: 2
Seizure Start Time: 480 seconds
Seizure End Time: 505 seconds
Seizure Start Time: 2451 seconds
Seizure End Time: 2476 seconds
"""


def test_parse_summary_handles_zero_and_single_seizure_files():
    out = cb._parse_summary(SINGLE_SEIZURE_SUMMARY)
    assert out["chb01_01.edf"] == []
    assert out["chb01_03.edf"] == [(2996.0, 3036.0)]


def test_parse_summary_handles_numbered_multi_seizure_files():
    out = cb._parse_summary(MULTI_SEIZURE_SUMMARY)
    assert out["chb04_08.edf"] == [(1679.0, 1781.0), (3782.0, 3898.0)]


def test_parse_summary_handles_chb24_unnumbered_multi_seizure_style():
    out = cb._parse_summary(CHB24_STYLE_SUMMARY)
    assert out["chb24_01.edf"] == [(480.0, 505.0), (2451.0, 2476.0)]


def test_chb21_merges_into_chb01_patient_id():
    assert cb._patient_id("chb21") == "chb01"
    assert cb._patient_id("chb01") == "chb01"
    assert cb._patient_id("chb05") == "chb05"


def test_reve_channel_names_use_anchor_electrode_of_each_bipolar_pair():
    names = cb._reve_channel_names()
    assert names[0] == "FP1"  # FP1-F7
    assert names[1] == "F7"   # F7-T7
    assert len(names) == len(cb.STD_BIPOLAR)


def test_label_windows_flags_overlapping_seizure_window():
    starts = [0.0, 10.0, 20.0, 30.0]
    y = cb.label_windows(starts, 10.0, [(15.0, 25.0)])
    assert list(y) == [0, 1, 1, 0]
