"""SeizeIT2 events.tsv seizure-label extraction (SCORE HED vocabulary).
scripts/eegfm_t9.sh python -m pytest tests/test_seizeit2_section.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_epilepsy_seizeit2 as sz  # noqa: E402


def test_load_seiz_events_filters_sz_prefix_only(tmp_path):
    tsv = tmp_path / "sub-001_ses-01_task-szMonitoring_run-01_events.tsv"
    tsv.write_text(
        "onset\tduration\teventType\tlateralization\n"
        "10.0\t5.0\tbckg\tn/a\n"
        "20.0\t2.0\timpd\tn/a\n"
        "30.0\t45.0\tsz_foc_ia_m_automatisms\tleft\n"
        "100.0\t12.0\tsz_foc_ua_nm\tn/a\n"
    )
    events = sz.load_seiz_events(str(tsv))
    assert events == [(30.0, 75.0), (100.0, 112.0)]
    # bckg/impd must NOT appear as seizures
    assert not any(a == 10.0 or a == 20.0 for a, _ in events)


def test_load_seiz_events_empty_when_no_seizures(tmp_path):
    tsv = tmp_path / "sub-002_ses-01_task-szMonitoring_run-01_events.tsv"
    tsv.write_text("onset\tduration\teventType\n0.0\t5.0\tbckg\n10.0\t1.0\timpd\n")
    assert sz.load_seiz_events(str(tsv)) == []
