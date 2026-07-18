"""MOABB cohort builder: known-incompatible-subject exclusion.

Root cause of the PhysionetMotorImagery shape-mismatch crash (FAILED:
ValueError: operands could not be broadcast together with shapes (602,)
(601,)): MOABB's own docs say subject 88 in PhysionetMI was recorded at
128 Hz instead of 160 Hz like everyone else, and "loading subject 88 together
with other subjects will cause errors in any paradigm". Resampling a 128 Hz-
native trial and a 160 Hz-native trial to the same target sfreq rounds to
different sample counts, producing exactly this kind of off-by-one shape
mismatch when the cohort's per-subject arrays get combined.

scripts/eegfm_t9.sh python -m pytest tests/test_moabb_cohort_incompatible_subjects.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import emeg_fm.moabb_cohort as mc  # noqa: E402


class _FakeDataset:
    def __init__(self, code):
        self.code = code


def test_excludes_known_bad_subject_for_physionet_motor_imagery():
    ds = _FakeDataset("PhysionetMotorImagery")
    kept = mc._exclude_incompatible_subjects(ds, list(range(1, 6)))
    assert 88 not in kept  # trivially true here, but exercises the real path
    kept_with_88 = mc._exclude_incompatible_subjects(ds, [1, 2, 88, 5])
    assert kept_with_88 == [1, 2, 5]


def test_leaves_other_datasets_untouched():
    ds = _FakeDataset("BNCI2014-001")
    subjects = [1, 2, 88, 5]
    assert mc._exclude_incompatible_subjects(ds, subjects) == subjects


def test_falls_back_to_class_name_when_no_code_attr():
    class NoCodeDataset:
        pass
    ds = NoCodeDataset()
    subjects = [1, 2, 3]
    # unknown code -> no exclusion, no crash
    assert mc._exclude_incompatible_subjects(ds, subjects) == subjects
