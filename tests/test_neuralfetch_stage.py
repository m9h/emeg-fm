"""Tests for the neuralfetch staging helper's roadmap-coverage map.

The staging CLI (scripts/neuralfetch_stage.py) classifies our dataset-staging
roadmap against neuralfetch's live catalog so we know which tasks it unblocks
(TUEG/HBN/NSD) and which still need a hand-rolled pull (CHBP/NSRR/Cam-CAN/...).
``roadmap_coverage`` is pure (takes a list of catalog names) so it is testable
without neuralfetch installed.
"""
from __future__ import annotations

import importlib.util
import os

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts",
                       "neuralfetch_stage.py")


def _load():
    spec = importlib.util.spec_from_file_location("_nfs", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# A stand-in for `neuralset.Study.catalog().keys()` (the studies that exist).
FAKE_CATALOG = [
    "Obeid2016Tueg", "Lopez2017Tuab", "Shirazi2024Hbn", "Allen2022MassiveSample",
    "Kemp2000Analysis", "Schalk2004Bci2000Moabb", "Test2023Fmri",
]


def test_covered_datasets_resolve():
    cov = _load().roadmap_coverage(FAKE_CATALOG)
    assert cov["TUEG"]["covered"] is True
    assert "Obeid2016Tueg" in cov["TUEG"]["matches"]
    assert cov["HBN"]["covered"] is True
    assert cov["NSD"]["covered"] is True


def test_uncovered_datasets_flagged():
    cov = _load().roadmap_coverage(FAKE_CATALOG)
    # These are NOT in neuralfetch -> must report covered=False so we keep the
    # hand-rolled pulls rather than assume they're handled.
    for missing in ("CHBP", "NSRR", "Cam-CAN", "TDBRAIN"):
        assert cov[missing]["covered"] is False, missing
        assert cov[missing]["matches"] == []


def test_every_roadmap_entry_carries_a_task_ref():
    cov = _load().roadmap_coverage(FAKE_CATALOG)
    # each entry names the staging task it bears on, so the CLI output is actionable
    for label, rec in cov.items():
        assert rec["task"], label
