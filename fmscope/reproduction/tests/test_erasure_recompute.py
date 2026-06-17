"""Recompute the paper's subject-axis erasure from the bundled frozen features.

The paper-reproduction suite (``test_paper_reproduce.py``) checks the bundled
JSON aggregates against the manuscript. This test goes one level deeper: it runs
the *library's* ``subject_axis_erasure`` on the bundled per-window feature NPZs,
with the paper's own ``(subject, condition)`` recording grouping, and asserts the
recomputed label / subject balanced-accuracies match the paper's bundled
``fmscope_mechanism_erasure.json`` numbers. This is the verification that *we*
run FMScope's diagnostic correctly, not just that the snapshot is self-consistent.

Coverage: the 4 panel cells × 3 FMs (12 frozen NPZs). eegmat/sleepdep/stress have
label-erasure references; adftd is a trait cell (one label/subject) with only a
subject-probe reference in the mechanism table, so we check its subject probe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fmscope.diagnostics.erasure import subject_axis_erasure

_DATA = Path(__file__).resolve().parents[1] / "data"
_NPZ = _DATA / "features_cache"
_MECH = json.loads((_DATA / "source_tables" / "fmscope_mechanism_erasure.json").read_text())

_FMS = ("reve", "labram", "cbramod")
_LABEL_CELLS = ("eegmat", "sleepdep", "stress")  # have label_ba reference
_TRAIT_CELLS = ("adftd",)                          # subject-probe reference only


def _recompute(fm, cell):
    z = np.load(_NPZ / f"frozen_{fm}_{cell}_perwindow.npz", allow_pickle=True)
    return subject_axis_erasure(
        z["features"], z["window_pids"], z["window_labels"],
        window_recording=z["window_rec_idx"],
        rec_labels=z["rec_labels"], rec_pids=z["rec_pids"],
        cv="stratified-kfold",
    )


@pytest.mark.parametrize("fm", _FMS)
@pytest.mark.parametrize("cell", _LABEL_CELLS)
def test_label_erasure_matches_paper(fm, cell):
    ref = _MECH[f"{fm}/{cell}"]
    er = _recompute(fm, cell)
    # Library reproduces the paper's pooled (subject,condition) erasure. Most
    # cells match to <1e-3 (eegmat-reve is exact); near-chance cells (e.g.
    # sleepdep, BA~0.55) can differ by one pooled-recording prediction flipping
    # in one of 3 fixed seeds under a sklearn/numpy version delta — that shifts
    # the 3-seed mean by ~1/72/3 ≈ 0.0046, well inside the across-seed std.
    # Tolerance is one fold-flip's worth (1e-2); tighter would assert numerical
    # identity across library versions, which is not the contract being checked.
    assert er.label_ba_raw == pytest.approx(ref["label_ba_raw"], abs=1e-2)
    assert er.label_ba_erased == pytest.approx(ref["label_ba_erased"], abs=1e-2)


@pytest.mark.parametrize("fm", _FMS)
@pytest.mark.parametrize("cell", _TRAIT_CELLS)
def test_trait_subject_probe_matches_paper(fm, cell):
    ref = _MECH[f"{fm}/{cell}"]
    er = _recompute(fm, cell)
    # Trait cells: verify the identity subspace is recovered + erased as the
    # paper reports (subject probe pre/post), independent of any label decode.
    assert er.subj_ba_linear_pre == pytest.approx(ref["subj_ba_linear_pre"], abs=1e-2)
    assert er.subj_ba_linear_post == pytest.approx(ref["subj_ba_linear_post"], abs=1e-2)
