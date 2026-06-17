"""Tests for the corrected identity-trap verdict in the MOABB leaderboard.

The original ``_verdict`` stamped TRAP whenever subject variance dominated and
``free_ba >= raw_ba`` — with no interpretability check, so a below-gate, no-lift
per-trial result (e.g. BNCI2014-001: raw 0.537, Δ+0.002, not interpretable) was
mislabelled TRAP. The fix makes the verdict a function of the cross-subject
decode: no-transfer (below gate) / TRAP (interpretable + real lift) /
task-carried (interpretable, no lift).
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts",
                       "moabb_identity_leaderboard.py")


def _load():
    spec = importlib.util.spec_from_file_location("_mil", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


VERDICT = _load()._verdict


def test_below_gate_is_no_transfer_not_trap():
    # BNCI2014-001 per-trial: raw 0.537, free 0.539, not interpretable.
    assert VERDICT(0.537, 0.539, False) == "no-transfer"
    # Even with free >= raw (which the old logic called TRAP), below-gate is not.
    assert VERDICT(0.52, 0.99, False) == "no-transfer"


def test_interpretable_with_real_lift_is_trap():
    # Pooled BNCI2014-001: raw 0.667, free 0.963, interpretable → genuine (for
    # whatever metric produced it) recovery of masked skill.
    assert VERDICT(0.667, 0.963, True) == "TRAP"


def test_interpretable_without_lift_is_task_carried():
    assert VERDICT(0.78, 0.78, True) == "task-carried"
    # A trivially small lift under the epsilon is not a trap.
    assert VERDICT(0.78, 0.79, True) == "task-carried"


def test_nan_is_not_applicable():
    assert VERDICT(float("nan"), 0.5, True) == "n/a"


def test_interpretable_accepts_string_truthy():
    # Rows re-read from CSV carry "True"/"False" strings.
    assert VERDICT(0.667, 0.963, "True") == "TRAP"
    assert VERDICT(0.667, 0.963, "False") == "no-transfer"
