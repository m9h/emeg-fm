"""Tests for montage presets + the headless MNE validator (emeg_fm.montage).

The validator is the live-session pre-flight: every Emotiv label must resolve
to MNE standard_1020 geometry (the system reve-positions derive from) before we
spend a GPU + gated model on the session. These tests need only MNE.
"""
import numpy as np
import pytest

from emeg_fm.montage import (
    EMOTIV_EPOCFLEX32, canonicalize, get_preset, validate_montage,
)

mne = pytest.importorskip("mne")


def test_epocflex32_has_32_unique_channels():
    assert len(EMOTIV_EPOCFLEX32) == 32
    assert len(set(EMOTIV_EPOCFLEX32)) == 32


def test_epocflex32_all_resolve_in_standard_1020():
    # The whole point of the preset: no label fails the pre-flight.
    resolved = validate_montage(EMOTIV_EPOCFLEX32)
    assert resolved == EMOTIV_EPOCFLEX32       # already canonical names


def test_get_preset_case_and_separator_insensitive():
    assert get_preset("EPOCFLEX32") == EMOTIV_EPOCFLEX32
    assert get_preset("epoc_flex-32") == EMOTIV_EPOCFLEX32
    with pytest.raises(KeyError):
        get_preset("nonsuch")


def test_legacy_aliases_canonicalized():
    assert canonicalize("T3") == "T7"
    assert canonicalize("T4") == "T8"
    assert canonicalize("T5") == "P7"
    assert canonicalize("  t6 ") == "P8"
    assert canonicalize("Cz") == "Cz"         # untouched


def test_validator_accepts_legacy_names_and_returns_canonical():
    resolved = validate_montage(["T3", "T4", "Cz"])
    assert resolved == ["T7", "T8", "Cz"]


def test_validator_case_insensitive():
    assert validate_montage(["cz", "fp1", "OZ"]) == ["Cz", "Fp1", "Oz"]


def test_strict_raises_on_unknown_label():
    with pytest.raises(ValueError) as exc:
        validate_montage(["Cz", "BOGUS", "Fz"])
    assert "BOGUS" in str(exc.value)


def test_nonstrict_reports_unresolved():
    resolved, unresolved = validate_montage(
        ["Cz", "BOGUS", "Fz"], strict=False)
    assert resolved == ["Cz", "Fz"]
    assert unresolved == ["BOGUS"]
