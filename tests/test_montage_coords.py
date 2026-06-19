"""Tests for montage-flexible 3D channel coordinates (for LuMamba's encode()).

LuMamba.encode(x, channel_locations) takes per-channel 3D xyz coords (LUNA is
topology-agnostic). MOABB datasets use 10-20 names (standard_1005); HBN uses EGI
GSN-HydroCel labels (E1..E128). `channel_coords3d` resolves each label across a
montage list, returning the kept indices + coords and dropping the unresolvable.
"""
from __future__ import annotations

import numpy as np

from emeg_fm.montage import channel_coords3d


def test_ten_twenty_names_resolve_via_standard_1005():
    keep, coords = channel_coords3d(["Fz", "C3", "Pz", "Cz"])
    assert keep == [0, 1, 2, 3]
    assert coords.shape == (4, 3)
    assert np.isfinite(coords).all()


def test_egi_names_resolve_via_gsn_hydrocel():
    keep, coords = channel_coords3d(["E1", "E50", "E128"])
    assert keep == [0, 1, 2]
    assert coords.shape == (3, 3)


def test_unresolvable_labels_are_dropped_with_aligned_indices():
    keep, coords = channel_coords3d(["Fz", "NOTACHAN", "C3"])
    assert keep == [0, 2]                 # the bogus middle label dropped
    assert coords.shape == (2, 3)


def test_legacy_alias_resolves():
    # T3 is the legacy name for T7 (applied via canonicalize before lookup).
    keep, coords = channel_coords3d(["T3"])
    assert keep == [0] and coords.shape == (1, 3)
