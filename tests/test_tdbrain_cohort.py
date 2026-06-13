"""Tests for the TDBRAIN trait-cohort label parsing (no EEG stack needed)."""

from __future__ import annotations

import textwrap

from emeg_fm.tdbrain_cohort import _sid_to_int, load_trait_labels


def _write_tsv(tmp_path, body: str) -> str:
    p = tmp_path / "participants.tsv"
    p.write_text(textwrap.dedent(body).lstrip("\n"))
    return str(p)


def test_sid_to_int():
    assert _sid_to_int("sub-19681349") == 19681349
    assert _sid_to_int("19681349") == 19681349
    assert _sid_to_int("  sub-42 ") == 42
    assert _sid_to_int("n/a") is None
    assert _sid_to_int("") is None


def test_load_trait_labels_filters_and_indexes(tmp_path):
    tsv = _write_tsv(tmp_path, """
        participant_id\tindication\tformal Dx\tage
        sub-1\tMDD\tMDD\t40
        sub-2\tADHD\tn/a\t12
        sub-3\tSMC\tn/a\t66
        sub-4\tmdd\tn/a\t55
        sub-5\tBURNOUT\tn/a\t30
    """)
    labels = load_trait_labels(tsv, classes=("MDD", "ADHD"))
    # class index follows classes order: MDD=0, ADHD=1; non-target dx dropped;
    # matching is case-insensitive (sub-4 "mdd").
    assert labels == {1: 0, 2: 1, 4: 0}


def test_load_trait_labels_first_session_row_wins(tmp_path):
    # TDBRAIN has one row per session; the first (ses-1) row must win.
    tsv = _write_tsv(tmp_path, """
        participant_id\tindication\tage
        sub-7\tADHD\t10
        sub-7\tADHD\t11
        sub-8\tMDD\t44
    """)
    labels = load_trait_labels(tsv, classes=("MDD", "ADHD"))
    assert labels == {7: 1, 8: 0}
    assert len(labels) == 2  # sub-7 counted once


def test_load_trait_labels_custom_classes_and_column(tmp_path):
    tsv = _write_tsv(tmp_path, """
        participant_id\tindication\tformal Dx
        sub-1\tMDD\tMDD
        sub-2\tn/a\tHC
        sub-3\tn/a\tMDD
    """)
    # contrast on the sparse "formal Dx" column, MDD vs HC; order sets indices.
    labels = load_trait_labels(tsv, classes=("HC", "MDD"), label_col="formal Dx")
    assert labels == {1: 1, 2: 0, 3: 1}
