"""LoRA-REVE HMC pilot: bounded train-subject selection stays deterministic
and disjoint-safe relative to the full split.
scripts/eegfm_t9.sh python -m pytest tests/test_lora_reve_hmc_pilot.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lora_reve_hmc_pilot as lp  # noqa: E402


def test_bounded_train_subjects_is_deterministic_first_n():
    subs = {"SN010", "SN002", "SN050", "SN001", "SN099"}
    got = lp.bounded_train_subjects(subs, 3)
    assert got == {"SN001", "SN002", "SN010"}


def test_bounded_train_subjects_returns_all_when_n_exceeds_size():
    subs = {"SN001", "SN002"}
    assert lp.bounded_train_subjects(subs, 20) == subs


def test_bounded_train_subjects_is_subset_of_input():
    subs = {f"SN{i:03d}" for i in range(1, 108)}
    got = lp.bounded_train_subjects(subs, 20)
    assert len(got) == 20
    assert got <= subs
