"""Shared loader for braindecode BaseConcatDataset Hugging Face Hub releases.

Arithmetic (Zyma 2019), BCI Comp 2020 Track 3 (imagined speech), and MDD
(Mumtaz 2016) all use the SAME schema: `BaseConcatDataset.pull_from_hub(hf_id)`
returns a concatenation of per-recording `WindowsDataset`s, each iterable as
`ds[i] -> (X (C,T) float32, y int, metainfo)`, with the source subject id at
`windows_dataset.description["subject"]`. One shared grouping helper instead
of reimplementing the same iteration three times.
"""
from __future__ import annotations

import numpy as np


def pull(hf_id, cache_dir="/mnt/t9/hf"):
    from braindecode.datasets import BaseConcatDataset
    return BaseConcatDataset.pull_from_hub(hf_id, cache_dir=cache_dir)


def subject_of(windows_dataset):
    return str(windows_dataset.description.get("subject"))


def group_by_subject(per_recording_windows):
    """[(subject_id, [(X, y), ...]), ...] -> {subject_id: (X (n,C,T) f32, y (n,) int)}.
    Multiple recordings for the same subject are concatenated. Pure function
    over already-extracted (X, y) pairs so it's testable without a real
    braindecode WindowsDataset."""
    by_subj = {}
    for subj, windows in per_recording_windows:
        if not windows:
            continue
        Xarr = np.stack([np.asarray(x, dtype=np.float32) for x, _ in windows])
        yarr = np.asarray([int(y) for _, y in windows], dtype=int)
        if subj in by_subj:
            oX, oy = by_subj[subj]
            by_subj[subj] = (np.concatenate([oX, Xarr]), np.concatenate([oy, yarr]))
        else:
            by_subj[subj] = (Xarr, yarr)
    return by_subj


def concat_dataset_to_subject_groups(concat_ds):
    """Real braindecode BaseConcatDataset -> {subject_id: (X, y)} via
    group_by_subject. Kept separate from group_by_subject so the latter
    stays testable without braindecode/torch."""
    per_recording = []
    for wd in concat_ds.datasets:
        subj = subject_of(wd)
        windows = [(wd[i][0], wd[i][1]) for i in range(len(wd))]
        per_recording.append((subj, windows))
    return group_by_subject(per_recording)


def subject_splits(subject_ids, seed_frac=(0.7, 0.15, 0.15)):
    subs = sorted(subject_ids)
    n = len(subs)
    n_dv = max(1, int(n * seed_frac[1])) if n >= 3 else 0
    n_ev = max(1, int(n * seed_frac[2])) if n >= 3 else 0
    n_tr = n - n_dv - n_ev
    return set(subs[:n_tr]), set(subs[n_tr:n_tr + n_dv]), set(subs[n_tr + n_dv:])


def stratified_subject_splits(subject_to_label, seed_frac=(0.7, 0.15, 0.15)):
    """Like subject_splits, but split WITHIN each label group first, then
    union -- required whenever subject id correlates with class (e.g. MDD's
    "HS*"/"MDDS*" naming), where naive lexicographic sort-and-slice can put
    an entire class into one split (single-class eval -> degenerate kappa).
    ``subject_to_label``: {subject_id: label} (one label per subject)."""
    by_label = {}
    for subj, label in subject_to_label.items():
        by_label.setdefault(label, []).append(subj)
    tr, dv, ev = set(), set(), set()
    for label, subs in by_label.items():
        t, d, e = subject_splits(subs, seed_frac)
        tr |= t; dv |= d; ev |= e
    return tr, dv, ev
