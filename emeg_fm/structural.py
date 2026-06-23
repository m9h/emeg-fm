"""Tier-1 structural feature extraction + EEG↔anatomy assembly for the volume-conduction analysis.

Pure-numpy cores (`block_pool`, `assemble`) are unit-tested; the nibabel/I-O helpers
(`map_features`, `reve_embeddings`) load the verified HBN paths and are exercised on the cluster run.
See `docs/volume_conduction_plan.md`.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np


# ---------------------------------------------------------------------------- pure cores (tested)
def block_pool(vol: np.ndarray, grid=(8, 8, 8)) -> np.ndarray:
    """Average-pool a 3-D volume into a `grid` of cells → flattened feature vector (len = prod(grid)).
    Atlas-free spatial summary; robust to any input shape (uses `array_split`, no divisibility req)."""
    vol = np.asarray(vol, float)
    xs, ys, zs = (np.array_split(np.arange(vol.shape[d]), grid[d]) for d in range(3))
    out = np.empty(grid)
    for i, xi in enumerate(xs):
        for j, yj in enumerate(ys):
            for k, zk in enumerate(zs):
                blk = vol[np.ix_(xi, yj, zk)] if xi.size and yj.size and zk.size else np.array([0.0])
                out[i, j, k] = float(blk.mean()) if blk.size else 0.0
    return out.ravel()


def assemble(eeg_ids, eeg_X: np.ndarray, ages: np.ndarray, anat_by_sub: dict):
    """Align EEG embeddings (rows = `eeg_ids`) with per-subject anatomy features, keeping subjects
    present in both. Returns (E, A, y, kept_ids)."""
    eeg_X, ages = np.asarray(eeg_X), np.asarray(ages, float)
    keep = [i for i, s in enumerate(eeg_ids) if s in anat_by_sub]
    if not keep:
        raise ValueError("no subjects in common between EEG embeddings and anatomy features")
    E = eeg_X[keep]
    y = ages[keep]
    A = np.vstack([np.asarray(anat_by_sub[eeg_ids[i]], float).ravel() for i in keep])
    return E, A, y, [eeg_ids[i] for i in keep]


# ---------------------------------------------------------------------------- I/O helpers (cluster)
def map_features(nii_path: str, grid=(8, 8, 8), mask_path: str | None = None) -> np.ndarray:
    """Load a scalar map (FA/MD/GM-probseg), optionally mask, block-pool → feature vector."""
    import nibabel as nib
    vol = np.asarray(nib.load(nii_path).dataobj, float)
    if mask_path:
        m = np.asarray(nib.load(mask_path).dataobj, float) > 0
        vol = np.where(m, vol, 0.0)
    return block_pool(vol, grid)


_HBN_EPO_GLOB = "/data/derivatives/brain_age/HBN_EEG/sub-*/eeg/*proc-autoreject_epo.fif"
_HBN_PARTICIPANTS = "/data/datasets/hbn-eeg/participants.tsv"
_SID_RE = re.compile(r"sub-([A-Za-z0-9]+)")


def reve_embeddings(npz: str = "/mnt/t9/reve_hbn_emb.npz",
                    epochs_glob: str = _HBN_EPO_GLOB,
                    participants: str = _HBN_PARTICIPANTS):
    """Load REVE EEG embeddings and reconstruct the (un-stored) subject-ID order, **replicating the
    producer** `scripts/reve_brain_age.py`: row order = `sorted(glob(epochs_glob))`, filtered to
    subjects whose regex-extracted id has a finite age. The length assertion is load-bearing — a silent
    mismatch would misalign every row, so we fail loudly rather than guess.
    """
    d = np.load(npz)
    X, ages = np.asarray(d["X"]), np.asarray(d["ages"], float)
    age = _participant_ages(participants)            # keys without the "sub-" prefix
    subs = []
    for f in sorted(glob.glob(epochs_glob)):
        m = _SID_RE.search(f)
        if m and m.group(1) in age and np.isfinite(age[m.group(1)]):
            subs.append("sub-" + m.group(1))
    if len(subs) != len(X):
        raise AssertionError(f"ID reconstruction mismatch: {len(subs)} epo-files-with-age vs {len(X)} "
                             f"rows — confirm epochs_glob / participants match the producer args")
    return subs, X, ages


def _participant_ages(tsv: str) -> dict:
    """Map bare participant id (no 'sub-') → finite age from a BIDS participants.tsv (tab-separated)."""
    out = {}
    with open(tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        si, ai = header.index("participant_id"), header.index("age")
        for line in f:
            c = line.rstrip("\n").split("\t")
            try:
                if c[ai] not in ("", "n/a", "NaN"):
                    out[c[si].replace("sub-", "")] = float(c[ai])
            except (ValueError, IndexError):
                pass
    return out
