"""FastSurfer/FreeSurfer DKT regional-volume morphometry → per-subject anatomy features for tier-1.

The tier-1 anatomy block is currently crude (block-pooled qsiprep GM-probseg + global FA/MD). FastSurfer's
`--seg_only` run (GPU, aarch64-native on the GB10) yields `aparc.DKTatlas+aseg` and a `.stats` file with
**per-region GM/subcortical volumes** — a richer, interpretable structural descriptor (DKT regions +
aseg structures) at no x86 cost. This parses those stats and aligns them into a feature matrix that drops
straight into `variance_partition` as the anatomy matrix `A` (or concatenated with the existing features).

Pure file-parse + numpy core, unit-tested; no FreeSurfer/nibabel needed (FastSurfer already computed the
volumes). See `scripts/build_morphometry.py`.
"""
from __future__ import annotations

import numpy as np


def parse_volume_stats(path: str) -> dict:
    """Parse a FreeSurfer/FastSurfer `.stats` table → {StructName: Volume_mm3}.

    Data rows are whitespace-delimited `Index SegId NVoxels Volume_mm3 StructName ...`; lines starting
    with `#` (the header/columns block) are skipped. Robust to extra trailing columns."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            if len(c) >= 5:
                try:
                    out[c[4]] = float(c[3])               # col 3 = Volume_mm3, col 4 = StructName
                except ValueError:
                    pass
    return out


def assemble_morphometry(stats_by_sub: dict, regions=None):
    """Align per-subject {region: volume} dicts → (X, regions, ids), rows = subjects.

    `regions` defaults to the sorted union across subjects; a region missing for a subject is filled 0.
    Passing an explicit `regions` list pins a canonical order (e.g. to match a training set)."""
    ids = list(stats_by_sub)
    if not ids:
        raise ValueError("no subjects")
    if regions is None:
        regions = sorted({r for d in stats_by_sub.values() for r in d})
    X = np.array([[stats_by_sub[s].get(r, 0.0) for r in regions] for s in ids], float)
    return X, list(regions), ids


def normalize_by_total(X: np.ndarray) -> np.ndarray:
    """Divide each subject's regional volumes by that subject's total (sum) → composition fractions,
    removing global head-size. Use when you want regional *shape* independent of overall size; omit to
    keep absolute volumes (which carry the head-size/age signal relevant to volume conduction)."""
    X = np.asarray(X, float)
    tot = X.sum(1, keepdims=True)
    return X / np.where(tot > 0, tot, 1.0)
