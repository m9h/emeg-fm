"""Compact, cross-subject-comparable summary of a SimNIBS EEG leadfield (tier-3).

A raw GM-volume leadfield is (n_elec × ~1M tets × 3) ≈ 2 GB/subject — ~3.7 TB over the n≈1534 cohort,
so it cannot be stored at scale. This reduces each leadfield to a fixed-length per-subject **descriptor**
(a few thousand floats) that the cohort-scale tier-3 analysis can actually hold:

* `gain` — per-electrode RMS field magnitude over GM (len = n_elec). The overall coupling strength of
  each electrode to cortex; head size / skull thickness (age-dependent) move it. Absolute scale, so it
  carries the head-geometry effect we care about.
* `descriptor` — for each electrode, its GM field magnitude block-pooled into a normalized-bbox grid
  (len = n_elec · ∏grid). Captures the *spatial shape* of each electrode's sensitivity, comparable
  across subjects because the per-subject GM bounding box is mapped to [0,1]³ first.

`block_pool_field` is a pure-numpy core (unit-tested); `leadfield_descriptor` is the h5py loader.
"""
from __future__ import annotations

import numpy as np


def block_pool_field(mag: np.ndarray, centroids: np.ndarray, grid=(4, 4, 4)) -> np.ndarray:
    """Block-pool per-element field magnitudes into a normalized-bbox grid, per electrode.

    mag: (n_elec, n_elem) field magnitude of each electrode at each GM element.
    centroids: (n_elem, 3) element centroid coordinates (subject space).
    Returns (n_elec, prod(grid)) mean magnitude per grid cell (empty cells → 0). The bbox is normalized
    per subject ([0,1]³) so the descriptor compares spatial *shape* across heads of different sizes."""
    mag = np.asarray(mag, float)
    c = np.asarray(centroids, float)
    g = np.asarray(grid, int)
    lo, hi = c.min(0), c.max(0)
    nc = (c - lo) / (hi - lo + 1e-12)
    idx = np.clip((nc * g).astype(int), 0, g - 1)
    flat = (idx[:, 0] * g[1] + idx[:, 1]) * g[2] + idx[:, 2]      # row-major cell index
    ncells = int(g.prod())
    counts = np.bincount(flat, minlength=ncells).astype(float)
    out = np.empty((mag.shape[0], ncells))
    for e in range(mag.shape[0]):
        out[e] = np.bincount(flat, weights=mag[e], minlength=ncells)
    counts[counts == 0] = 1.0
    return out / counts[None, :]


def leadfield_descriptor(hdf5_path: str, grid=(4, 4, 4)) -> dict:
    """Load a SimNIBS TDCSLEADFIELD HDF5 (GM-volume ROI) and return its compact descriptor.

    Returns dict: descriptor (n_elec·∏grid,), gain (n_elec,), electrode_names, n_tet, grid. The raw
    (n_elec × n_tet × 3) field is read once and dropped — only the summary is returned/stored."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        d = f["mesh_leadfield/leadfields/tdcs_leadfield"]
        L = d[:]                                                  # (n_elec, n_tet, 3)
        names = list(d.attrs.get("electrode_names", []))
        nodes = f["mesh_leadfield/nodes/node_coord"][:]
        nnl = f["mesh_leadfield/elm/node_number_list"][:]         # (n_tet, 4), 1-indexed
    mag = np.linalg.norm(L, axis=2)                               # (n_elec, n_tet)
    centroids = nodes[nnl - 1].mean(axis=1)                       # (n_tet, 3)
    desc = block_pool_field(mag, centroids, grid)
    gain = np.sqrt((mag ** 2).mean(axis=1))                       # per-electrode RMS gain
    return {"descriptor": desc.ravel(), "gain": gain, "n_tet": int(mag.shape[1]),
            "electrode_names": [n.decode() if isinstance(n, bytes) else str(n) for n in names],
            "grid": tuple(int(x) for x in grid)}
