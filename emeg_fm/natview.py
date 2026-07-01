"""Primitives for within-subject EEG->BOLD prediction on natview (simultaneous EEG-fMRI).

Pure, unit-tested building blocks (see tests/test_natview.py); the orchestration that
loads the .set / mcflirt-corrects the BOLD / runs the ridge + null lives in
scripts/natview_eeg_to_bold.py and composes these.
"""
from __future__ import annotations

import numpy as np


def roi_timecourse(bold, mask):
    """Mean BOLD timecourse over a boolean ROI mask.

    bold: (X, Y, Z, T) array; mask: (X, Y, Z) bool -> (T,) mean over masked voxels.
    """
    bold = np.asarray(bold, float)
    mask = np.asarray(mask, bool)
    return bold[mask].mean(axis=0)


def brain_mask(bold, frac=0.15):
    """Crude brain mask: mean-over-time intensity above ``frac * max``."""
    m = np.asarray(bold, float).mean(axis=3)
    return m > frac * m.max()


def regress_confounds(y, confounds):
    """OLS residuals of ``y`` on [intercept, confounds] (removes drift / motion nuisance)."""
    y = np.asarray(y, float)
    C = np.asarray(confounds, float)
    if C.ndim == 1:
        C = C[:, None]
    design = np.column_stack([np.ones(len(y)), C])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def scan_onset_sample(onsets, values, trigger, sfreq):
    """EEG sample index of the FIRST event whose ``value == trigger`` (scan onset).

    Raises ValueError if the trigger label is absent.
    """
    onsets = np.asarray(onsets, float)
    values = np.asarray(values)
    idx = np.where(values == trigger)[0]
    if idx.size == 0:
        raise ValueError(f"trigger {trigger!r} not found in events")
    return int(round(float(onsets[idx[0]]) * sfreq))


def hrf(tr, length=32.0):
    """Canonical double-gamma HRF sampled at ``tr`` (peaks ~5 s, returns to baseline)."""
    from scipy.stats import gamma
    t = np.arange(0, length, tr)
    return gamma.pdf(t, 6) - 0.35 * gamma.pdf(t, 16)


def bin_to_tr(x, sfreq, tr, n_tr):
    """Bin a per-sample signal into up to ``n_tr`` TR-length windows (mean per window)."""
    x = np.asarray(x, float)
    spt = int(round(tr * sfreq))
    usable = min(len(x) // spt, n_tr) * spt
    return x[:usable].reshape(-1, spt).mean(axis=1)


def occipital_mask(mask, affine, frac=0.33):
    """Posterior ``frac`` of a brain mask (visual-cortex heuristic), orientation-robust.

    Uses the affine to work in world space, where posterior = low world-Y (RAS), so it is
    correct regardless of how the A-P axis maps onto the voxel grid.
    """
    mask = np.asarray(mask, bool)
    aff = np.asarray(affine, float)
    idx = np.array(np.where(mask)).T                       # (n_vox, 3) voxel coords
    world = idx @ aff[:3, :3].T + aff[:3, 3]               # -> world coords
    y = world[:, 1]
    keep = y <= y.min() + frac * (y.max() - y.min())
    out = np.zeros_like(mask)
    kept = idx[keep]
    out[kept[:, 0], kept[:, 1], kept[:, 2]] = True
    return out


def select_occipital_channels(ch_names):
    """Indices of occipital / parieto-occipital EEG channels (O1/O2/Oz/PO*)."""
    import re
    pat = re.compile(r"^(O[0-9z]|PO[0-9z]+)$", re.I)
    return [i for i, c in enumerate(ch_names) if pat.match(str(c))]
