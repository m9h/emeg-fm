"""Resting-EEG treatment-response biomarkers (Arns / Leuchter-Cook lineage).

Generic band-power carries no antidepressant/rTMS treatment-response signal
(TDBRAIN-Challenge Tx track ran at chance). The MDD treatment-prediction
literature instead rests on a few *specific* resting-EEG markers:

- **Individual Alpha Peak Frequency (iAPF)** — the posterior alpha center-of-
  gravity; low iAPF is associated with poorer antidepressant/rTMS response
  (Arns et al.). Center-of-gravity over 7-13 Hz is more robust than argmax.
- **Prefrontal theta cordance** — Leuchter/Cook: combine each channel's absolute
  and relative theta power (each z-scored across the montage) and sum; baseline
  and early-change prefrontal theta cordance predict antidepressant response.
- **Frontal alpha asymmetry (FAA)** — log(right) - log(left) alpha at F4/F3 and
  F8/F7; linked to depression and treatment outcome.

Pure PSD math (functions take Welch ``pxx`` (n_ch, F), frequency vector ``f``,
and ``ch_names``) so they are unit-testable without an EEG stack. P300 is
deliberately absent: it needs the oddball task, not the resting recording that
the challenge replication set ships.
"""

from __future__ import annotations

import numpy as np

# TDBRAIN 10-20 scalp channel groups
POSTERIOR = ("O1", "Oz", "O2", "P3", "Pz", "P4", "P7", "P8")
PREFRONTAL = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8")
FRONTAL = ("F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4")
FAA_PAIRS = (("faa_f4f3", "F4", "F3"), ("faa_f8f7", "F8", "F7"))

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
ALPHA_SEARCH = (7.0, 13.0)  # extended alpha window for iAPF center-of-gravity


def band_power(pxx, f, lo, hi):
    """Mean PSD in [lo, hi) per channel -> (n_ch,)."""
    m = (f >= lo) & (f < hi)
    return pxx[:, m].mean(axis=1)


def _rows(pxx, ch_names, wanted):
    """Row indices of ``wanted`` channels that are present, preserving order."""
    idx = {c: i for i, c in enumerate(ch_names)}
    return [idx[c] for c in wanted if c in idx]


def individual_alpha_peak(pxx, f, ch_names, posterior=POSTERIOR):
    """(iAPF Hz, peak power) as the power-weighted center-of-gravity over 7-13 Hz
    on the mean posterior spectrum. Robust when no sharp peak exists."""
    rows = _rows(pxx, ch_names, posterior) or list(range(len(ch_names)))
    spec = pxx[rows].mean(axis=0)
    m = (f >= ALPHA_SEARCH[0]) & (f <= ALPHA_SEARCH[1])
    fa, pa = f[m], spec[m]
    cog = float((fa * pa).sum() / (pa.sum() + 1e-30))
    return cog, float(pa.max())


def frontal_alpha_asymmetry(pxx, f, ch_names):
    """log(right) - log(left) alpha power for F4/F3 and F8/F7 (0.0 if a channel
    is absent)."""
    alpha = band_power(pxx, f, *BANDS["alpha"])
    idx = {c: i for i, c in enumerate(ch_names)}
    out = {}
    for name, right, left in FAA_PAIRS:
        if right in idx and left in idx:
            out[name] = float(np.log(alpha[idx[right]] + 1e-30) - np.log(alpha[idx[left]] + 1e-30))
        else:
            out[name] = 0.0
    return out


def _zscore(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def theta_cordance(pxx, f, ch_names, region=PREFRONTAL):
    """Prefrontal theta cordance (Leuchter/Cook): per channel, z-score absolute
    theta power and relative theta power (relative = band/total over the 5 bands)
    across the whole montage, sum them, average over the region."""
    A = band_power(pxx, f, *BANDS["theta"])                       # absolute theta (n_ch,)
    total = np.stack([band_power(pxx, f, lo, hi) for lo, hi in BANDS.values()]).sum(axis=0)
    R = A / (total + 1e-30)                                        # relative theta
    cordance = _zscore(A) + _zscore(R)                            # per channel
    rows = _rows(pxx, ch_names, region)
    if not rows:
        return 0.0
    return float(cordance[rows].mean())


def tx_feature_vector(pxx, f, ch_names):
    """Assemble the treatment-response biomarker vector -> (values, names).

    Low-dimensional, montage-summary (not per-channel), so it is robust and does
    not swamp a small treatment cohort with dimensions."""
    iapf, ipow = individual_alpha_peak(pxx, f, ch_names)
    faa = frontal_alpha_asymmetry(pxx, f, ch_names)
    tc = theta_cordance(pxx, f, ch_names)

    pf = _rows(pxx, ch_names, PREFRONTAL) or list(range(len(ch_names)))
    fr = _rows(pxx, ch_names, FRONTAL) or list(range(len(ch_names)))
    post = _rows(pxx, ch_names, POSTERIOR) or list(range(len(ch_names)))

    def logband(rows, band):
        return float(np.log(band_power(pxx, f, *BANDS[band])[rows].mean() + 1e-30))

    fr_theta = band_power(pxx, f, *BANDS["theta"])[fr].mean()
    fr_beta = band_power(pxx, f, *BANDS["beta"])[fr].mean()

    feats = {
        "iapf": iapf,
        "iapf_logpow": float(np.log(ipow + 1e-30)),
        "prefrontal_theta_cordance": tc,
        "prefrontal_theta_logpow": logband(pf, "theta"),
        "frontal_alpha_logpow": logband(fr, "alpha"),
        "posterior_alpha_logpow": logband(post, "alpha"),
        "frontal_theta_beta_ratio": float(fr_theta / (fr_beta + 1e-30)),
        **faa,
    }
    names = list(feats)
    return np.array([feats[n] for n in names], dtype=np.float64), names
