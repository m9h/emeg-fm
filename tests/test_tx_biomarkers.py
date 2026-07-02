"""Tests for resting-EEG treatment-response biomarkers (Arns/Leuchter lineage).

Pure PSD math (no EEG stack): individual alpha peak frequency, frontal alpha
asymmetry, prefrontal theta cordance. Synthetic PSDs with known structure.
"""

from __future__ import annotations

import numpy as np

from emeg_fm.tx_biomarkers import (
    band_power,
    frontal_alpha_asymmetry,
    individual_alpha_peak,
    theta_cordance,
    tx_feature_vector,
)


def _grid():
    return np.arange(1.0, 45.0, 0.25)


def _bump(f, center, width, amp):
    return amp * np.exp(-0.5 * ((f - center) / width) ** 2)


def test_band_power_flat_psd():
    f = _grid()
    pxx = np.ones((3, len(f)))
    bp = band_power(pxx, f, 8, 13)
    assert np.allclose(bp, 1.0, atol=1e-9)  # mean of a flat unit spectrum


def test_individual_alpha_peak_recovers_bump():
    f = _grid()
    ch = ["O1", "Oz", "O2", "P3", "Pz", "P4", "P7", "P8"]
    # posterior channels carry a 10.5 Hz alpha bump on a 1/f floor
    floor = 1.0 / f
    pxx = np.stack([floor + _bump(f, 10.5, 1.0, 5.0) for _ in ch])
    iapf, ipow = individual_alpha_peak(pxx, f, ch, ch)
    assert abs(iapf - 10.5) < 0.5
    assert ipow > 0


def test_frontal_alpha_asymmetry_sign():
    f = _grid()
    ch = ["F3", "F4", "F7", "F8"]
    idx = {c: i for i, c in enumerate(ch)}
    pxx = np.tile(1.0 / f, (4, 1))
    # more alpha power on the right (F4) than left (F3) -> positive asymmetry
    pxx[idx["F4"]] += _bump(f, 10.0, 1.5, 4.0)
    faa = frontal_alpha_asymmetry(pxx, f, ch)
    assert faa["faa_f4f3"] > 0.3


def test_theta_cordance_finite_and_responsive():
    f = _grid()
    ch = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C3", "Cz", "C4", "Pz", "Oz"]
    rng = np.random.RandomState(0)
    pxx = np.abs(rng.rand(len(ch), len(f))) + 1.0 / f
    base = theta_cordance(pxx, f, ch)
    # boost theta on prefrontal channels -> prefrontal theta cordance rises
    pf = [ch.index(c) for c in ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8")]
    pxx2 = pxx.copy()
    for i in pf:
        pxx2[i] += _bump(f, 6.0, 1.5, 8.0)
    boosted = theta_cordance(pxx2, f, ch)
    assert np.isfinite(base) and np.isfinite(boosted)
    assert boosted > base


def test_tx_feature_vector_shape_and_finite():
    f = _grid()
    ch = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "O1", "Oz", "O2", "P3", "Pz", "P4", "P7", "P8"]
    rng = np.random.RandomState(1)
    pxx = np.abs(rng.rand(len(ch), len(f))) + 1.0 / f
    vec, names = tx_feature_vector(pxx, f, ch)
    assert len(vec) == len(names)
    assert np.isfinite(vec).all()
    assert "iapf" in names and "prefrontal_theta_cordance" in names
