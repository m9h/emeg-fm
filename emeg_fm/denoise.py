"""Donoho-optimal denoising primitives — parameter-free replacements for hand-picked ranks/thresholds.

The high-d cross-modal CCA (cross_modal.py) whitens each modality with `1/sqrt(w + reg*w_max)`, `reg=1e-3`
hand-picked — an ad-hoc Tikhonov loading that amplifies exactly the Marchenko–Pastur noise eigenvalues and
inflates the canonical correlations (the upward bias the permutation null exposed). This module gives the
principled fix:

* `optimal_hard_threshold` — Gavish & Donoho (2014) "4/sqrt(3)" optimal hard threshold for singular values
  (known- and unknown-noise rules) → a parameter-free rank.
* `recover_spike` — BBP / Baik–Silverstein inverse spike map: the population eigenvalue behind a sample
  covariance eigenvalue, plus the eigenvector cos².
* `denoise_cov_eigs` / `denoise_whiten` — shrink the MP noise bulk to the estimated noise floor and debias
  the genuine spikes, then whiten by the denoised spectrum (no `reg`). Reduces the CCA bias floor.

Pure numpy, unit-tested. Shared with neurojax's jaxctrl denoising (same maths, two call sites).
Refs: Gavish & Donoho 2014; Donoho–Gavish–Johnstone 2018; Baik–Ben Arous–Péché / Baik–Silverstein.
"""
from __future__ import annotations

import numpy as np


def _gd_lambda(beta: float) -> float:
    """Gavish–Donoho optimal-hard-threshold coefficient λ(β); β = min/max matrix aspect ∈ (0,1].
    λ(1) = 4/√3 ≈ 2.309."""
    return float(np.sqrt(2 * (beta + 1) + 8 * beta / ((beta + 1) + np.sqrt(beta ** 2 + 14 * beta + 1))))


def _mp_median(beta: float) -> float:
    """Median μ_β of the Marchenko–Pastur law with ratio β (σ²=1), by numeric CDF inversion."""
    lo, hi = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
    x = np.linspace(max(lo, 1e-12), hi, 50000)
    dens = np.sqrt(np.clip((hi - x) * (x - lo), 0.0, None)) / (2 * np.pi * beta * x)
    cdf = np.cumsum(dens) * (x[1] - x[0])
    cdf /= cdf[-1]
    return float(x[int(np.searchsorted(cdf, 0.5))])


def optimal_hard_threshold(sv, shape, sigma: float | None = None):
    """Gavish–Donoho (2014) optimal hard threshold for singular values `sv` of a matrix of given `shape`
    (rows, cols). Returns (threshold, rank). `sigma=None` → unknown-noise rule (median-singular-value)."""
    sv = np.sort(np.asarray(sv, float))[::-1]
    m, n = sorted(shape)                                  # m <= n
    beta = m / n
    if sigma is not None:                                 # known noise: τ = λ(β)·√n·σ
        tau = _gd_lambda(beta) * np.sqrt(n) * sigma
    else:                                                 # unknown noise: τ = ω(β)·median(sv)
        tau = (_gd_lambda(beta) / np.sqrt(_mp_median(beta))) * float(np.median(sv))
    return float(tau), int((sv > tau).sum())


def recover_spike(lam: float, gamma: float):
    """Population spike ℓ and eigenvector cos² behind a sample-covariance eigenvalue `lam` (noise σ²=1),
    ratio γ=p/n, via the BBP map λ = ℓ(1 + γ/(ℓ−1)). Below the bulk edge (1+√γ)² → (lam, 0)."""
    edge = (1 + np.sqrt(gamma)) ** 2
    if lam <= edge:
        return float(lam), 0.0
    ell = (lam + 1 - gamma + np.sqrt((lam + 1 - gamma) ** 2 - 4 * lam)) / 2
    c2 = (1 - gamma / (ell - 1) ** 2) / (1 + gamma / (ell - 1))
    return float(ell), float(min(1.0, max(0.0, c2)))


def denoise_cov_eigs(w, gamma: float):
    """Spiked-model denoising of sample-covariance eigenvalues `w` (any order), ratio γ=p/n: estimate the
    noise floor σ² from the MP median, shrink bulk eigenvalues to σ², debias spikes to their population
    value. Returns denoised eigenvalues in the input order."""
    w = np.asarray(w, float)
    sigma2 = float(np.median(w)) / _mp_median(gamma)      # median(w) = σ²·μ_γ
    edge = sigma2 * (1 + np.sqrt(gamma)) ** 2
    out = np.full_like(w, sigma2)
    spike = w > edge
    for i in np.nonzero(spike)[0]:
        ell, _ = recover_spike(w[i] / sigma2, gamma)
        out[i] = sigma2 * ell
    return out


def denoise_whiten(X):
    """**Rank-reduced** whitening of X (n×p): keep only the spiked (signal) eigen-directions above the
    Marchenko–Pastur bulk edge, whiten them by their BBP-debiased eigenvalues, and **discard the noise
    bulk** — the Donoho-optimal replacement for the `1/sqrt(w + reg·w_max)` Tikhonov whitener. Cutting the
    noise bulk is what removes the high-d CCA inflation (reg keeps all p directions and amplifies them).
    Pure noise → no spikes → zeros (so spurious cross-modal coupling collapses to ~0)."""
    Xc = X - X.mean(0, keepdims=True)
    n, p = Xc.shape
    gamma = p / n
    w, V = np.linalg.eigh((Xc.T @ Xc) / n)
    order = np.argsort(w)[::-1]
    w, V = np.clip(w[order], 0.0, None), V[:, order]      # descending
    sigma2 = float(np.median(w)) / _mp_median(gamma)
    keep = w > sigma2 * (1 + np.sqrt(gamma)) ** 2         # spikes above the bulk edge
    if not keep.any():
        return np.zeros_like(Xc)
    ld = np.array([sigma2 * recover_spike(wi / sigma2, gamma)[0] for wi in w[keep]])
    Vk = V[:, keep]
    return Xc @ (Vk * (1.0 / np.sqrt(ld))) @ Vk.T        # whiten the signal subspace only
