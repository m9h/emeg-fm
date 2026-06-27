"""Cross-modal spectrum (EEG↔structural shared subspace) — the E4 core, vendored into emeg-fm so the
volume-conduction analysis is self-contained. Mirrors `wwj/benchmarks/zeta_law/e4_cross_modal.py`.

The canonical correlations ρ₁≥ρ₂≥…∈[0,1] of the whitened cross-covariance measure how strongly the two
modalities co-vary, mode by mode; strong ρ = a shared mode (structure–function coupling). Pass
`covariate=age` to residualize out the age-driven / volume-conduction term and expose coupling beyond
conduction. Pure numpy.
"""
from __future__ import annotations

import numpy as np

from denoise import denoise_whiten


def _residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, float)
    if C.ndim == 1:
        C = C[:, None]
    Cb = np.column_stack([np.ones(len(C)), C])
    return X - Cb @ (np.linalg.pinv(Cb) @ X)


def _whiten(X: np.ndarray, reg: float) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / len(Xc)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    return Xc @ (V @ np.diag(1.0 / np.sqrt(w + reg * float(w.max()))) @ V.T)


def _w(X: np.ndarray, reg: float, denoise: bool) -> np.ndarray:
    """Whitener: Donoho spiked-model denoising (parameter-free) or the `reg`-loaded Tikhonov whitener."""
    return denoise_whiten(X) if denoise else _whiten(X, reg)


def cross_modal_spectrum(Xa: np.ndarray, Xb: np.ndarray, reg: float = 1e-3,
                         covariate: np.ndarray | None = None, denoise: bool = False) -> np.ndarray:
    """Canonical correlations between Xa, Xb (descending, ∈[0,1]); `covariate` (e.g. age) residualized
    out of both first if given. `denoise=True` uses Donoho spiked-model whitening (parameter-free, lower
    high-d bias) instead of the `reg`-loaded whitener."""
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    if covariate is not None:
        Xa, Xb = _residualize(Xa, covariate), _residualize(Xb, covariate)
    M = (_w(Xa, reg, denoise).T @ _w(Xb, reg, denoise)) / len(Xa)
    return np.clip(np.sort(np.linalg.svd(M, compute_uv=False))[::-1], 0.0, 1.0)


def permutation_null(Xa: np.ndarray, Xb: np.ndarray, n_perm: int = 1000, reg: float = 1e-3,
                     covariate: np.ndarray | None = None, seed: int = 0, denoise: bool = False) -> dict:
    """Permutation null for the **top** canonical correlation, needed because CCA at high d is upward
    biased: with d≈n even unrelated modalities show large ρ₁, so the absolute value is uninterpretable
    without a null. Shuffle the Xa↔Xb subject pairing `n_perm` times and recompute ρ₁.

    Whitening is row-order invariant (a permutation leaves the covariance unchanged), so we whiten once
    and only permute rows of the whitened Xb — the null costs one SVD of the cross-covariance per draw.
    `covariate` (e.g. age) is residualized from both *before* permuting, matching the observed statistic.
    Returns observed ρ₁, the null mean / 95th pct (the bias floor), and a one-sided p-value."""
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    if covariate is not None:
        Xa, Xb = _residualize(Xa, covariate), _residualize(Xb, covariate)
    n = len(Xa)
    Wa, Wb = _w(Xa, reg, denoise), _w(Xb, reg, denoise)
    top = lambda B: float(np.linalg.svd(Wa.T @ B / n, compute_uv=False)[0])
    obs = top(Wb)
    rng = np.random.default_rng(seed)
    null = np.array([top(Wb[rng.permutation(n)]) for _ in range(n_perm)])
    return {"observed": obs, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + np.sum(null >= obs)) / (1 + n_perm)), "n_perm": n_perm}


def shared_subspace_summary(rho: np.ndarray, thresh: float = 0.5) -> dict:
    rho = np.asarray(rho, float)
    pr = (rho.sum() ** 2) / (np.sum(rho ** 2) + 1e-30) if rho.size else 0.0
    return {"n_strong": int(np.sum(rho > thresh)), "participation_ratio": float(pr),
            "top": float(rho[0]) if rho.size else float("nan"),
            "mean": float(rho.mean()) if rho.size else float("nan")}
