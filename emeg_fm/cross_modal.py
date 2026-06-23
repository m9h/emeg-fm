"""Cross-modal spectrum (EEG↔structural shared subspace) — the E4 core, vendored into emeg-fm so the
volume-conduction analysis is self-contained. Mirrors `wwj/benchmarks/zeta_law/e4_cross_modal.py`.

The canonical correlations ρ₁≥ρ₂≥…∈[0,1] of the whitened cross-covariance measure how strongly the two
modalities co-vary, mode by mode; strong ρ = a shared mode (structure–function coupling). Pass
`covariate=age` to residualize out the age-driven / volume-conduction term and expose coupling beyond
conduction. Pure numpy.
"""
from __future__ import annotations

import numpy as np


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


def cross_modal_spectrum(Xa: np.ndarray, Xb: np.ndarray, reg: float = 1e-3,
                         covariate: np.ndarray | None = None) -> np.ndarray:
    """Canonical correlations between Xa, Xb (descending, ∈[0,1]); `covariate` (e.g. age) residualized
    out of both first if given."""
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
    if covariate is not None:
        Xa, Xb = _residualize(Xa, covariate), _residualize(Xb, covariate)
    M = (_whiten(Xa, reg).T @ _whiten(Xb, reg)) / len(Xa)
    return np.clip(np.sort(np.linalg.svd(M, compute_uv=False))[::-1], 0.0, 1.0)


def shared_subspace_summary(rho: np.ndarray, thresh: float = 0.5) -> dict:
    rho = np.asarray(rho, float)
    pr = (rho.sum() ** 2) / (np.sum(rho ** 2) + 1e-30) if rho.size else 0.0
    return {"n_strong": int(np.sum(rho > thresh)), "participation_ratio": float(pr),
            "top": float(rho[0]) if rho.size else float("nan"),
            "mean": float(rho.mean()) if rho.size else float("nan")}
