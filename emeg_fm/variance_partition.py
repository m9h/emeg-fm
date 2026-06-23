"""Tier-1 EEG↔anatomy variance partition (commonality analysis).

How much of EEG's age signal is **redundant with structural anatomy** (VBM/DWI features — i.e.
reproducible from structure alone, *consistent with* volume conduction) vs **unique to EEG** (not
linearly recoverable from anatomy — candidate neural signal)?

Method: subject-level k-fold ridge CV R² of age from EEG features (`E`), anatomy features (`A`), and
their concatenation, then Mood–Nimon commonality:

    redundant (common) = R²(E) + R²(A) − R²(E,A)      # shared predictive variance
    eeg_unique         = R²(E,A) − R²(A)              # age info only EEG has
    anat_unique        = R²(E,A) − R²(E)

`redundant_fraction = redundant / R²(E)` is the share of EEG's age signal that anatomy reproduces.

IMPORTANT (honest scope): this is a *correlational redundancy* split, not causal proof of conduction —
age is a common cause, so a neural age effect that merely tracks the same age as anatomy also reads as
"redundant." A near-zero `eeg_unique` is *consistent with* (not proof of) volume conduction; a large
`eeg_unique` is positive evidence of EEG-specific (functional/neural) age information. The causal
conduction test is the forward model (tier 3), which asks whether the anatomy-derived lead field
actually reproduces the EEG age effect.

Pure numpy (no sklearn/torch) so it runs under any numpy interpreter and is unit-testable in isolation.
"""
from __future__ import annotations

import numpy as np


def _ridge_cv_r2(X: np.ndarray, y: np.ndarray, k: int = 5, alpha: float = 10.0, seed: int = 0) -> float:
    """k-fold ridge-CV coefficient of determination (out-of-fold), per-fold standardised."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    yp = np.empty(n)
    for fold in np.array_split(idx, k):
        tr = np.setdiff1d(idx, fold, assume_unique=False)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[fold] - mu) / sd
        yb = y[tr].mean()
        w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (y[tr] - yb))
        yp[fold] = Xte @ w + yb
    sse = float(np.sum((y - yp) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def variance_partition(E: np.ndarray, A: np.ndarray, y: np.ndarray,
                       k: int = 5, alpha: float = 10.0, seed: int = 0) -> dict:
    """Commonality partition of age (`y`) variance explained by EEG (`E`) vs anatomy (`A`).

    Rows are subjects (REVE per-subject embeddings), so k-fold over rows *is* subject-level CV —
    no pseudoreplication. Returns the R²s, the commonality components, and the two fractions.
    """
    r2_e = _ridge_cv_r2(E, y, k, alpha, seed)
    r2_a = _ridge_cv_r2(A, y, k, alpha, seed)
    r2_ea = _ridge_cv_r2(np.hstack([np.asarray(E, float), np.asarray(A, float)]), y, k, alpha, seed)
    redundant = r2_e + r2_a - r2_ea
    eeg_unique = r2_ea - r2_a
    anat_unique = r2_ea - r2_e
    denom = r2_e if r2_e > 1e-6 else float("nan")
    return {
        "r2_eeg": r2_e, "r2_anat": r2_a, "r2_joint": r2_ea,
        "redundant": redundant, "eeg_unique": eeg_unique, "anat_unique": anat_unique,
        "redundant_fraction": redundant / denom,         # EEG age-signal reproducible from anatomy
        "eeg_unique_fraction": eeg_unique / denom,        # EEG age-signal anatomy cannot reproduce
    }
