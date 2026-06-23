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

The ridge penalty is **selected per model by nested inner-CV**, and near-constant feature columns are
**pruned** before the fit. Both are load-bearing:

* A single fixed `alpha` under-regularises the joint model (d_E + d_A features) relative to the
  marginals, so its CV-R² falls *below* either marginal — the impossible `redundant_fraction > 1` /
  `eeg_unique < 0`. Per-model alpha keeps R²(E,A) ≥ max(R²(E), R²(A)).
* The block-pooled GM anatomy has ~half its columns near-constant (empty out-of-brain blocks). After
  per-column standardisation these become a huge noise tail (condition number ~1e35); GCV-style
  closed-form alpha then collapses to ~0 and the model overfits to a catastrophic negative R². Dropping
  the degenerate columns and choosing alpha by *measured held-out* error (inner CV, not a closed-form
  proxy) is robust to that spectrum.
"""
from __future__ import annotations

import numpy as np


def _standardize_prune(Xtr: np.ndarray, Xte: np.ndarray, rel_tol: float = 1e-6):
    """Z-score columns by train stats, dropping near-constant columns (sd ≤ rel_tol·max sd) — e.g. the
    empty out-of-brain GM-probseg blocks that otherwise blow up the condition number. No y → no leak."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > rel_tol * (sd.max() + 1e-30)
    mu, sd = mu[keep], sd[keep] + 1e-12
    return (Xtr[:, keep] - mu) / sd, (Xte[:, keep] - mu) / sd


def _ridge_path_pred(Xtr: np.ndarray, yc: np.ndarray, Xte: np.ndarray, alphas: np.ndarray):
    """Ridge predictions on Xte for every alpha, from one economy SVD of Xtr (yc = centred train y)."""
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    s2 = s * s
    d = U.T @ yc
    V = Vt.T
    return [Xte @ (V @ ((s / (s2 + a)) * d)) for a in alphas]


def _ridge_cv_r2(X: np.ndarray, y: np.ndarray, k: int = 5, alphas: np.ndarray | None = None,
                 seed: int = 0, k_inner: int = 5) -> float:
    """k-fold ridge-CV R² (out-of-fold), per-fold standardised + degenerate-column pruned, with the
    ridge alpha chosen by an **inner k-fold CV** on each outer-train fold (no leakage to the test fold).
    Default alpha grid is scaled to the design's mean eigenvalue so it brackets the useful range."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    yp = np.empty(n)
    for fold in np.array_split(idx, k):
        tr = np.setdiff1d(idx, fold, assume_unique=False)
        Xtr_s, Xte_s = _standardize_prune(X[tr], X[fold])
        yb = y[tr].mean()
        if alphas is not None:
            grid = np.asarray(alphas, float)
        else:                                          # scale grid to mean eigenvalue (≈ trace/rank)
            scale = (Xtr_s ** 2).sum() / max(Xtr_s.shape[1], 1)
            grid = np.logspace(-3, 5, 25) * scale
        # inner CV over the training fold to pick alpha by measured held-out SSE
        ii = rng.permutation(len(tr))
        sse_inner = np.zeros(len(grid))
        for jf in np.array_split(ii, k_inner):
            jtr = np.setdiff1d(np.arange(len(tr)), jf, assume_unique=False)
            Xj_tr, Xj_te = _standardize_prune(X[tr][jtr], X[tr][jf])
            yjb = y[tr][jtr].mean()
            preds = _ridge_path_pred(Xj_tr, y[tr][jtr] - yjb, Xj_te, grid)
            yjte = y[tr][jf]
            for m, p in enumerate(preds):
                sse_inner[m] += float(np.sum((yjte - (p + yjb)) ** 2))
        a = grid[int(np.argmin(sse_inner))]
        yp[fold] = _ridge_path_pred(Xtr_s, y[tr] - yb, Xte_s, [a])[0] + yb
    sse = float(np.sum((y - yp) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def variance_partition(E: np.ndarray, A: np.ndarray, y: np.ndarray,
                       k: int = 5, alphas: np.ndarray | None = None, seed: int = 0) -> dict:
    """Commonality partition of age (`y`) variance explained by EEG (`E`) vs anatomy (`A`).

    Rows are subjects (REVE per-subject embeddings), so k-fold over rows *is* subject-level CV —
    no pseudoreplication. Ridge alpha is GCV-selected per model. Returns the R²s, the commonality
    components, and the two fractions.
    """
    r2_e = _ridge_cv_r2(E, y, k, alphas, seed)
    r2_a = _ridge_cv_r2(A, y, k, alphas, seed)
    r2_ea = _ridge_cv_r2(np.hstack([np.asarray(E, float), np.asarray(A, float)]), y, k, alphas, seed)
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
