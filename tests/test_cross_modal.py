"""TDD for the vendored cross-modal core (emeg_fm.cross_modal) — mirrors the wwj E4 tests."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from cross_modal import cross_modal_spectrum, shared_subspace_summary, permutation_null   # noqa: E402


def _shared(n, da, db, strengths, seed, latent=None):
    rng = np.random.default_rng(seed)
    k = len(strengths)
    Z = rng.standard_normal((n, k)) if latent is None else latent
    Wa, Wb = rng.standard_normal((k, da)), rng.standard_normal((k, db))
    s = np.asarray(strengths)
    Xa = (Z * s) @ Wa + rng.standard_normal((n, da))
    Xb = (Z * s) @ Wb + rng.standard_normal((n, db))
    return Xa, Xb


def test_recovers_shared_dimension():
    Xa, Xb = _shared(800, 30, 25, [4, 4, 4, 0, 0, 0], seed=1)
    rho = cross_modal_spectrum(Xa, Xb)
    assert (rho[:3] > 0.7).all() and rho[3] < 0.5
    assert shared_subspace_summary(rho, 0.5)["n_strong"] == 3


def test_independent_modalities_low():
    rng = np.random.default_rng(2)
    rho = cross_modal_spectrum(rng.standard_normal((1000, 25)), rng.standard_normal((1000, 25)))
    assert rho[0] < 0.45


def test_residualize_age_removes_shared_age_mode():
    rng = np.random.default_rng(3)
    age = rng.standard_normal(800)
    Z = np.column_stack([age, rng.standard_normal(800), rng.standard_normal(800)])
    Xa, Xb = _shared(800, 30, 25, [5, 4, 4], seed=4, latent=Z)
    assert shared_subspace_summary(cross_modal_spectrum(Xa, Xb), 0.5)["n_strong"] == 3
    assert shared_subspace_summary(cross_modal_spectrum(Xa, Xb, covariate=age), 0.5)["n_strong"] == 2


def test_permutation_null_significant_when_coupled():
    # high-d so ρ₁ is upward biased (null mean well above 0) — the null must still flag real coupling
    Xa, Xb = _shared(300, 60, 55, [4, 4, 0, 0], seed=5)
    r = permutation_null(Xa, Xb, n_perm=200, seed=0)
    assert r["null_mean"] > 0.3                       # confirms the high-d upward bias is present
    assert r["observed"] > r["null_p95"]              # real coupling beats the bias floor
    assert r["p_value"] < 0.01


def test_permutation_null_not_significant_when_independent():
    rng = np.random.default_rng(6)
    Xa, Xb = rng.standard_normal((300, 60)), rng.standard_normal((300, 55))
    r = permutation_null(Xa, Xb, n_perm=200, seed=0)
    assert r["observed"] <= r["null_p95"] + 1e-9      # observed sits inside the null
    assert r["p_value"] > 0.05


if __name__ == "__main__":
    for _fn in (test_recovers_shared_dimension, test_independent_modalities_low,
                test_residualize_age_removes_shared_age_mode,
                test_permutation_null_significant_when_coupled,
                test_permutation_null_not_significant_when_independent):
        _fn(); print(f"PASS  {_fn.__name__}")
    print("all cross_modal tests passed")
