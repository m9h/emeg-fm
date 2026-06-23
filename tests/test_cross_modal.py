"""TDD for the vendored cross-modal core (emeg_fm.cross_modal) — mirrors the wwj E4 tests."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from cross_modal import cross_modal_spectrum, shared_subspace_summary   # noqa: E402


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


if __name__ == "__main__":
    for _fn in (test_recovers_shared_dimension, test_independent_modalities_low,
                test_residualize_age_removes_shared_age_mode):
        _fn(); print(f"PASS  {_fn.__name__}")
    print("all cross_modal tests passed")
