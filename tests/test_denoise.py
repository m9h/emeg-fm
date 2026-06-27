"""TDD for emeg_fm.denoise — Gavish–Donoho optimal thresholding + spiked-model whitening (pure numpy)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from denoise import (_gd_lambda, optimal_hard_threshold, recover_spike,      # noqa: E402
                     denoise_whiten)
from cross_modal import cross_modal_spectrum                                  # noqa: E402


def test_gd_lambda_square_is_4_over_sqrt3():
    assert abs(_gd_lambda(1.0) - 4 / np.sqrt(3)) < 1e-9                       # the famous 4/√3


def test_optimal_hard_threshold_recovers_planted_rank():
    rng = np.random.default_rng(0)
    n, p, r = 800, 200, 5
    U = rng.standard_normal((n, r)); Vt = rng.standard_normal((r, p))
    X = (U * np.array([40, 30, 20, 12, 7])) @ Vt + rng.standard_normal((n, p))   # noise σ=1
    sv = np.linalg.svd(X, compute_uv=False)
    _, rank = optimal_hard_threshold(sv, X.shape)
    assert rank == r                                                         # parameter-free rank = planted


def test_optimal_hard_threshold_pure_noise_is_zero():
    rng = np.random.default_rng(1)
    sv = np.linalg.svd(rng.standard_normal((400, 120)), compute_uv=False)
    _, rank = optimal_hard_threshold(sv, (400, 120))
    assert rank == 0                                                         # no signal → keep nothing


def test_bbp_spike_round_trip():
    g = 0.25
    for ell in (3.0, 6.0, 12.0):
        lam = ell * (1 + g / (ell - 1))                                      # forward BBP map
        back, c2 = recover_spike(lam, g)
        assert abs(back - ell) < 1e-6 and 0.0 < c2 <= 1.0
    assert recover_spike(1.0, g)[0] == 1.0                                   # below edge → passthrough, c2=0
    assert recover_spike(1.0, g)[1] == 0.0


def test_denoise_whiten_keeps_signal_drops_noise():
    rng = np.random.default_rng(2)
    n, p, r = 600, 100, 4
    Z = rng.standard_normal((n, r)) * np.array([20, 15, 10, 7])              # r strong signal directions
    B = np.linalg.qr(rng.standard_normal((p, r)))[0]                          # orthonormal loadings
    Xw = denoise_whiten(Z @ B.T + rng.standard_normal((n, p)))
    assert np.isfinite(Xw).all()
    assert np.linalg.matrix_rank(Xw, tol=1e-6) <= r + 1                       # rank-reduced to ≈ signal rank
    Xn = rng.standard_normal((600, 100))                                     # pure noise -> bulk discarded
    assert np.linalg.norm(denoise_whiten(Xn)) < 0.5 * np.linalg.norm(Xn)


def test_denoise_reduces_highd_cca_bias():
    # independent high-d modalities (d≈n) -> CCA ρ₁ is spuriously inflated; denoising must lower it
    rng = np.random.default_rng(3)
    Xa, Xb = rng.standard_normal((200, 150)), rng.standard_normal((200, 150))
    reg_top = cross_modal_spectrum(Xa, Xb, denoise=False)[0]
    dn_top = cross_modal_spectrum(Xa, Xb, denoise=True)[0]
    assert reg_top > 0.4                                                     # confirms the inflation exists
    assert dn_top < reg_top                                                  # denoising shrinks the bias floor


if __name__ == "__main__":
    for _fn in (test_gd_lambda_square_is_4_over_sqrt3,
                test_optimal_hard_threshold_recovers_planted_rank,
                test_optimal_hard_threshold_pure_noise_is_zero,
                test_bbp_spike_round_trip,
                test_denoise_whiten_keeps_signal_drops_noise,
                test_denoise_reduces_highd_cca_bias):
        _fn(); print(f"PASS  {_fn.__name__}")
    print("all denoise tests passed")
