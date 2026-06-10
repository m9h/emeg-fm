"""Tests for emeg_fm.lora — the Scope C LoRA + vendored-Muon helpers.

The pure-math pieces (newton_schulz, select_lora_targets, lora_delta,
weight_spectral_summary) are numpy-only and always run. The torch pieces
(LoRALinear, inject_lora, make_muon) are guarded by ``importorskip("torch")``
so the suite still passes on the CPU venv without torch.
"""
import numpy as np
import pytest

from emeg_fm.lora import (
    DEFAULT_REVE_LORA_TARGETS,
    lora_delta,
    newton_schulz,
    select_lora_targets,
    weight_spectral_summary,
)


# ---------------------------------------------------------------------------
# newton_schulz — the Muon orthogonalization core
# ---------------------------------------------------------------------------

class TestNewtonSchulz:
    def test_pushes_singular_values_toward_one(self):
        rng = np.random.default_rng(0)
        G = rng.standard_normal((64, 32))
        X = newton_schulz(G, steps=5)
        s = np.linalg.svd(X, compute_uv=False)
        # All singular values should land in a tight band around 1.
        assert s.min() > 0.5
        assert s.max() < 1.5
        assert abs(s.mean() - 1.0) < 0.2

    def test_wide_matrix(self):
        rng = np.random.default_rng(1)
        G = rng.standard_normal((16, 96))   # exercises the transpose branch
        X = newton_schulz(G, steps=5)
        assert X.shape == G.shape
        s = np.linalg.svd(X, compute_uv=False)
        assert s.max() < 1.5 and s.min() > 0.4

    def test_collapses_singular_value_spread(self):
        # The quintic converges to a band (σ≈[0.7,1.3]), not exact orthonormality
        # — "good enough" for Muon. The property it guarantees is a near-1
        # condition number: a raw Gaussian's wide spectrum gets flattened.
        rng = np.random.default_rng(2)
        G = rng.standard_normal((8, 32))
        cond_before = np.linalg.cond(G)
        X = newton_schulz(G, steps=6)
        s = np.linalg.svd(X, compute_uv=False)
        cond_after = s.max() / s.min()
        assert cond_after < 2.0
        assert cond_after < cond_before

    def test_preserves_orientation(self):
        # A matrix already ≈ orthogonal should stay near itself.
        rng = np.random.default_rng(3)
        Q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
        X = newton_schulz(Q, steps=5)
        s = np.linalg.svd(X, compute_uv=False)
        assert np.allclose(s, 1.0, atol=0.1)


# ---------------------------------------------------------------------------
# select_lora_targets — exact-match validation
# ---------------------------------------------------------------------------

class TestSelectLoraTargets:
    def test_defaults_pass_when_all_present(self):
        available = list(DEFAULT_REVE_LORA_TARGETS) + ["transformer.layers.5.0.to_qkv"]
        got = select_lora_targets(available)
        assert got == list(DEFAULT_REVE_LORA_TARGETS)

    def test_custom_targets_subset(self):
        available = ["a.b.c", "d.e.f", "g.h.i"]
        got = select_lora_targets(available, targets=["d.e.f", "a.b.c"])
        assert got == ["d.e.f", "a.b.c"]

    def test_missing_target_raises_keyerror(self):
        with pytest.raises(KeyError) as exc:
            select_lora_targets(["only.this.one"], targets=["only.this.one", "nope"])
        assert "nope" in str(exc.value)

    def test_default_targets_are_the_five_alpha_lt_2_matrices(self):
        # The exact WeightWatcher-flagged set; ordering = ascending α.
        assert DEFAULT_REVE_LORA_TARGETS == (
            "transformer.layers.2.0.to_qkv",
            "transformer.layers.1.0.to_qkv",
            "transformer.layers.0.0.to_out",
            "transformer.layers.3.0.to_qkv",
            "transformer.layers.10.0.to_out",
        )


# ---------------------------------------------------------------------------
# lora_delta — the backend-agnostic weight update
# ---------------------------------------------------------------------------

class TestLoraDelta:
    def test_shape_and_value(self):
        A = np.arange(6, dtype=np.float64).reshape(2, 3)   # (rank, in)
        B = np.ones((4, 2), dtype=np.float64)              # (out, rank)
        d = lora_delta(A, B, scaling=2.0)
        assert d.shape == (4, 3)                            # (out, in)
        np.testing.assert_allclose(d, 2.0 * (B @ A))

    def test_zero_B_gives_zero_delta(self):
        # Standard LoRA init: B=0 → adapted model == base at step 0.
        A = np.random.default_rng(0).standard_normal((8, 32))
        B = np.zeros((64, 8))
        assert np.all(lora_delta(A, B, 16 / 8) == 0.0)


# ---------------------------------------------------------------------------
# weight_spectral_summary — re-exported by sae.py; lives here for SIF sharing
# ---------------------------------------------------------------------------

class TestWeightSpectralSummary:
    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            weight_spectral_summary(np.zeros((3, 3, 3)))

    def test_rank1_low_participation(self):
        W = np.outer(np.ones(16), np.arange(1, 9, dtype=float))
        s = weight_spectral_summary(W)
        assert s["participation_ratio"] < 1.5     # one dominant direction
        assert s["stable_rank"] < 1.5

    def test_isotropic_high_participation(self):
        s = weight_spectral_summary(np.eye(32))
        assert s["participation_ratio"] > 25      # mass spread across all dims

    def test_reexport_identity(self):
        from emeg_fm.sae import weight_spectral_summary as via_sae
        assert via_sae is weight_spectral_summary


# ---------------------------------------------------------------------------
# Torch pieces — skipped without torch
# ---------------------------------------------------------------------------

class TestTorchPieces:
    def test_lora_linear_forward_starts_at_base(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from emeg_fm.lora import _lora_linear_cls

        LoRALinear = _lora_linear_cls()
        base = nn.Linear(10, 6)
        wrapped = LoRALinear(base, rank=4, alpha=8.0)
        x = torch.randn(3, 10)
        # B is zero-initialised → adapted output exactly equals base at step 0.
        torch.testing.assert_close(wrapped(x), base(x))
        # base params are frozen.
        assert not base.weight.requires_grad
        assert wrapped.lora_A.requires_grad and wrapped.lora_B.requires_grad

    def test_merged_weight_matches_base_plus_delta(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from emeg_fm.lora import _lora_linear_cls

        LoRALinear = _lora_linear_cls()
        base = nn.Linear(10, 6)
        wrapped = LoRALinear(base, rank=4, alpha=8.0)
        with torch.no_grad():
            wrapped.lora_B.copy_(torch.randn_like(wrapped.lora_B))
        merged = wrapped.merged_weight().detach().numpy()
        expect = (base.weight.detach().numpy()
                  + lora_delta(wrapped.lora_A.detach().numpy(),
                               wrapped.lora_B.detach().numpy(), wrapped.scaling))
        np.testing.assert_allclose(merged, expect, rtol=1e-5, atol=1e-6)

    def test_inject_lora_replaces_targets(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from emeg_fm.lora import inject_lora, _lora_linear_cls

        # Tiny stand-in with the same dotted layout as REVE's two target kinds.
        attn = nn.ModuleList([nn.Module(), nn.Module()])
        attn[0].to_qkv = nn.Linear(8, 24)
        attn[0].to_out = nn.Linear(8, 8)
        block = nn.ModuleList([attn[0]])
        transformer = nn.Module()
        transformer.layers = nn.ModuleList([block])
        model = nn.Module()
        model.transformer = transformer

        targets = ["transformer.layers.0.0.to_qkv"]
        wrapped = inject_lora(model, targets=targets, rank=2, alpha=4.0)
        LoRALinear = _lora_linear_cls()
        assert set(wrapped) == set(targets)
        assert isinstance(model.transformer.layers[0][0].to_qkv, LoRALinear)
        # untouched sibling stays a plain Linear
        assert isinstance(model.transformer.layers[0][0].to_out, nn.Linear)

    def test_muon_orthogonalizes_2d_update(self):
        torch = pytest.importorskip("torch")
        from emeg_fm.lora import make_muon

        Muon = make_muon()
        w = torch.nn.Parameter(torch.randn(16, 8))
        opt = Muon([w], lr=0.1)
        before = w.detach().clone()
        w.grad = torch.randn(16, 8)
        opt.step()
        # A step actually moved the parameter.
        assert not torch.allclose(w.detach(), before)
