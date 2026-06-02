"""LoRA adapters + a vendored Muon optimizer for the Scope C experiment.

``docs/MUON_EXPERIMENT.md`` Scope C fine-tunes frozen REVE with low-rank
adapters on the five attention matrices that WeightWatcher flags as
under-trained (α < 2), under two optimizers — AdamW vs Muon — to test whether
Muon's gradient-orthogonalization pulls those matrices' spectra toward the
self-averaging α ≈ 2 boundary (Martin's RG prediction, H1) while improving the
HBN bifactor probe (H3).

Layering, mirroring ``eeg_fm.py``: the pure-math / pure-python pieces
(``newton_schulz``, ``select_lora_targets``, ``lora_delta``,
``DEFAULT_REVE_LORA_TARGETS``) live at module top level and import only numpy,
so the package stays importable on machines without torch. The torch pieces
(``LoRALinear``, ``inject_lora``, ``make_muon``) import torch lazily inside
factory functions and are only touched on the GPU cluster.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# WeightWatcher-identified LoRA targets for brain-bzh/reve-base
# ---------------------------------------------------------------------------
# Per-block HT-SR analysis (scripts/analyze_reve_weightwatcher.py) found exactly
# five attention matrices with α < 2 — the under-trained / rank-collapsed
# matrices Martin's recipe says to fine-tune. All are attention (no FFs); the
# block list is sorted by ascending α (most under-trained first):
#
#   transformer.layers.2.0.to_qkv   α=1.56
#   transformer.layers.1.0.to_qkv   α=1.58
#   transformer.layers.0.0.to_out   α=1.72
#   transformer.layers.3.0.to_qkv   α=1.81
#   transformer.layers.10.0.to_out  α=1.92
#
# ``layers.{k}.0`` is the attention sub-module of block k (each block is a
# ModuleList([attn, ff])); ``to_qkv`` / ``to_out`` are its Linear projections.
DEFAULT_REVE_LORA_TARGETS: tuple[str, ...] = (
    "transformer.layers.2.0.to_qkv",
    "transformer.layers.1.0.to_qkv",
    "transformer.layers.0.0.to_out",
    "transformer.layers.3.0.to_qkv",
    "transformer.layers.10.0.to_out",
)


def select_lora_targets(available_names: Iterable[str],
                        targets: Iterable[str] | None = None) -> list[str]:
    """Validate requested LoRA target module names against what the model has.

    ``available_names`` is typically ``dict(model.named_modules()).keys()``.
    Each requested target must match an available name *exactly* — REVE's
    module layout is known and stable, so a miss means either the wrong target
    list or that the checkpoint's internals have shifted (in which case re-run
    ``analyze_reve_weightwatcher.py`` and update the names). We fail loudly
    rather than silently fine-tuning nothing.
    """
    available = set(available_names)
    targets = list(targets) if targets is not None else list(DEFAULT_REVE_LORA_TARGETS)
    missing = [t for t in targets if t not in available]
    if missing:
        raise KeyError(
            f"LoRA target module(s) not found on the model: {missing}. "
            f"REVE's layout may have changed — inspect dict(model.named_modules()) "
            f"and re-run analyze_reve_weightwatcher.py to refresh the target list."
        )
    return targets


def lora_delta(A, B, scaling: float):
    """The weight delta a LoRA pair contributes: ``scaling · (B @ A)``.

    ``A`` is ``(rank, in)``, ``B`` is ``(out, rank)``; the product is the
    ``(out, in)`` update merged into the base ``weight``. Backend-agnostic
    (numpy or torch) — used both by the torch ``LoRALinear.merged_weight`` and
    by the host-side post-hoc spectral check, which merges the trained deltas
    in numpy before calling ``sae.weight_spectral_summary``.
    """
    return scaling * (B @ A)


def newton_schulz(G, steps: int = 5,
                  coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
                  eps: float = 1e-7):
    """Quintic Newton–Schulz orthogonalization of a 2-D matrix (Muon's core).

    Pushes every singular value of ``G`` toward 1 — i.e. returns ≈ ``U Vᵀ`` of
    ``G = U Σ Vᵀ`` — which is exactly the spectral-mass-spreading step the RG
    theory predicts should move a layer toward α ≈ 2. Backend-agnostic: uses
    only ``@``, ``.T`` and elementwise ops, so it runs on numpy (tested here)
    and on torch tensors (the optimizer). Iterates on the wide orientation for
    numerical conditioning, then transposes back.
    """
    a, b, c = coeffs
    X = G
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    X = X / ((X * X).sum() ** 0.5 + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def weight_spectral_summary(weight, *, tail_frac: float = 0.5) -> dict:
    """HTSR-style spectral statistics of a 2-D weight matrix.

    The weight-side analogue of the activation participation-ratio audit, used
    for the Muon-vs-AdamW experiment (see ``docs/MUON_EXPERIMENT.md``). Computes
    the empirical spectral density (ESD) of the correlation matrix ``WᵀW`` from
    the singular values ``s`` (eigenvalues ``λ = s²``) and reports:

      * ``participation_ratio`` — Martin's ``M_tr = (Σλ)² / Σλ²``. The effective
        number of eigen-directions carrying the matrix's variance. Higher =
        spectral mass spread across many components (self-averaging); low =
        a few directions dominate (the non-self-averaging / α<2 regime).
      * ``alpha_hill`` — a Hill-estimator proxy for WeightWatcher's power-law
        exponent α of the ESD tail ``ρ(λ) ∝ λ^{-α}``, fit over the top
        ``tail_frac`` of eigenvalues. This is a lightweight, fixed-cutoff
        proxy — NOT WeightWatcher's KS-optimised ``powerlaw.Fit``. Use it for
        relative A/B comparisons (Muon vs AdamW on identical architecture),
        not as an absolute α report. α near 2 is Martin's RG ideal.
      * ``stable_rank`` — ``Σλ / λ_max`` (= ‖W‖_F² / ‖W‖_2²).
      * ``lambda_max``, ``n_eigs``.

    Lives here (pure-numpy, torch-free *and* jax-free) so both the JAX SAE
    bakeoff (Scope A) and the PyTorch LoRA-REVE run (Scope C) can call the
    identical α-fit code — the experiment's controls demand it. ``sae.py``
    re-exports it. Host-side SVD: these matrices are small and this runs once
    at end-of-training, so JIT isn't worth it.
    """
    W = np.asarray(weight, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"weight_spectral_summary needs a 2-D matrix, got {W.shape}")

    s = np.linalg.svd(W, compute_uv=False)
    lam = np.sort((s ** 2)[s > 0])[::-1]          # eigenvalues, descending
    n = int(lam.size)
    if n == 0:
        return {"participation_ratio": 0.0, "alpha_hill": float("nan"),
                "stable_rank": 0.0, "lambda_max": 0.0, "n_eigs": 0,
                "tail_frac": float(tail_frac)}

    sum_lam = float(lam.sum())
    sum_lam2 = float((lam ** 2).sum())
    participation_ratio = (sum_lam ** 2) / (sum_lam2 + 1e-30)
    lam_max = float(lam[0])
    stable_rank = sum_lam / (lam_max + 1e-30)

    # Hill tail-index on the upper ``tail_frac`` eigenvalues. ξ = n/Σln(λ/λ_min)
    # estimates the CCDF exponent; the ESD *density* exponent is α = ξ + 1,
    # which matches WeightWatcher's α convention.
    n_tail = max(2, int(round(tail_frac * n)))
    n_tail = min(n_tail, n)
    tail = lam[:n_tail]
    lam_min = float(tail[-1])
    ratios = np.log(tail / (lam_min + 1e-30))
    denom = float(ratios.sum())
    alpha_hill = (1.0 + n_tail / denom) if denom > 0 else float("nan")

    return {
        "participation_ratio": float(participation_ratio),
        "alpha_hill": float(alpha_hill),
        "stable_rank": float(stable_rank),
        "lambda_max": lam_max,
        "n_eigs": n,
        "tail_frac": float(tail_frac),
    }


# ---------------------------------------------------------------------------
# Torch pieces — imported lazily so ``import eeg_fm_spectral.lora`` is torch-free
# ---------------------------------------------------------------------------

def _lora_linear_cls():
    """Build (and cache) the ``LoRALinear`` nn.Module class.

    Defined inside a function so torch is only imported on the GPU cluster.
    Wraps a frozen base ``nn.Linear`` and adds a trainable low-rank path
    ``x → (B @ A) x · (alpha/rank)``. The base weight is detached from the
    graph; only ``A``/``B`` (and the head) receive gradients.
    """
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base: "nn.Linear", rank: int = 8, alpha: float = 16.0):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            in_f, out_f = base.in_features, base.out_features
            self.rank = rank
            self.scaling = alpha / rank
            self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
            self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
            # Standard LoRA init: A ~ small Gaussian, B = 0 → delta starts at 0
            # so the adapted model exactly equals the base at step 0.
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        def forward(self, x):
            out = self.base(x)
            delta = (x @ self.lora_A.T) @ self.lora_B.T
            return out + self.scaling * delta

        @torch.no_grad()
        def merged_weight(self):
            """Base weight + LoRA delta as a single ``(out, in)`` tensor — what
            the post-hoc spectral / WeightWatcher check analyzes."""
            return self.base.weight + lora_delta(self.lora_A, self.lora_B,
                                                 self.scaling)

    return LoRALinear


def inject_lora(model, targets: Iterable[str] | None = None,
                rank: int = 8, alpha: float = 16.0) -> dict:
    """Replace each target ``nn.Linear`` in ``model`` with a ``LoRALinear``.

    Returns ``{target_name: LoRALinear}`` for the wrapped modules, so the
    caller can collect their ``lora_A``/``lora_B`` as the trainable parameter
    set and later read ``merged_weight()`` for the spectral check. The base
    model is otherwise frozen by the wrapper.
    """
    import torch.nn as nn  # noqa: F401 — ensures torch present; raises early if not

    LoRALinear = _lora_linear_cls()
    named = dict(model.named_modules())
    chosen = select_lora_targets(named.keys(), targets)

    wrapped: dict = {}
    for name in chosen:
        base = named[name]
        parent_name, _, attr = name.rpartition(".")
        parent = named[parent_name] if parent_name else model
        lora_mod = LoRALinear(base, rank=rank, alpha=alpha)
        setattr(parent, attr, lora_mod)
        wrapped[name] = lora_mod
    return wrapped


def make_muon():
    """Build (and return) a single-device ``Muon`` torch optimizer.

    Vendored compact version of Keller Jordan's Muon: SGD-momentum whose 2-D
    updates are orthogonalized by ``newton_schulz`` before the step (1-D params
    fall back to plain momentum, but in Scope C all Muon-group params are the
    2-D LoRA matrices; the head + biases go to a separate AdamW group). The
    update is scaled by ``√max(1, rows/cols)`` so its RMS is comparable to
    AdamW's, which is why Muon needs its own learning rate.
    """
    import torch

    class Muon(torch.optim.Optimizer):
        def __init__(self, params, lr: float = 2e-2, momentum: float = 0.95,
                     nesterov: bool = True, ns_steps: int = 5):
            super().__init__(params, dict(lr=lr, momentum=momentum,
                                          nesterov=nesterov, ns_steps=ns_steps))

        @torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                mom, nesterov = group["momentum"], group["nesterov"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "mom" not in state:
                        state["mom"] = torch.zeros_like(g)
                    buf = state["mom"]
                    buf.mul_(mom).add_(g)
                    upd = g.add(buf, alpha=mom) if nesterov else buf
                    if upd.ndim == 2:
                        upd = newton_schulz(upd, steps=group["ns_steps"])
                        scale = max(1.0, upd.shape[0] / upd.shape[1]) ** 0.5
                        p.add_(upd, alpha=-group["lr"] * scale)
                    else:
                        p.add_(upd, alpha=-group["lr"])
            return loss

    return Muon
