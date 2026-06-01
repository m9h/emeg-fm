# Muon vs AdamW: an RG-theory optimizer experiment

This repo already owns both of the instruments needed to test a specific
prediction from Charles Martin's *Renormalization Group Theory of Learning*:
that an optimizer which **whitens / orthogonalizes the gradient** should drive
layers toward the self-averaging boundary **α ≈ 2** and away from the α < 2
"correlation-trap" (dominant-tail) regime. Muon *is* that optimizer — its
Newton–Schulz step orthogonalizes the 2-D weight gradient, which is literally
spectral-mass-spreading. AdamW has no such pressure.

We measure the effect with the two diagnostics this repo is built around:

- **Weight side** — the HTSR power-law exponent α and the participation count
  `M_tr = (Σλ)² / Σλ²`. See [WEIGHTWATCHER_EEGFM_ANALYSIS.md](WEIGHTWATCHER_EEGFM_ANALYSIS.md).
- **Activation side** — the participation ratio of block activations (the same
  `(Σλ)²/Σλ²`) and the **SAE yield** (live-feature fraction at fixed `d_dict`,
  matched reconstruction). See the findings table in the [README](../README.md).

## Hypotheses (falsifiable)

| # | Side | Prediction (Muon vs AdamW twin) |
|---|------|---------------------------------|
| **H1** | weights | tighter α distribution centered nearer 2; smaller `%α<2` |
| **H2** | activations | higher activation participation ratio (less rank-collapsed) |
| **H3** | interpretability | higher SAE yield (more live features, same EV) |

Clean null: if Muon and AdamW land at the same α and the same participation
ratio, the RG-optimizer claim doesn't bind in this setting.

## Three scopes (cheapest → gold standard)

### Scope A — Muon on the SAE itself *(scaffolded; run this first)*

The TopK-SAE is a 1-layer autoencoder trained with `optax`. Swapping its
optimizer is a one-flag change, so this is the cheapest directional signal and
an instrument check. It tests the optimizer claim on the SAE's *own* decoder
matrix, not on a deep transformer — so it is suggestive, not the real test.

```bash
# A/B on the same REVE L-1 activations already on disk. Single GPU, FIFO.
ACTS=/data/derivatives/eeg_sae/acts/brain-bzh_reve-base_L-1_EEG2025R1_RestingState.npz \
  D_DICT=4096 MUON_LR=2e-2 ADAM_LR=1e-3 \
  sbatch scripts/muon_sae_bakeoff.sbatch
```

The bakeoff trains two SAEs that differ *only* in optimizer (shared
seed / `d_dict` / `k` / epochs), then prints a comparison of dictionary health
and the HTSR spectral summary of the learned dictionary:

```
=== Muon vs AdamW SAE bakeoff (dec_weight) ===
adam   EV=0.974  dead=75.0%  live=   1024  l0=32.0  PR=  ...  alpha_hill=...
muon   EV=0.974  dead=...    live=   ...   l0=32.0  PR=  ...  alpha_hill=...
Delta(PR)  muon-adam = +...
Delta(alpha_hill)     = +...  (RG ideal ~2.0)
Delta(dead_frac)      = ...
```

Under the hand-tools:

- `scripts/train_sae.py --optimizer {adam,muon}` — `muon` uses
  `optax.contrib.muon`, which orthogonalizes the 2-D `enc_weight` / `dec_weight`
  gradients via Newton–Schulz and routes the 1-D biases to its internal AdamW.
- `eeg_fm_spectral.sae.weight_spectral_summary(W)` — host-side SVD →
  `participation_ratio` (= `M_tr`), `alpha_hill` (a Hill-estimator proxy for
  WeightWatcher's α, **not** the KS-optimised `powerlaw.Fit` — use it for
  relative A/B only), and `stable_rank`. Recorded in each run's
  `<prefix>.json` under `"spectral"`.

> **Tune the LRs separately.** Muon's update is RMS-matched and needs its own
> learning rate; comparing it at AdamW's optimal LR is a rigged test. Sweep
> `MUON_LR` and `ADAM_LR` independently and compare best-of-best.

### Scope C — LoRA-Muon on REVE's under-trained attention *(the feasible real test)*

The WeightWatcher sweep already pinpoints REVE blocks 0–3 + 10 as carrying the
α < 2 attention matrices (the exact LoRA-QKVO sites — see
[WEIGHTWATCHER_EEGFM_ANALYSIS.md](WEIGHTWATCHER_EEGFM_ANALYSIS.md)). Attach LoRA
on those matrices and fine-tune frozen REVE on the HBN psychopathology bifactor
eval (the NeuralBench harness) twice — Muon vs AdamW on the LoRA params. Then
re-run all three measurements:

1. WeightWatcher α on the merged `base + LoRA` matrices → **H1**
2. re-extract block activations → participation ratio → **H2**
3. retrain a fixed-`d_dict` SAE → yield → **H3**
4. downstream bifactor accuracy as the does-it-actually-help check

Compute-cheap (LoRA), and scientifically the most interesting: it ties the
RG-optimizer prediction directly to the specific α < 2 matrices and the SAE
yield this repo already reports. **Not yet scaffolded.**

### Scope B — from-scratch small encoder *(gold standard, expensive)*

Pretrain a small LaBraM-style encoder on HBN from scratch under each optimizer;
full per-block α sweep + activation + SAE yield. Cleanest causal claim (the
optimizer is the *only* difference, no frozen-base confound) but multiple
GPU-days and a pretraining loop we don't have yet. Park unless A + C show a real
effect worth nailing down.

## Controls that matter

- **Per-optimizer LR sweep** (see warning above).
- **Hybrid is mandatory** — Muon only on 2-D matmul weights; AdamW on
  embeddings, norms, biases, and any output head. `optax.contrib.muon` does this
  split by parameter rank automatically; keep it.
- **Match the compute budget**, same seed / init / data order, and run the
  *identical* α-fit and participation-ratio code on both checkpoints.

## References

- C. H. Martin, *Renormalization Group Theory of Learning* (2026).
- Jordan et al., *Muon: an optimizer for the hidden layers of neural networks* (2024).
- L. Gao et al., *Scaling and evaluating sparse autoencoders* (2024).
