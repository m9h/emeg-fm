# eeg-fm-spectral

**Spectral diagnostics of EEG foundation models — and the sparse-autoencoder interpretability they predict.**

Two complementary, weights-first lines of analysis on the current generation of
EEG foundation models (REVE, LaBraM, LUNA, BENDR, BIOT, CBraMod, ZUNA):

1. **WeightWatcher / HTSR analysis** — per-layer heavy-tailed power-law exponents
   (α) of every weight matrix, ranking pretraining quality *without any
   training or test data*. ([docs/WEIGHTWATCHER_EEGFM_ANALYSIS.md](docs/WEIGHTWATCHER_EEGFM_ANALYSIS.md))
2. **TopK sparse autoencoders (SAEs)** on block activations — mechanistic
   interpretability that decomposes a model's internal representations into a
   sparse dictionary, then probes each feature against HBN clinical concepts
   (p-factor, internalizing, externalizing, attention, age, sex).

The pure-JAX SAE (Gao 2024 aux_k dead-feature revival) lives in
[`eeg_fm_spectral/sae.py`](eeg_fm_spectral/sae.py); the activation-extraction
adapters in [`eeg_fm_spectral/eeg_fm.py`](eeg_fm_spectral/eeg_fm.py).

## Why these two analyses belong together

Charles H. Martin's *Renormalization Group Theory of Learning* (May 2026) unifies
his earlier **HTSR** (Heavy-Tailed Self-Regularization — the WeightWatcher theory)
and **SETOL** theories into a Wilsonian RG account of layer convergence. The
central claim: a well-trained dense layer sits near power-law exponent **α ≈ 2**,
which is simultaneously the *self-averaging boundary* (where the trace-law gain is
spread across many eigencomponents instead of dominated by a few) and the *RG
scale-balance point*. Below α = 2 ("correlation traps," dominant-tail takeover)
a layer overfits; its representation is carried by a handful of directions.

This repo connects that spectral picture to interpretability on two fronts:

- **Weight side (HTSR/WeightWatcher).** We measure per-block α for each EEG-FM.
  Martin's diagnostics — fitted α, dominant-tail burden, and the **participation
  count** `M_tr = (Σλ)²/Σλ²` — are exactly the quantities we report.
- **Activation side (SAE yield).** We measure the participation ratio (the same
  `(Σλ)²/Σλ²` statistic) of the *activations*, and find it directly predicts SAE
  yield. A near-rank-2 activation spectrum has almost nothing for a wide
  dictionary to decompose — the activation-space analogue of Martin's
  non-self-averaging / dominant-tail regime.

| model (FM) | weight α-mean | % α<2 | activation participation ratio | SAE viability |
|---|---:|---:|---:|---|
| **REVE**  | 3.61 | 5.6%  | ≈ 4.9 | viable substrate — structured single-feature clinical signal |
| **LaBraM**| 3.76 | 4.2%  | (in progress) | well-trained; second wide target |
| **LUNA**  | 3.93 | 10.4% | — | well-trained, heaviest tails |
| **BENDR** | 2.10 | 68.0% | — | severely under-trained (largest model in suite) |
| **ZUNA**  | —    | —     | ≈ 1.7 | poor SAE target — ~99% dead features is *correct*, not a bug |

ZUNA is a masked-diffusion autoencoder with a 32-d latent bottleneck: its pooled
encoder activations are intrinsically ≈ rank-2 at every layer, so a wide TopK-SAE
has nothing to decompose. REVE (wide contrastive/MAE) is the genuinely rich
substrate. **Measure the activation participation ratio before sizing a
dictionary.**

## Repository layout

```
eeg_fm_spectral/
  adapters.py   model-agnostic HuggingFace adapter registry + JAX factory
  eeg_fm.py     REVE / LaBraM / ZUNA activation-extraction adapters (forward hooks)
  sae.py        TopK sparse autoencoder, pure JAX (Gao 2024 aux_k, dictionary_health)
scripts/
  extract_eeg_fm_acts.py        eegdash/HBN -> windows @ 200 Hz -> block activations -> .npz
  extract_eeg_fm_acts_local_bids.py   per-subject streaming variant (large releases)
  train_sae.py                  fit a TopK SAE to extracted activations
  sae_concept_probes.py         per-feature linear/logistic probes -> HBN clinical concepts
  analyze_eegfm_weightwatcher.py, analyze_reve_weightwatcher.py   WeightWatcher HTSR analysis
  download_hbn_s3_direct.py, predownload_hbn_all.py               HBN data staging
  *.sbatch                      Slurm wrappers (single-GPU GB10; Apptainer SIF)
tests/        pytest suite (mocks all model downloads; no GPU / gated access needed)
docs/         WeightWatcher analysis writeup + eegdash issue notes
```

## Quickstart (CPU, no models)

```bash
pip install -e .            # jax, numpy, optax
PYTHONPATH=. python -m pytest tests/ -q
```

## Full pipeline (GPU cluster)

The extraction/training runs use Apptainer SIFs (PyTorch for extraction, JAX for
SAE training) on a single GPU via Slurm. See the `*.sbatch` headers for the exact
env contract. End-to-end:

```bash
# 1. extract block activations (REVE block 6, HBN release 1, RestingState)
RELEASE=1 FULL=1 LAYER=6 sbatch scripts/extract_eeg_fm_acts.sbatch

# 2. WeightWatcher HTSR analysis (per-block alpha, no data needed)
sbatch scripts/analyze_reve_weightwatcher.sbatch   # (or analyze_eegfm_weightwatcher.py)

# 3. train a TopK SAE on the activations
ACTS=.../brain-bzh_reve-base_L6_EEG2025R1_RestingState.npz \
  K=32 D_DICT=4096 AUX_K=32 sbatch scripts/train_sae.sbatch

# 4. probe each SAE feature against HBN clinical concepts
ACTS=...npz SAE=...sae.npz sbatch scripts/sae_concept_probes.sbatch
```

## Models & data

- **REVE** (`brain-bzh/reve-base`) is **gated** — accept the Responsible Use
  Agreement on HuggingFace first. No token or weights are stored in this repo.
- **LaBraM** (`braindecode/labram-pretrained`) and **ZUNA** (`mhough/zuna-base`)
  load without a token.
- EEG data is HBN via [eegdash](https://github.com/sccn/EEGDash) / OpenNeuro.
  See [docs/EEGDASH_ISSUES.md](docs/EEGDASH_ISSUES.md) for routing-bug workarounds.

Large artifacts (`.npz` activations, `.sif` containers, model weights) are kept on
the cluster and are git-ignored — this repo is code + derived results only.

## References

- C. H. Martin, *Renormalization Group Theory of Learning* (2026).
- C. H. Martin & M. W. Mahoney, *Implicit Self-Regularization in Deep Neural
  Networks* (JMLR 2021); *Predicting trends in the quality of state-of-the-art
  neural networks without access to training or test data* (Nature Comms 2021).
- C. H. Martin & C. Hinrichs, *SETOL: A Semi-Empirical Theory of (Deep) Learning*
  (arXiv:2507.17912).
- [WeightWatcher](https://weightwatcher.ai/) — the open-source HTSR diagnostic tool.
- L. Gao et al., *Scaling and evaluating sparse autoencoders* (2024) — TopK + aux_k.

## License

MIT — see [LICENSE](LICENSE).
