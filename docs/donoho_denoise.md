# Donoho spiked-model denoising — and when it does NOT apply (the E4 finding)

`emeg_fm/denoise.py` implements the parameter-free Donoho tools from the methods brief:
- `optimal_hard_threshold` — Gavish–Donoho (2014) 4/√3 optimal hard threshold for singular values
  (known/unknown noise) → a rank with no magic number.
- `recover_spike` — BBP / Baik–Silverstein inverse spike map (population eigenvalue + eigenvector cos²
  behind a sample-covariance eigenvalue).
- `denoise_cov_eigs` — shrink the MP noise bulk to σ², debias the spikes (covariance denoiser; e.g. for
  the Langevin Σ/D EPR fix, where `D⁻¹` amplifies MP-spread small eigenvalues).
- `denoise_whiten` — **rank-reduced** whitening: keep only the spikes above the MP edge, whiten by their
  debiased eigenvalues, discard the bulk. The principled replacement for the `reg=1e-3` Tikhonov
  whitener in the cross-modal CCA (`cross_modal_spectrum(..., denoise=True)`).

All validated on synthetic spiked data (planted-rank recovery, BBP round-trip, noise-collapse) — tests
green in `tests/test_denoise.py`.

## The honest result: it does NOT improve E4 — because the E4 features aren't spiked
Applying `denoise=True` to the real EEG↔structural CCA made it **worse**, not better:

| whitener | full obs | full null p95 | resid obs | resid p |
|---|---|---|---|---|
| `reg=1e-3` (committed) | 0.633 | 0.358 | 0.420 | **0.022** |
| Donoho rank-reduced    | 0.838 | 0.814 | 0.754 | 0.200 |

The null **bias floor rose** (0.32→0.74) and the residual coupling **lost significance**. The reason is
in the spectra — they violate the spiked model's "finite rank of spikes + white-noise MP bulk":

- **REVE EEG embeddings: power-law**, top-50 log-log slope ≈ **−2.2**, cond ≈ 2e8 → the HT-SR critical
  heavy-tail (α≈2). A continuous power-law has no clean bulk edge; ~166/512 dims sit "above edge",
  which is meaningless. (This is exactly the zeta-law / weight-vs-data-spectrum regime.)
- **Block-pooled structural: rank-deficient**, median eigenvalue ≈ 1e-19 (≈267 near-constant out-of-brain
  GM columns) → the MP-median σ² estimate degenerates.

When half the spectrum is "signal" (power-law) or the median is ≈0 (rank-deficient), the spiked-MP
denoiser mis-estimates σ² and the edge, so rank-reduced whitening over-fits a high-dim noisy subspace and
inflates the canonical correlations.

## Verdict
- **Keep `reg=1e-3` + the permutation null for E4.** For heavy-tailed / rank-deficient embeddings, the
  permutation null (which calibrates *whatever* bias the whitener has) is the honest tool, not spiked-model
  denoising. `denoise=True` is available but inappropriate here.
- **Where the denoiser IS right:** genuinely spiked, moderate-dimension sample covariances — the 40-D
  Langevin Σ/D in the SMNI/jaxctrl EPR estimate, MPPCA of the WAND DWI, the DMD/Koopman rank — where a
  small number of spikes sit above a real noise bulk. The tool is shared; the *applicability test* is
  whether the spectrum actually has a bulk (check the log-log slope / a visible MP edge first).
