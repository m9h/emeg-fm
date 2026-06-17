# Decoding ERP CORE components from a frozen EEG foundation model under the ERPLAB protocol

Does a frozen EEG foundation model (FM) carry the single-component ERP
information that the ERP CORE decoding pipeline extracts from scalp voltage,
when both are scored under the **same** decoding protocol? This is the question
the Steve Luck lab's ERP CORE decoding scripts answer for raw voltage; here we
hold their protocol fixed and swap the *feature* — 28-channel scalp voltage vs.
a frozen REVE embedding — so the comparison isolates what the FM adds.

Two results:

- **Within participant, the frozen FM matches or beats the raw-voltage baseline
  in 5/7 components** — most strongly the N170 (+0.150) — under ERPLAB's own
  linear-SVM / pseudo-ERP / regularization-sweep protocol, with no task
  training.
- **The FM's cross-subject decoding is not an identity shortcut.** Each
  embedding carries a strong linear subject axis, but erasing it (LEACE) yields
  no label-decode lift on any component, under a strict single-trial,
  leave-subjects-out metric.

This is the spectral/ERP counterpart to the identity-trap audit the repo runs on
MOABB BCI datasets (see the [README](../README.md) leaderboards): same erasure
machinery, applied to ERP contrasts under the originating lab's protocol.

## Data

ERP CORE (Kappenman et al., 2021): 40 participants, seven components (N170, MMN,
N2pc, P3, N400, ERN, LRP), each a binary contrast. Epoch windows follow the ERP
CORE decoding scripts' `epochTW` per component: (−200, 800) ms for
N170/MMN/N2pc/P3/N400, (−600, 400) ms for ERN, (−800, 200) ms for LRP. (MOABB's
`ErpCore2021` swaps the ERN and LRP intervals; the driver overrides
`dataset.interval` back to the Luck values.) Trials are bandpassed 0.5–99.5 Hz
and resampled to 200 Hz; the first 28 scalp channels are retained. The two
classes per component are the ERP CORE Target/NonTarget contrasts (e.g. N170
face vs. object; ERN error vs. correct response; LRP contralateral vs.
ipsilateral response).

## Decoders

**Classical scalp baseline (ERPLAB DECODE).** A linear SVM is trained on the
28-channel voltage vector at each timepoint within the component's measurement
window (every 5th sample), and the score is the mean accuracy across that
window. This reimplements the ERPLAB12 `pop_decoding` configuration: linear SVM,
pseudo-ERP crossblock cross-validation (3 folds), `nIter = 100` random block
assignments, `EqualizeTrials = 'classes'`, and the regularization grid
`Gamma = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]` swept as the SVM box constraint,
reporting the best grid value. Measurement windows: N170 110–150, MMN 125–225,
N2pc 200–275, P3 300–600, N400 300–500, ERN 0–100, LRP −100–0 ms.

**Foundation-model decoder.** Per-trial embeddings are taken from a frozen REVE
model (`brain-bzh/reve-base`, block 6, token-mean-pooled; weights never
updated). Inputs follow REVE's published contract: per-participant z-scoring and
amplitude clamping at ±15 SD. Each trial yields one embedding vector, decoded
with the identical SVM / crossblock / Gamma-grid protocol as the baseline (the
embedding has no time axis, so it is decoded directly).

Both decoders run within participant and are grand-averaged across the 40
participants. The only difference between the two columns is the feature; the
classifier, cross-validation, trial-equalization, regularization sweep, and
aggregation are held identical.

## Identity-shortcut audit

A frozen FM could appear to decode an ERP while in fact separating participants
whose trials differ in class proportion. We test this with a cross-subject,
single-trial decode (one trial = one sample, balanced accuracy under
`StratifiedGroupKFold` grouped by participant, so train and test never share a
participant) before and after **linear erasure of the subject-identity subspace**
(LEACE; Belrose et al., 2023). The lift `free − raw` is reported only when the
raw decode clears an interpretability gate (≥ 0.55) and the subject subspace
does not fill the feature space (rank < 0.95·dim); a linear subject-identity
probe (balanced accuracy vs. chance) documents the axis that was removed.

This is the same `fmscope.diagnostics.erasure.subject_axis_erasure` used by the
MOABB identity-free leaderboards — deliberately *stricter* than a pooled-
condition erasure, which averages each (participant, condition) into a handful
of near-noiseless pseudo-recordings and inflates the apparent identity
dependence.

## Results

| Component | Scalp (within) | REVE (within) | Δ (REVE−scalp) | Cross raw → id-free | Lift | Subject BA (pre→post / chance) |
|---|---:|---:|---:|---:|---:|---:|
| N170 | 0.666 | **0.816** | +0.150 | 0.652 → 0.637 | −0.015 | 0.787 → 0.029 / 0.025 |
| N2pc | 0.620 | **0.664** | +0.044 | 0.567 → 0.566 | −0.002 | 0.770 → 0.019 / 0.025 |
| P3 | 0.709 | **0.755** | +0.046 | 0.643 → 0.628 | −0.015 | 0.774 → 0.016 / 0.025 |
| N400 | 0.741 | **0.779** | +0.038 | 0.608 → 0.601 | −0.007 | 0.777 → 0.002 / 0.025 |
| ERN | 0.862 | **0.897** | +0.035 | 0.803 → 0.773 | −0.029 | 0.707 → 0.027 / 0.025 |
| MMN | **0.661** | 0.638 | −0.023 | 0.562 → 0.561 | −0.001 | 0.584 → 0.043 / 0.025 |
| LRP | **0.839** | 0.770 | −0.069 | 0.649 → 0.643 | −0.006 | 0.665 → 0.025 / 0.025 |

Within participant, the frozen FM embedding matches or exceeds the raw-voltage
baseline in 5/7 components (N170, N2pc, P3, N400, ERN), most strongly for the
N170 (+0.150). The baseline wins for MMN (−0.023) and LRP (−0.069), the two
contrasts most tied to focal or lateralized voltage that the FM's pooled
embedding does not preserve.

The identity audit is valid (interpretable, non-degenerate) for all seven
components. Each embedding carries a strong linear subject axis — subject
balanced accuracy 0.58–0.79 against a 0.025 chance, erased to near chance — yet
removing it produces no positive lift in the cross-subject label decode anywhere
(−0.001 to −0.029). The FM's component decoding is therefore not riding on a
recoverable subject-identity shortcut.

## Notes and limitations

- Preprocessing is shared between the two decoders (the FM's broadband contract)
  rather than the ERP CORE pipeline's component-specific filtering, so the
  comparison isolates the *feature* under a common front-end rather than
  reproducing the ERP CORE preprocessing verbatim. Baseline correction is not
  applied; per-feature standardization centers each decode input. A strict
  ERPLAB-preprocessing replication of the scalp baseline is a small follow-up.
- We use the first 28 of the 30 ERP CORE scalp sites; the exact channel subset
  can be aligned to the ERP CORE decoding scripts if required.
- The cross-subject single-trial metric is deliberately stricter than a
  pooled-condition erasure (see the audit section).

## Reproduction

- Driver: [`scripts/erpcore_luck_parity.py`](../scripts/erpcore_luck_parity.py)
- Protocol probe: [`fmscope/training/svm_probe.py`](../fmscope/fmscope/training/svm_probe.py)
  — ERPLAB-protocol linear-SVM probe + scalp DECODE baseline, unit-tested in
  [`tests/test_svm_probe.py`](../tests/test_svm_probe.py).
- Launch: `sbatch scripts/erpcore_luck_parity.sbatch --n-iter 100`
  (Docker NGC PyTorch 26.05 + /mnt/t9; needs the gated `brain-bzh/reve-base`).
- Outputs: `results/erpcore_luck_parity/erpcore_luck_parity.{csv,md}`.
