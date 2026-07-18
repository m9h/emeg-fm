# FACED (Synapse syn50614194) affect/emotion recognition — results

*First emotion-recognition dataset in the NeuroTechX Atlas program — a new
task domain (BCI/ERP/SSVEP/brain-age/sleep/epilepsy → now affect), and one
of OpenEEGBench's 12 (github.com/braindecode/OpenEEGBench). 123 subjects,
28 emotion-eliciting video-clip trials each, 9-class label (anger, disgust,
fear, sadness, neutral, amusement, inspiration, joy, tenderness).*

## Setup

- 87 train / 18 dev / 18 eval subjects, subject-disjoint.
- Pre-processed release (`Processed_data/subXXX.pkl`, 250 Hz, 30s/trial,
  30 real EEG channels — genuine monopolar 10-20 names, no anchor-electrode
  approximation needed).
- Same evaluation harness as sleep/epilepsy (`atlas_sleep_section.evaluate`):
  StandardScaler + optional LEACE patient-erasure + LogisticRegression/GBM
  probe, balanced-acc + Cohen's κ.

## Results (eval split)

| Model | Config | balanced-acc | κ |
|---|---|---|---|
| classical (log-band-power) | normal | 16.3% | 0.059 |
| classical (log-band-power) | identity-free | 17.2% | 0.068 |
| REVE | normal | 29.3% | 0.205 |
| REVE | identity-free | 28.9% | 0.197 |

Identity-free Δ(eval κ): classical **−0.009**, REVE **+0.008** — both
effectively zero.

## Finding — honest, not spun

**REVE shows real above-chance signal on 9-class emotion decoding** (κ 0.205,
balanced-acc 29.3% vs 11.1% chance) where **classical barely clears chance**
(κ 0.059-0.068, bacc 16-17%) — a genuine FM-vs-classical win, the first time
in this program a foundation model has meaningfully out-performed the
classical baseline on a task domain classical is normally strong on
(spectral/oscillatory features), which fits, since affect decoding from EEG
plausibly draws on higher-order structure closer to REVE's training
distribution than raw band power.

**Neither model shows a meaningful identity-free Δ.** This is worth
contextualizing against the published literature on this dataset family
(`docs/seed_identity_trap_literature.md`): SEED/SEED-VIG show large
(10-17 point) subject-dependent-vs-subject-independent accuracy gaps — but
that gap is between a **pooled** (non-subject-disjoint) split and a
**subject-disjoint (LOSO)** split. Our evaluation here is *already*
subject-disjoint by construction (87/18/18 split, no subject crosses
train/eval), so the identity-free LEACE erasure is being applied *on top of*
an already-disjoint protocol — there's little residual subject-identity
signal left for it to find and remove, which is exactly what a small Δ here
would predict. **This is consistent with, not contradicting, the literature's
pooled/disjoint gap** — it doesn't mean FACED has no identity-leakage risk at
all, only that our specific (disjoint-split) protocol has already absorbed
most of it, the same way several of our MOABB per-trial re-checks found the
pooled-split trap shrinking once the split itself was made honest.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_affect_faced.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_affect_faced.py --model reve --both
```
