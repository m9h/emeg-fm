# Ear-EEG (EESM23, ds005178) sleep-staging results

*4th sleep dataset in the Atlas program, and the first on a genuinely
different **acquisition modality**: 4-channel wearable ear-EEG (RB/RT/LB/LT)
rather than scalp PSG. Found via NeuroAdapt-Bench
(github.com/leegabriel/NeuroAdapt-Bench, arXiv 2604.16926), which uses this
dataset as their "extreme modality shift" case. Only 10 subjects have
manually-scored sessions (usually 1-2 of each subject's ~12 recorded nights);
those ~20 sessions are the full cohort here. Same 5-class AASM label space
(W/N1/N2/N3/REM) as Sleep-EDF/HMC/DOD, Artefact epochs dropped.*

## Data-quality note: NaN epochs from wearable dropout

Both the classical (log-band-power) and REVE embeddings produced NaN for a
meaningful fraction of epochs — 14.2% (train), 18.6% (dev), 16.8% (eval) —
traced to intermittent electrode dropout / flat-line segments in the raw
ear-EEG signal (a real property of a wearable sensor across full nights, not
a loader bug). Epochs with any NaN feature are dropped before fitting/eval
(`scripts/atlas_sleep_eareeg.py::_drop_nan_rows`); this is disclosed here
rather than silently imputed, since dropping ~15-19% of one already-small
cohort is not free.

## Setup

- 8 train / 1 dev / 1 eval subjects (subject-level split, no train/eval
  session leakage).
- eval split = 2 recordings (both scored sessions of the held-out subject),
  1135 epochs pre-drop / 944 epochs post-drop.
- REVE channel mapping reuses NeuroAdapt-Bench's own published ear-electrode
  aliases (RB→A2, RT→T8, LB→A1, LT→T7) rather than inventing a new
  approximation.

## Results (eval split)

| Model | Config | balanced-acc | κ |
|---|---|---|---|
| classical (log-band-power) | normal | 64.2% | 0.495 |
| classical (log-band-power) | identity-free | 54.7% | 0.402 |
| REVE | normal | 59.2% | 0.448 |
| REVE | identity-free | 59.7% | 0.434 |

Identity-free Δ(eval κ): classical **+0.093**, REVE **+0.014**.

## Finding — honest, not spun

On this 4th sleep dataset, **classical log-band-power traps harder on subject
identity than REVE does** — the opposite of HMC/Sleep-EDF (where REVE traps
hardest) but the same direction as DOD-O (classical Δ=+0.165 there). REVE's
Δ here is close to zero, similar in spirit to its near-zero Δ on DOD-O's
sibling DOD-H being REVE-worse — i.e. **the sleep-dataset identity-trap
direction record is now genuinely mixed across all 4 datasets** (HMC/Sleep-EDF:
REVE traps harder; DOD-O: classical traps harder; DOD-H: REVE traps harder;
Ear-EEG: classical traps harder), not a universal FM-vs-classical pattern.
Reported as-is rather than smoothed into either "FMs always trap" or "FMs
never trap."

Absolute performance is modest for both models on this cohort (κ 0.40-0.50
range) — expected given the small N (10 scored subjects total), the reduced
information in 4 ear electrodes vs full scalp PSG, and the ~15-19% of epochs
lost to NaN dropout.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_sleep_eareeg.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_sleep_eareeg.py --model reve --both
```
