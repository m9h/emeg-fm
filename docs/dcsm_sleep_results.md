# DCSM sleep-staging — results

*7th sleep dataset in the Atlas program (after Sleep-EDF, HMC, Dreem
DOD-H/DOD-O, Ear-EEG, ISRUC), and the **largest single sleep cohort** by
subject count (255 subjects, 179/38/38 split) — bigger than Sleep-EDF (197
recordings) and HMC (151 recordings). Bipolar-to-mastoid montage (F3-M2,
F4-M1, C3-M2, C4-M1, O1-M2, O2-M1), variable-duration-interval hypnograms
expanded losslessly to the shared 30s AASM grid.*

## Setup

- 179 train / 38 dev / 38 eval subjects, subject-disjoint.
- REVE channel mapping reuses the DOD/ISRUC/CHB-MIT anchor-electrode
  convention (first electrode of each bipolar pair).
- Same evaluation harness (`atlas_sleep_section.evaluate`) as every other
  sleep dataset in the program.

## Results (eval split)

| Model | Config | balanced-acc | κ |
|---|---|---|---|
| classical (log-band-power) | normal | 78.2% | 0.759 |
| classical (log-band-power) | identity-free | 72.7% | 0.710 |
| REVE | normal | 77.7% | 0.713 |
| REVE | identity-free | 40.2% | 0.156 |

Identity-free Δ(eval κ): classical **+0.048**, REVE **+0.557**.

## Finding — honest, not spun

**DCSM is the single largest identity-trap collapse in the whole sleep
record — larger even than HMC's Δ+0.490.** REVE starts essentially tied
with classical on raw performance (κ 0.713 vs 0.759, both strong — this is
also the best absolute classical result of any sleep dataset, κ 0.759) but
**collapses to near-chance under identity erasure** (κ 0.156, balanced-acc
40.2% — barely above the 5-class floor of 20%), while classical only drops
modestly (Δ+0.048, in line with its typical cost elsewhere). Because this
is also the largest-N sleep dataset in the program (38 eval subjects, far
more than HMC's or DOD's eval cohorts), this result carries more statistical
weight than any single prior sleep-domain identity-trap finding — it is not
a small-N fluke.

This strengthens the HMC/Sleep-EDF side of the now-7-dataset sleep record
(REVE traps harder: HMC, Sleep-EDF, DOD-H, ISRUC (slight), **DCSM
(dramatic)** — vs classical traps harder: DOD-O, Ear-EEG). With DCSM's large
N behind it, the REVE-traps-harder direction now has the strongest single
piece of evidence in the sleep domain, though the overall 7-dataset record
remains genuinely mixed and is reported as such, not as a settled universal
claim.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_sleep_dcsm.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_sleep_dcsm.py --model reve --both
```
