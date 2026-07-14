# ISRUC-Sleep (subgroups I/II/III) — results

*6th sleep dataset in the Atlas program (after Sleep-EDF, HMC, Dreem
DOD-H/DOD-O, Ear-EEG). Adult sleep-disorder clinical PSG cohort across 3
subgroups (I: 100 single-night subjects, II: 8 subjects with 2 nights each,
III: 10 healthy-control single-night subjects), mastoid-referenced bipolar
EEG derivations (F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1).*

## Setup

- 84 train / 17 dev / 17 eval subjects (subject-disjoint, prefixed by
  subgroup so numbering doesn't collide across the 3 cohorts) — the largest
  subject count of any sleep dataset in the program besides Sleep-EDF/HMC.
- Ground truth: scorer-1 hypnogram (single-scorer convention, unlike
  Helsinki's 3-annotator majority vote); ISRUC's on-disk {0,1,2,3,5} labels
  remapped to the shared 0-4 AASM (W/N1/N2/N3/REM) space.
- REVE channel mapping reuses the DOD/CHB-MIT anchor-electrode convention
  (first electrode of each bipolar-to-mastoid pair).

## Results (eval split)

| Model | Config | balanced-acc | κ |
|---|---|---|---|
| classical (log-band-power) | normal | 66.4% | 0.618 |
| classical (log-band-power) | identity-free | 62.6% | 0.553 |
| REVE | normal | 69.7% | 0.637 |
| REVE | identity-free | 64.1% | 0.560 |

Identity-free Δ(eval κ): classical **+0.064**, REVE **+0.077**.

## Finding — honest, not spun

Both models perform respectably here (κ 0.55-0.64, the best absolute
performance of any epilepsy/sleep set with a real identity-free comparison
this session besides HMC) — unlike the near-floor REVE collapses seen on
Helsinki/CHB-MIT/Siena, ISRUC's channel positions and montage are close
enough to standard 10-20 for REVE to work properly. **The identity-free
cost is modest and nearly equal for both models** (classical +0.064, REVE
+0.077) — REVE traps very slightly more, the same *direction* as
Sleep-EDF/HMC/DOD-H, but at a much smaller magnitude than HMC's dramatic
+0.490 collapse. Folded into the now-6-dataset sleep record: HMC and
Sleep-EDF (REVE traps much harder), DOD-H and ISRUC (REVE traps slightly
harder), DOD-O and Ear-EEG (classical traps harder) — a real mixed record,
not a clean universal pattern in either direction.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_sleep_isruc.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_sleep_isruc.py --model reve --both
```
