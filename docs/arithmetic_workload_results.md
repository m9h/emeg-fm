# Arithmetic (Zyma 2019) mental-workload — results

*New task domain: mental workload / task-engagement (rest vs mental
arithmetic), and one of OpenEEGBench's 12. 36 subjects, binary
classification, 19-channel standard monopolar 10-20 montage.*

## Setup

- 26 train / 5 dev / 5 eval subjects, subject-disjoint.
- Pulled via braindecode's own `BaseConcatDataset.pull_from_hub`
  (`braindecode/arithmetic_zyma2019`, open HF mirror).

## Results (eval split)

| Model | Config | balanced-acc | κ |
|---|---|---|---|
| classical (log-band-power) | normal | 66.6% | 0.408 |
| classical (log-band-power) | identity-free | 73.0% | 0.483 |
| REVE | normal | 62.4% | 0.244 |
| REVE | identity-free | 60.7% | 0.192 |

Identity-free Δ(eval κ): classical **−0.075** (identity-free actually
*better*), REVE **+0.052** (small).

## Finding — honest, not spun

**Classical beats REVE here on raw performance** (κ 0.408 vs 0.244) — a
state-detection task (rest vs mental-arithmetic engagement) plays to
classical spectral features' strengths, consistent with the program's
general pattern on non-affect tasks. **Neither model shows a large
identity-free Δ** — classical's is actually negative (identity-free
slightly *improves* performance, likely small-N noise at only 5 eval
subjects) and REVE's is small and positive. With only 36 total subjects,
these Δ magnitudes should be read as indicative, not conclusive.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_workload_arithmetic.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_workload_arithmetic.py --model reve --both
```
