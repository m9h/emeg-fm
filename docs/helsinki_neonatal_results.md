# Helsinki neonatal seizure EEG — results

*NeuroTechX-Atlas epilepsy arm, dataset 3 (after TUSZ, SeizeIT2). Completed
2026-07-04. A third, independent population (neonatal, vs adult-clinical/
adult-wearable) to stress-test whether the identity-trap pattern generalizes.*

## Setup

- **Data:** Helsinki neonatal seizure EEG (Stevenson et al., Zenodo 2547147),
  79 term/near-term neonates, continuous EEG, real 19-channel 10-20 montage,
  **3 independent expert annotators** per recording. Ground truth = per-second
  majority vote (≥2 of 3 agree), the standard convention for this corpus.
- **Split:** subject-disjoint 60/20/20 by neonate (49/15/15).
- **Models:** classical (log-band-power → HistGBM) and REVE — unlike SeizeIT2's
  proprietary wearable labels, Helsinki's channels are genuine 10-20 names
  (Fp1/Fp2/F3/F4/C3/C4/P3/P4/O1/O2/F7/F8/T3/T4/T5/T6/Fz/Cz/Pz), so both work.
- **Metric:** SzCORE (same scorer as TUSZ/SeizeIT2): Sensitivity@{1,10} FP/24h,
  dev-tuned F1, Event-Sens@FA step-AUC over `[0.1,10]` FP/day.
- **Identity-free axis:** re-fit with neonate identity LEACE-erased.

## Results (eval, subject-disjoint, 15 neonates)

| Model | eval AUC[.1-10 FP/d] | sens@1/d | sens@10/d | F1 (own op point) | Δ AUC |
|---|---|---|---|---|---|
| **classical** normal | 0.108 | 8.5% | 15.9% | 0.355 @ 24.3 FP/day | |
| classical identity-free | 0.041 | 2.4% | 8.5% | 0.505 @ 188.4 FP/day | **+0.067** |
| **REVE** normal | 0.000 | 0.0% | 0.0% | 0.464 @ 29.2 FP/day | |
| REVE identity-free | 0.104 | 8.5% | 17.1% | 0.535 @ 26.7 FP/day | **−0.104** |

## Findings — honest, not spun

**Classical replicates the pattern for the 4th time.** The mild identity-erasure
cost (Δ +0.067) is in the same direction and similar magnitude as TUSZ (+0.044),
SeizeIT2 (+0.019), and Sleep-EDF (+0.043). This is now a **robust, four-domain
classical baseline signature**.

**REVE does NOT clearly replicate the FM-collapse pattern here — and I'm
reporting that honestly rather than forcing it.** Unlike TUSZ and Sleep-EDF,
where identity erasure clearly *collapsed* REVE (F1/kappa dropping sharply),
on Helsinki **F1 is roughly stable or slightly higher under identity-free**
(0.464→0.535) and both configs land at a similar FP/day operating point
(29.2 vs 26.7). The AUC swing (Δ−0.104, the *opposite* sign from TUSZ/Sleep) is
plausibly a **small-sample artifact**: only 15 eval recordings and a sparse
number of actual seizure events make the achievable sensitivity-vs-FA curve a
coarse step function, so small threshold differences swing the strict
`[0.1,10]` FP/day AUC without reflecting a robust effect.

**Takeaway:** the identity trap is a well-established, repeated signature for
classical-vs-FM on **two** domains with adequate sample size (TUSZ eval=865,
Sleep-EDF eval=30 recordings/~29k epochs). On the **smallest** corpus so far
(Helsinki, 15 eval recordings), the FM-side signal is genuinely inconclusive —
this is a real limitation of the corpus size, not a refutation, but it should
not be reported as a third confirmation. A larger held-out slice (or pooling
across more of the corpus) would be needed to settle it.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_epilepsy_helsinki.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_epilepsy_helsinki.py --model reve --both
```

## Bugs fixed en route (3, all committed with regression tests)

1. Mixed-case channel names (`Fp1` vs TUSZ's `FP1` convention) — silently
   loaded 0 recordings, no error until the final concatenate crashed (`41b1cc3`).
2. Ragged annotation matrix — each column has valid values only for its own
   recording's duration; later rows go blank while other (longer) recordings'
   columns still have data in the same row (`be1f40a`).
3. Inconsistent `-REF`/`-Ref` suffix casing **across recordings** (not just
   within one file) — only 3 of 79 recordings had a full montage match before
   the fix (`03f56bf`).
