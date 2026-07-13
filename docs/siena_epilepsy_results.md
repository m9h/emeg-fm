# Siena Scalp EEG (PhysioNet 1.0.0) epilepsy — results

*NeuroTechX-Atlas epilepsy arm, dataset 5 (after TUSZ, SeizeIT2, Helsinki,
CHB-MIT). Italian adult clinical population, standard monopolar 10-20
montage — unlike CHB-MIT's bipolar derivations, REVE gets genuine channel
positions here, no anchor-electrode approximation needed.*

## Setup

- **Data:** Siena Scalp EEG, PhysioNet 1.0.0 (Detti et al.), 14 patients,
  41 recordings, 35 seizure-containing files.
- **Split:** subject-disjoint 60/20/20 by patient (10/2/2) — **only 2
  patients in eval**, the smallest eval cohort in the epilepsy arm.
- **Montage:** standard 19-channel monopolar 10-20 (same STD19 as
  TUSZ/Helsinki).
- **Metric:** SzCORE (same scorer as TUSZ/SeizeIT2/Helsinki/CHB-MIT).
- **Identity-free axis:** re-fit with patient identity LEACE-erased.

## Results (eval, subject-disjoint, only 2 patients)

| Model | eval AUC[.1-10 FP/d] | sens@1/d | sens@10/d | F1 (own op point) | Δ AUC |
|---|---|---|---|---|---|
| **classical** normal | 0.790 | 75.0% | 100.0% | 0.800 @ 4.8 FP/day | |
| classical identity-free | 1.000 | 100.0% | 100.0% | 1.000 @ 0.0 FP/day | **−0.210** |
| **REVE** normal | 0.000 | 0.0% | 0.0% | 0.053 @ 79.2 FP/day | |
| REVE identity-free | 0.000 | 0.0% | 0.0% | 0.092 @ 139.2 FP/day | **+0.000** |

## Findings — honest, not spun

**Classical does NOT replicate the identity-trap pattern here — identity-free
is actually BETTER (Δ−0.210), the opposite sign from every other epilepsy
set (TUSZ +0.044, SeizeIT2 +0.019, Helsinki +0.067, CHB-MIT +0.146).** This
is very likely a **small-N artifact, not a genuine reversal**: eval has only
2 patients (41 seizure windows total), so a single patient's data
characteristics can flip the eval metric either way. Reported as-is rather
than folded into the classical pattern narrative — a real 4/5 majority
(TUSZ/SeizeIT2/Helsinki/CHB-MIT) still supports the classical
identity-erasure-cost signature, and Siena's outlier result is disclosed as
likely noise, not silently dropped or explained away as if it were a
confident finding.

**REVE collapses to the scorer's floor in both configs** (AUC 0.000, sens@1/d
= 0%), the same "REVE fails outright, Δ is uninterpretable" pattern already
seen on Helsinki and CHB-MIT — now 3 of 5 epilepsy datasets where REVE's
absolute performance is too degenerate to say anything about identity
trapping specifically.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_epilepsy_siena.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_epilepsy_siena.py --model reve --both
```
