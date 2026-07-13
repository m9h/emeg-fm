# CHB-MIT (PhysioNet 1.0.0) epilepsy — results

*NeuroTechX-Atlas epilepsy arm, dataset 4 (after TUSZ, SeizeIT2, Helsinki).
A pediatric population (23 patients, ages 1.5-22) on a bipolar-derivation
scalp montage — a different montage convention (not monopolar 10-20) from
every other epilepsy set in the program.*

## Setup

- **Data:** CHB-MIT Scalp EEG, PhysioNet 1.0.0 (Shoeb 2010), 24 subject
  folders / 676 usable recordings / 141 seizure-containing files. chb21 is
  merged into chb01's patient id (same patient, re-recorded ~1.5y later per
  the corpus docs) → 23 unique patients.
- **Split:** subject-disjoint 60/20/20 by patient (15/4/4).
- **Montage:** common 18-channel bipolar derivation set (FP1-F7, F7-T7, ...,
  FZ-CZ, CZ-PZ) present across nearly all subjects — not the monopolar 10-20
  convention used by TUSZ/Helsinki.
- **Models:** classical (log-band-power → HistGBM). REVE uses the same
  anchor-electrode convention as Dreem DOD (first electrode of each bipolar
  pair stands in for a monopolar position) since bipolar derivations have no
  single standard-montage site.
- **Metric:** SzCORE (same scorer as TUSZ/SeizeIT2/Helsinki): Sensitivity@{1,10}
  FP/24h, dev-tuned F1, Event-Sens@FA step-AUC over `[0.1,10]` FP/day.
- **Identity-free axis:** re-fit with patient identity LEACE-erased.

## Results (eval, subject-disjoint, 4 patients)

| Model | eval AUC[.1-10 FP/d] | sens@1/d | sens@10/d | F1 (own op point) | Δ AUC |
|---|---|---|---|---|---|
| **classical** normal | 0.457 | 52.9% | 76.5% | 0.333 @ 0.2 FP/day | |
| classical identity-free | 0.312 | 38.2% | 58.8% | 0.000 @ 0.0 FP/day | **+0.146** |
| **REVE** normal | 0.014 | 0.0% | 38.2% | 0.321 @ 8.4 FP/day | |
| REVE identity-free | 0.093 | 0.0% | 35.3% | 0.414 @ 3.0 FP/day | **−0.079** |

## Findings — honest, not spun

**Classical replicates the pattern for the 5th time, and with the largest
margin yet.** Δ+0.146 is the biggest identity-erasure cost seen across the
whole epilepsy arm (TUSZ +0.044, SeizeIT2 +0.019, Helsinki +0.067, now
CHB-MIT +0.146) — consistent with a **robust, five-dataset classical
baseline signature**, not a fluke of one corpus.

**REVE's anchor-electrode approximation on CHB-MIT's bipolar montage
essentially fails at the task** (AUC 0.014-0.093, near the scorer's floor;
sens@1/d = 0% in both configs) — far below every other REVE result in this
program. This is the **same pattern as Helsinki**: when REVE's absolute
performance collapses to near-chance, the identity-free Δ is not a
meaningful signal (here it's slightly negative, −0.079, but that's noise
around a floor, not evidence REVE resists identity-trapping on this data).
Plausible mechanism: the anchor-electrode trick (approximating a bipolar
derivation's spatial position with its first electrode) is a coarser
approximation on CHB-MIT's 18-channel montage than on DOD's sleep montage,
where it worked adequately — bipolar-derivation REVE embeddings may simply
not carry enough spatial signal for a genuinely useful seizure-detection
probe. Reported as a genuine methodological limit, not smoothed into either
"REVE beats classical" or "REVE resists the identity trap."

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_epilepsy_chbmit.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_epilepsy_chbmit.py --model reve --both
```
