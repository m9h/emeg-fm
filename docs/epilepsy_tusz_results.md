# Epilepsy section (TUSZ v2.0.3) — results

*NeuroTechX-Atlas epilepsy arm. Full official patient-disjoint split, dual-scorer
(SzCORE primary + official Temple NEDC v6.0.0 cross-check). Completed 2026-07-03.*

## Setup

- **Data:** TUSZ v2.0.3 (Temple University Hospital Seizure Corpus), the official
  patient-disjoint split: **train 4667 / dev 1832 / eval 865** recordings, each with
  its `.csv_bi` seizure annotation. Continuous EEG → 10 s windows on the standard-19
  10–20 montage @ 200 Hz. A window is positive if it overlaps any `.csv_bi` seizure.
- **Models (frozen embedding → probe):** classical `logbandpower` (expert band-power/
  line-length → HistGBM); **REVE** and **EEGDINO** (braindecode 1.6.1 FMs, per-window
  embedding → balanced logistic probe). Same windows/labels for all — apples-to-apples.
- **Protocol (Sci-Rep 2026 s41598-026-41358-w):** fit on train, tune the operating
  point on dev, report on the blind eval. Never mix patients across splits.
- **Scorers:**
  - **SzCORE `timescoring`** (EpilepsyBench / Sci-Rep reference; 30 s/60 s tolerance,
    90 s refractory, 5 min max event) — **primary**. Reported: Sensitivity at 1 and
    10 FP/24 h, dev-tuned F1, and an Event-Sens@FA step-AUC over a clinical [0.1, 10]
    FP/day band.
  - **Official Temple NEDC v6.0.0** (`nedc_eeg_eval`) OVLP + TAES — **TUH-native
    cross-check**. Run on synthesized ref/hyp `.csv_bi` from the cached scores.
- **Our added axis:** re-fit the probe with **patient identity LEACE-erased** →
  identity-free Sensitivity/F1. Seizure EEG is highly patient-specific → prime
  identity-trap territory.

## Results (eval = 865 patient-disjoint recordings)

| Model / config | Sz sens@1 FP/d | sens@10 FP/d | Sz F1 | NEDC-OVLP sens | OVLP FA/24h | TAES sens |
|---|---|---|---|---|---|---|
| **classical** normal | 14.1% | **40.9%** | 0.534 | 63.5% | 235 | 43.2% |
| classical identity-free | 4.1% | **25.0%** | 0.483 | 54.6% | 214 | 29.1% |
| **REVE** normal | 3.5% | 19.7% | 0.430 | 36.5% | 162 | 13.2% |
| REVE identity-free | 0.0% | **0.0%** | 0.337 | 100%† | 126 | 41.6% |
| **EEGDINO** normal | 0.0% | 38.2% | 0.547 | 42.2% | **42** | 18.8% |
| EEGDINO identity-free | 0.0% | **0.0%** | 0.337 | 100%† | 126 | 41.6% |

† OVLP "100%" for the identity-free FMs is the transparent **all-alarm** signature
(note the 126 FP/24 h); the robust metrics (sensitivity at a fixed FA budget, F1)
correctly show it is clinically useless.

## Two headlines

**1. The identity trap is dramatic on TUSZ — and FM-specific.** Both foundation
models **collapse to 0 % usable sensitivity** once patient identity is LEACE-erased
(sens@10 FP/24 h 19.7 %→0.0 % for REVE, 38.2 %→0.0 % for EEGDINO; F1 both →0.337).
**Classical is the only model that survives** erasure (40.9 %→25.0 %, still a working
detector). The FMs' seizure "detection" rides almost entirely on **patient-specific
signatures**, not seizure physiology. This is the identity trap at full force on
patient data — the opposite of clean ERP CORE (Δ≈0), and far larger for FMs than for
classical. It is the differentiated result the Atlas exists to show, here validated
under two independent scorers.

**2. No FM beats classical on the seizure task itself.** Under the proper SzCORE
scorer, **EEGDINO ≈ classical** on normal detection (F1 0.547 vs 0.534; sens@10 FP/24 h
38.2 % vs 40.9 %) — a tie, not a win. EEGDINO *is* the quietest detector by far
(42 vs 235 FP/24 h at its dev-tuned point), a genuine strength. REVE is clearly
weakest. So the newest FM *matches* classical but does not beat it, and none is
identity-robust. (An earlier number under a broken metric had EEGDINO at a spurious
0.626 "win" — corrected here.)

## Scorer correctness (two bugs found + fixed, unit-tested)

The identity-free FM results first appeared as a nonsensical **AUC = 1.000** ("perfect"
after erasure). Root-causing that surfaced two real bugs in the Event-Sens@FA metric,
both fixed in `epilepsy_scorer.py` (`tests/test_epilepsy_szcore.py`, 4 tests):

1. **Unreachable-FA extrapolation.** The AUC extrapolated the *maximum* sensitivity
   into FA budgets the classifier can't achieve (`left=sens[0]`). A degenerate
   all-alarm probe only operates at huge FP/day, yet scored a perfect 1.0. Fixed:
   sensitivity below the achievable-minimum FA is 0.
2. **Linear-interp fabrication.** For a bimodal-degenerate probe (curve points only at
   fa≈0/sens0 and fa≈250/sens1), `np.interp` drew a straight line across the empty FA
   gap and invented ~0.5 sensitivity at fa=10 where no operating point exists. Fixed:
   Event-Sens@FA is now a right-continuous **step function** — best sensitivity
   *achievable* at ≤ each FA budget — matching `sensitivity_at_fa_day`.

Because per-recording window scores are **cached** (`/mnt/t9/epilepsy_runs/scores/`),
every metric (both scorers, any FA range) is recomputed from cache with no re-embedding.

## Reproduce

```bash
# full split, per model (normal + identity-free in one embedding pass, caches scores):
scripts/eegfm_t9.sh python scripts/atlas_epilepsy_section.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_epilepsy_section.py --model reve --both
scripts/eegfm_t9.sh python scripts/atlas_epilepsy_section.py --model eegdino --both
# combined SzCORE + NEDC OVLP/TAES table from the caches (no re-embedding):
scripts/eegfm_t9.sh python scripts/epilepsy_final_report.py
```

Scorers: `scripts/epilepsy_scorer.py` (SzCORE wrappers + Event-Sens@FA), `scripts/
nedc_crosscheck.py` (official NEDC v6.0.0, at `/mnt/t9/nedc_eeg_eval/v6.0.0`). Anchor:
Sci-Rep 2026 s41598-026-41358-w. Related: `docs/MOABB_IDENTITY_TRAP_FM_VS_TCM.md`.
