# Dataset-count milestone: EMEG-FM program vs NeuroAtlas's 42

*Tracking whether our combined open-data EEG-FM benchmarking work has processed
more distinct datasets, with real persisted results, than NeuroAtlas's 42
(arXiv:2605.14698). Checked 2026-07-04, updated 2026-07-05 after the TCM
coverage-extension sweep completed all three paradigms, updated again
2026-07-06 after the NEMAR CogNeuro-extras + M3CV wave, updated again
2026-07-08 after Dreem DOD-H/DOD-O, updated again 2026-07-12 after Ear-EEG
(ds005178), updated again 2026-07-13 after CHB-MIT + Siena + ISRUC, updated
again 2026-07-16 after DCSM, updated again 2026-07-18 after FACED +
Arithmetic + BCI2020-3 + MDD.*

## The tally (current)

| Source | Distinct datasets (real results) | Evidence |
|---|---|---|
| **MOABB identity-free leaderboard** (union: MI pooled+per-trial, TCM MI/ERP/SSVEP, REVE ERP/SSVEP pooled+per-trial) | **111** | `results/moabb_fmscope/{leaderboard_leftright*,tcm_pertrial*,leaderboard_erp*,leaderboard_ssvep*}.csv` |
| **NeuroTechX Atlas** (non-MOABB): TDBRAIN, LEMON, HBN, Alljoined, TUSZ, SeizeIT2, Helsinki, Sleep-EDF, HMC, N170 (ds002718), ERN (ds004883), P3-aud (ds003061), P3-vis (ds006018), M3CV, DOD-H, DOD-O, Ear-EEG, CHB-MIT, Siena, ISRUC, DCSM, FACED, Arithmetic, BCI2020-3, MDD | **25** | this session + `~/dev/meeg-brain-age-benchmark-paper` |
| **Total distinct datasets processed** | **136** | |
| **NeuroAtlas** | 42 | arXiv:2605.14698 |

**We exceed NeuroAtlas's dataset count by 94 (136 vs 42, ~3.2×).**

**OpenEEGBench overlap update**: Arithmetic, BCI2020-3, and MDD bring us to
**8/12** OpenEEGBench datasets with real completed results (CHB-MIT, ISRUC,
BNCI2014-001, PhysionetMotorImagery, FACED, Arithmetic, BCI2020-3, MDD) —
and each opens a genuinely new task domain (mental workload, imagined
speech, clinical depression), not just an overlap checkbox. **MDD produced
the single largest identity-free Δ recorded anywhere in the program**
(REVE Δκ +0.591, complete collapse to chance) — see
`docs/mdd_depression_results.md` for the trait-vs-state hypothesis this
suggests. Remaining OpenEEGBench gap: SEED-V, SEED-VIG, TUAB, TUEV — all
confirmed gated even via braindecode's own HF mirrors (HTTP 401), unlike
these 3 (HTTP 200) — the BCMI application and open-ISIP-mirror TUAB/TUEV
pull remain the paths forward.

**Caveat — this is a dataset-COUNT metric, not a data-VOLUME metric.** On total processed
hours (the metric behind NeuroAtlas's own "~260k hours" headline), we are far behind, not
ahead: ≈4,187 hours vs their ≈260,000 (≈1.6%, they exceed us ~62×). See
`docs/atlas_vs_neuroatlas_data_hours.md` for the full computation and why the two numbers
point in opposite directions (their volume comes from a few massive clinical
sleep/epilepsy monitoring cohorts; ours comes from breadth across many shorter, more
diverse tasks).

Per-paradigm breakdown of the final MOABB coverage-extension sweep
(`scripts/moabb_tcm_pertrial.py`, full registry per paradigm):

| Paradigm | ok datasets (this sweep) |
|---|---|
| leftright (MI) | 22 |
| erp (P300) | 36 |
| ssvep | 8 |

(These are TCM's own counts; the 111 MOABB union also includes the earlier
REVE pooled/per-trial leaderboard runs on an overlapping but not identical set
— see `results/atlas_paper/moabb_leaderboard_summary.csv` for the full,
disk-verified per-method/per-paradigm breakdown.)

## What "processed" means here (methodology honesty)

A dataset counts if it has a **real, persisted result row** from an actual run —
not just downloaded, not just referenced. The MOABB 52 come from independent
identity-free-leaderboard work (a separate but related NeuroTechX project,
`project_identity_free_moabb_leaderboard`) that predates this session; folding it
into this comparison is a fair aggregation, not new/inflated work, since both
efforts are the same open EEG-FM benchmarking program under the same
organization.

## Coverage extension — COMPLETE (2026-07-05)

The sweep (`scripts/moabb_tcm_pertrial.py`, resumable — skips any dataset
already marked `status=ok`) ran to completion across all three wired paradigms
(leftright/MI, erp/P300, ssvep) against the full registered dataset list per
paradigm. Log: `/mnt/t9/epilepsy_runs/tcm_extend_orchestrator.log`. Failures
along the way were isolated per-dataset (bad downloads, missing optional deps,
too-few-subjects-for-CV, inconsistent channel sets across subjects) — none
systemic, all logged and skipped by design.

Also completed since the last update: Helsinki neonatal seizure EEG (3rd
epilepsy dataset), HMC sleep (2nd sleep dataset) — both wired, tested, and run
(classical + REVE, normal + identity-free). See `docs/helsinki_neonatal_results.md`,
`docs/hmc_sleep_results.md`.

Wired since (2026-07-06): all 4 candidate NEMAR CogNeuro datasets (N170
ds002718, ERN ds004883, P3-aud ds003061, P3-vis ds006018) + M3CV (95-subj
biometric-competition cohort, 13-condition decode with subject-identity as
the erased nuisance axis — see `scripts/atlas_m3cv_identity.py` module
docstring for why this framing is non-circular, unlike naively decoding
identity itself).

## Remaining gaps vs NeuroAtlas's 42 (what's still worth staging)

Per the domain cross-check in `docs/atlas_datasets.md` §C, our shortfall is
concentrated in two places, not spread evenly:

- **Non-Temple epilepsy**: Bonn, NMT — SeizeIT1 was investigated and confirmed
  genuinely unavailable (not a false-negative claim — re-checked on
  request). **CHB-MIT and Siena are now done** (Helsinki + SeizeIT2 + TUSZ +
  CHB-MIT + Siena — 5 of 7 NeuroAtlas epilepsy sets covered).
- **Sleep cohorts**: MASS — Sleep-EDF + HMC + Dreem DOD-H/DOD-O + ISRUC +
  **DCSM** now done (6 of ~7 named NeuroAtlas sleep sets); MASS needs an
  application (the last named sleep gap). DCSM (255 subjects, the largest
  single sleep cohort in the program) also produced the single largest
  identity-trap collapse in the whole sleep record (REVE Δκ+0.557, bigger
  than HMC's +0.490) — see `docs/dcsm_sleep_results.md`. Ear-EEG (ds005178)
  is an *additional* sleep dataset beyond NeuroAtlas's named list — a new
  acquisition-modality axis they don't have at all.
- **DREAMER** (emotion/affect) — the one NeuroAtlas BCI-arm dataset not in
  MOABB and not staged; needs a Zenodo application.

None of these are required to keep the dataset-count lead (132 vs 42 already
~3×) — they matter for closing the *domain-breadth* gap (epilepsy/sleep
depth), not the raw count.

## Caveat

This is a **dataset-count comparison**, not a claim of matching NeuroAtlas's domain
breadth 1:1 — see `docs/atlas_datasets.md` for the honest domain-by-domain
coverage map (we're strong on BCI breadth via MOABB, behind on non-Temple
epilepsy and most of their 15 sleep cohorts, ahead on CogNeuro/Brain-to-Image/
identity-free axes they don't have at all).
