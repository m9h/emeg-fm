# Dataset-count milestone: EMEG-FM program vs NeuroAtlas's 42

*Tracking whether our combined open-data EEG-FM benchmarking work has processed
more distinct datasets, with real persisted results, than NeuroAtlas's 42
(arXiv:2605.14698). Checked 2026-07-04, FINAL update 2026-07-05 after the TCM
coverage-extension sweep completed all three paradigms.*

## The tally (final)

| Source | Distinct datasets (real results) | Evidence |
|---|---|---|
| **MOABB identity-free leaderboard** (union: MI pooled+per-trial, TCM MI/ERP/SSVEP, REVE ERP/SSVEP pooled+per-trial) | **111** | `results/moabb_fmscope/{leaderboard_leftright*,tcm_pertrial*,leaderboard_erp*,leaderboard_ssvep*}.csv` |
| **NeuroTechX Atlas** (non-MOABB): TDBRAIN, LEMON, HBN, Alljoined, TUSZ, SeizeIT2, Helsinki, Sleep-EDF, HMC | **9** | this session + `~/dev/meeg-brain-age-benchmark-paper` |
| **Total distinct datasets processed** | **120** | |
| **NeuroAtlas** | 42 | arXiv:2605.14698 |

**We exceed NeuroAtlas's dataset count by 78 (120 vs 42, ~2.9×).**

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

Still not wired: 4 candidate NEMAR CogNeuro datasets (ds002718, ds004883,
ds003061, ds006018 — found, not yet integrated into the CogNeuro section).

## Caveat

This is a **dataset-count comparison**, not a claim of matching NeuroAtlas's domain
breadth 1:1 — see `docs/atlas_datasets.md` for the honest domain-by-domain
coverage map (we're strong on BCI breadth via MOABB, behind on non-Temple
epilepsy and most of their 15 sleep cohorts, ahead on CogNeuro/Brain-to-Image/
identity-free axes they don't have at all).
