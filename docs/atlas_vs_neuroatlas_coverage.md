# Dataset-count milestone: EMEG-FM program vs NeuroAtlas's 42

*Tracking whether our combined open-data EEG-FM benchmarking work has processed
more distinct datasets, with real persisted results, than NeuroAtlas's 42
(arXiv:2605.14698). Checked 2026-07-04.*

## The tally

| Source | Distinct datasets (real results) | Evidence |
|---|---|---|
| **MOABB identity-free leaderboard** (union: MI pooled+per-trial, TCM MI/ERP/SSVEP, REVE ERP/SSVEP pooled+per-trial) | **52** | `results/moabb_fmscope/{leaderboard_leftright*,tcm_pertrial*,leaderboard_erp*,leaderboard_ssvep*}.csv` |
| **NeuroTechX Atlas** (non-MOABB): TDBRAIN, LEMON, HBN, Alljoined, TUSZ, Sleep-EDF, SeizeIT2 | **7** | this session + `~/dev/meeg-brain-age-benchmark-paper` |
| **Total distinct datasets processed** | **59** | |
| **NeuroAtlas** | 42 | arXiv:2605.14698 |

**We already exceed NeuroAtlas's dataset count by 17 (59 vs 42).**

## What "processed" means here (methodology honesty)

A dataset counts if it has a **real, persisted result row** from an actual run —
not just downloaded, not just referenced. The MOABB 52 come from independent
identity-free-leaderboard work (a separate but related NeuroTechX project,
`project_identity_free_moabb_leaderboard`) that predates this session; folding it
into this comparison is a fair aggregation, not new/inflated work, since both
efforts are the same open EEG-FM benchmarking program under the same
organization.

## Coverage extension in progress

An additional sweep (`scripts/moabb_tcm_pertrial.py`, resumable — skips any
dataset already marked `status=ok`) is running across all three wired paradigms
(leftright/MI, erp/P300, ssvep) against the full registered dataset list per
paradigm, to push coverage further before reporting a final number. Log:
`/mnt/t9/epilepsy_runs/tcm_extend_orchestrator.log`.

Not yet counted (real but unfinished): Helsinki neonatal seizure EEG (downloading),
HMC sleep (downloaded, not wired), 4 candidate NEMAR CogNeuro datasets (ds002718,
ds004883, ds003061, ds006018 — found, not yet wired).

## Caveat

This is a **dataset-count comparison**, not a claim of matching NeuroAtlas's domain
breadth 1:1 — see `docs/atlas_datasets.md` for the honest domain-by-domain
coverage map (we're strong on BCI breadth via MOABB, behind on non-Temple
epilepsy and most of their 15 sleep cohorts, ahead on CogNeuro/Brain-to-Image/
identity-free axes they don't have at all).
