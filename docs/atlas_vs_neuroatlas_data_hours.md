# Data-volume (minutes/hours) comparison vs NeuroAtlas — a fairer metric than dataset count

*Dataset **count** (125 vs 42, `docs/atlas_vs_neuroatlas_coverage.md`) rewards breadth of
distinct tasks/domains. NeuroAtlas's headline "~260k hours" rewards raw recorded volume,
dominated by a handful of massive clinical mega-cohorts (sleep PSG, long-term epilepsy
monitoring). This doc computes our actual processed-data-hours the same way, so the two
programs can be compared on the axis NeuroAtlas actually advertises. Computed 2026-07-06.*

## Method (what's measured, and how)

- **MOABB (67 of 68 dataset×paradigm cells resolved)**: `n_windows` (or `n_trials`) from
  our own leaderboard/TCM result CSVs (deduped per dataset+paradigm across all
  models/scorings that touched it — never double-counted just because REVE and TCM both
  ran on the same recording) × each MOABB dataset class's own `interval` attribute
  (trial length in seconds, pure metadata, no download needed). `DemonsP300` (1 cell)
  didn't resolve to a MOABB class name — negligible, omitted.
- **Continuous-recording Atlas sets** (TUSZ, SeizeIT2, Helsinki, Sleep-EDF, HMC): real
  on-disk recording duration via header-only MNE reads (no data loaded) on a random
  20-file sample per dataset, average duration × the **actual n_eval we scored**
  (not the full corpus sitting on disk — e.g. Helsinki has 79 recordings downloaded but
  our eval split used 15; TUSZ eval dir has 865, matching our n_eval exactly).
- **Pre-epoched CogNeuro-extras + M3CV** (N170, ERN, P3-aud, P3-vis, M3CV): epochs actually
  used in the full run × epoch window length — same accounting convention as MOABB.
- **Brain-age** (TDBRAIN/LEMON/HBN): n_subjects actually used (1285 / 210 / 2537, the last
  from `meeg-brain-age-benchmark-paper` results — 2537 unique subjects) × each cohort's
  published resting-state protocol length (TDBRAIN 2min EC+2min EO=4min; LEMON 16×1min
  EC/EO blocks=16min; HBN 2.5min EO+2.5min EC=5min — protocol durations, not measured
  on-disk, since we don't hold the raw recordings for all of these locally).
- **Alljoined**: 1-subject smoke test — negligible, omitted (<1 hour).

## Result

| Component | Hours |
|---|---|
| MOABB identity-free leaderboard (67 dataset×paradigm cells) | 993.4 |
| Continuous-recording epilepsy+sleep (TUSZ+SeizeIT2+Helsinki+Sleep-EDF+HMC) | 2756.0 |
| Pre-epoched CogNeuro-extras + M3CV (N170+ERN+P3aud+P3vis+M3CV) | 84.5 |
| Brain-age (TDBRAIN+LEMON+HBN) | 353.1 |
| **Total** | **≈4,187 hours (≈174.5 days)** |
| **NeuroAtlas** (arXiv:2605.14698, their own headline) | **≈260,000 hours** |

**We are at ≈1.6% of NeuroAtlas's total data-hours — they exceed us ≈62×on this axis.**

## Why the two numbers (dataset-count vs data-hours) point opposite directions — honestly

This is not a case where one number is "the real one" and the other is spin — they measure
genuinely different things, and NeuroAtlas's own 260k-hour figure is itself dominated by
their **sleep arm** (∼15.8k patients, ∼201k of their ∼260k hours — see `docs/atlas_datasets.md`
§C) — full-night PSG recordings that are individually 100-1000× longer than a single BCI
trial or ERP epoch. Their epilepsy arm alone (∼58k h) also dwarfs any of our per-dataset
totals because long-term seizure monitoring runs for days per patient (our own SeizeIT2
measurement here — ∼1,912 h from 671 recordings, ∼171 min avg/recording — makes the same
point: epilepsy/sleep hours accumulate fast per subject in a way BCI/ERP trials never will).

**Their design optimizes for volume from few, deep, clinical-monitoring corpora. Ours
currently optimizes for breadth across many, shallow, task-diverse corpora** (a single MI
trial is 4-5s; a single ERP epoch is ~1s; even our largest single-subject processed chunks
are minutes, not days). Both are legitimate benchmark-construction choices, but they are
not the same thing, and the 125-vs-42 dataset-count comparison should not be read as "we
have more data" — we have more *distinct problems*, at a small fraction of the *volume*.

## What would close the hours gap fastest (if that becomes the goal)

Per-hour-added efficiency favors staging **long clinical monitoring cohorts**, not more
BCI/ERP datasets:
- **TUEG v2.0.2** (full 1.6TB Temple corpus, already staged/in-progress per task #72) —
  orders of magnitude more hours than TUSZ's eval-865 subset alone.
- **NSRR sleep cohorts** (SHHS 5,804/8,444 PSGs, MrOS 2,911/3,933, MESA 2,237/2,056 —
  see `~/dev/meeg-brain-age-benchmark-paper/BRAIN_AGE_DATASETS.md`) — each cohort alone
  would likely add more hours than our entire current total; gated behind a DAUA per
  dataset (~days each), not technically hard.
- **Full SeizeIT2 + Helsinki cohorts** (not just the eval-N splits scored here) — we're
  already holding the full corpora on disk in some cases; the eval-split accounting above
  understates what's actually downloaded.

None of this is needed to keep the dataset-count lead (125 vs 42 stands on its own) — it's
a separate lever, worth pulling only if "hours" becomes the metric that matters for a given
comparison or publication venue.
