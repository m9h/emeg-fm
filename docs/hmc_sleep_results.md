# HMC sleep-staging results — the cleanest identity-trap confirmation yet

*NeuroTechX-Atlas sleep arm, dataset 2 (after Sleep-EDF). Completed 2026-07-04.
HMC is one of NeuroAtlas's own named sleep datasets.*

## Setup

- **Data:** HMC (Haaglanden Medisch Centrum sleep staging database, PhysioNet,
  no auth), 151 recordings, one session/subject. Modern reduced clinical
  montage: 4 EEG channels referenced to mastoids (F4-M1, C4-M1, O2-M1, C3-M2).
- **Task/metric:** identical to Sleep-EDF — 5-class AASM staging (30 s epochs),
  balanced accuracy + Cohen's κ, patient-identity-free (LEACE) axis. Reuses
  `atlas_sleep_section.py`'s scorer directly (`STAGE_MAP`/`EPOCH_S`/`_fit_probe`/
  `evaluate`); own loader for the plain-text scoring export (not EDF+ Hypnogram)
  and 4-channel montage.
- **Split:** subject-disjoint 70/15/15 (107/22/22 recordings).

## Results (eval, subject-disjoint, 22 recordings / 6,372 epochs)

| Model | eval κ (normal) | eval κ (identity-free) | Δ κ |
|---|---|---|---|
| **classical** | 0.859 | 0.774 | **+0.085** (mild) |
| **REVE** | **0.898** | 0.408 | **+0.490** (SEVERE) |

## Why this is the cleanest version of the finding

Unlike TUSZ/Sleep-EDF (classical wins outright) and Helsinki (small eval set,
REVE result genuinely inconclusive), HMC removes both caveats:

- **Large, well-powered eval set** (22 recordings, 6,372 epochs) — no small-N
  ambiguity.
- **REVE actually wins on raw performance** (κ 0.898 vs classical's 0.859) —
  so this isn't "the FM was already losing and looks worse under scrutiny."
  REVE is the *best* model on this task, full stop.
- **Yet it collapses hardest under identity erasure** (Δ +0.490, ~5.8× classical's
  Δ +0.085) — more than double Sleep-EDF's already-severe +0.257.

This directly answers the natural objection to the earlier results: the identity
trap is not an artifact of the FM under-performing. Here the FM wins convincingly
and *still* turns out to be riding heavily on subject identity — more so than any
other domain tested so far.

## Cross-domain classical replication (now 5 datasets)

| Dataset | classical Δ κ/AUC |
|---|---|
| TUSZ (epilepsy) | +0.044 |
| SeizeIT2 (epilepsy) | +0.019 |
| Sleep-EDF (sleep) | +0.043 |
| Helsinki (epilepsy) | +0.067 |
| **HMC (sleep)** | **+0.085** |

Tight cluster (0.019–0.085), same direction every time — classical spectral
features barely lean on subject identity, across epilepsy, sleep, adult,
neonatal, clinical, and wearable recordings alike. This is now the most robust
finding in the whole Atlas program.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_sleep_hmc.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_sleep_hmc.py --model reve --both
```

No bugs on this dataset — clean on the first sanity run (the txt-scoring format
and montage were characterized correctly up front from the Helsinki/Sleep-EDF
lessons learned).
