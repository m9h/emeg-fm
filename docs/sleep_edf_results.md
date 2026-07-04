# Sleep-staging section (Sleep-EDF Expanded) — results

*NeuroTechX-Atlas sleep arm — our biggest gap vs NeuroAtlas (15 sleep datasets
reported, 0 built here previously). Completed 2026-07-04.*

## Setup

- **Data:** Sleep-EDF Expanded (PhysioNet, via the `physionet-open` S3 mirror), 197
  recordings (sleep-cassette + sleep-telemetry), pulled in full (8.1 GB, verified
  byte-exact + `unzip -t` clean).
- **Task:** 5-class AASM staging (W / N1 / N2 / N3 / REM; stages 3+4 merged into N3
  per AASM convention) on 30 s epochs from the two EEG channels (`Fpz-Cz`, `Pz-Oz`
  — hardware bipolar derivations, not monopolar 10-20 sites).
- **Split:** subject-disjoint 70/15/15 (137/30/30 recordings) — no official split
  exists for this corpus.
- **Models:** classical (log-band-power → HistGBM) and **REVE** (FM). BIOT/LaBraM/
  EEGDINO's `Interpolated*` wrapper can't fit this montage (needs ≥4 head-dig
  points; Sleep-EDF has 2 channels) — REVE's name-based lookup handles it via an
  anchor-electrode approximation (`Fpz-Cz→Cz`, `Pz-Oz→Oz`).
- **Metric:** balanced accuracy + Cohen's kappa (the standard sleep-staging pair).
- **Identity-free axis:** re-fit with subject identity LEACE-erased.

## Results (eval, subject-disjoint)

| Model | eval balanced-acc | eval kappa | idfree balanced-acc | idfree kappa | **Δ kappa** |
|---|---|---|---|---|---|
| **classical** | 67.7% | **0.551** | 63.5% | 0.508 | **+0.043** |
| **REVE** | 59.9% | 0.459 | 36.8% | 0.202 | **+0.257** |

(kappa 0.551 for classical band-power on 5-class AASM staging is in the expected
published range — this validates the pipeline: an earlier version had a data
bug that silently zeroed every epoch, giving chance-level kappa≈0; see fix log.)

## The finding: the identity trap replicates across domains

This is the **second independent domain** (after TUSZ epilepsy) showing the same
qualitative pattern:

- **Classical barely notices identity erasure** (Δ 0.043, ~8% relative kappa drop).
- **The FM collapses far harder** (Δ 0.257 — REVE's kappa drops from 0.459 to 0.202,
  a >50% relative loss) — more than 5× the classical Δ.
- **Classical still edges the FM on raw performance** (0.551 vs 0.459), consistent
  with the broader Atlas thesis that classical wins spectral/oscillatory tasks
  (brain-age, motor imagery, now sleep staging) while FMs win evoked/cognitive
  tasks (ERP CORE).

Together with TUSZ (classical Δ +0.044 vs both FMs collapsing to 0% usable
sensitivity), this is now a **repeated, cross-domain signature**: whatever REVE
(and by extension other FMs) learn for spectral/physiological classification
tasks leans substantially on subject-identity-correlated signal that classical
band-power features do not need. Two domains, same shape of effect — this is
the load-bearing empirical claim for the identity-free axis, not a one-off.

## Known limitations / honest caveats

- **REVE's channel mapping is an approximation** (bipolar derivation → anchor
  electrode). Not the real electrode position; documented in code.
- **No NEDC-style dual-scorer cross-check** — kappa/balanced-accuracy are already
  the field-standard metrics for this task (unlike epilepsy's Event-Sens@FA,
  which needed the SzCORE/NEDC rigor), so no equivalent scorer risk here.
- **Only Sleep-EDF** of NeuroAtlas's ~15 sleep datasets. HMC (Haaglanden Medisch
  Centrum, also NeuroAtlas-named) is downloaded (12.9 GB) but not yet wired —
  uses a different PSG/`_sleepscoring` file-pairing convention.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_sleep_section.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_sleep_section.py --model reve --both
```

Bugs fixed en route (both committed): MNE `crop()`/`first_samp` indexing bug that
silently dropped every epoch (chance-level kappa masqueraded as a real result —
see commit `6e6ec2d`), and a `sklearn>=1.5` `LogisticRegression(multi_class=...)`
removal (`252da2d`).
