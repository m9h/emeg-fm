# Dreem Open Datasets (DOD-H / DOD-O) — sleep-staging results

*NeuroTechX-Atlas sleep arm, dataset 3 (after Sleep-EDF, HMC). Completed
2026-07-08. Two cohorts from the same paper (Guillot et al. 2020): DOD-H (25
healthy subjects) and DOD-O (56 obstructive-sleep-apnea patients), kept
separate since they use different montages (12-ch vs 8-ch EEG).*

## Setup

- **Data:** Dreem Open Datasets, Zenodo record 15900394. HDF5-native, with a
  pre-epoched hypnogram (epoch *i* == seconds `[i*30, (i+1)*30)`) -- no
  annotation parsing or wake-cropping needed, unlike the EDF+-based Sleep-EDF/
  HMC loaders.
- **Split:** subject-disjoint 70/15/15 (DOD-H: 19/3/3, DOD-O: 40/8/8).
- **Models:** classical (log-band-power) and REVE (anchor-electrode name
  mapping: `C3-M2`→`C3`, `F3-F4`→`F3`, same convention as HMC).
- **Metric:** balanced accuracy + Cohen's kappa, same 5-class AASM task as
  Sleep-EDF/HMC.
- **Identity-free axis:** re-fit with subject identity LEACE-erased.

## Results

| Cohort | Model | eval bal-acc | eval κ (normal) | eval κ (id-free) | Δκ |
|---|---|---|---|---|---|
| DOD-H (25 subj) | classical | 69.9% | 0.657 | 0.424 | **+0.233** |
| DOD-H (25 subj) | REVE | 66.3% | 0.521 | 0.400 | **+0.121** |
| DOD-O (56 subj) | classical | 64.9% | 0.555 | 0.390 | **+0.165** |
| DOD-O (56 subj) | REVE | 69.0% | 0.572 | 0.472 | **+0.100** |

## Findings — honest, not spun

**Classical beats REVE on raw staging in DOD-H (0.657 vs 0.521); REVE
slightly beats classical in DOD-O (0.572 vs 0.555, close).** No consistent
FM-wins-on-raw-perf story here, unlike the CogNeuro-extras set.

**The identity-trap effect is REVERSED here vs Sleep-EDF and HMC.** Across
all 4 sleep results now on record:

| Dataset | classical Δκ | REVE Δκ | Which model traps harder? |
|---|---|---|---|
| Sleep-EDF | +0.043 | +0.257 | REVE |
| HMC | +0.085 | +0.490 | REVE (cleanest case so far) |
| DOD-H | **+0.233** | +0.121 | **classical** |
| DOD-O | **+0.165** | +0.100 | **classical** |

On Sleep-EDF and HMC, REVE showed the larger identity-erasure cost (the
"identity trap" headline for those two). On **both** DOD cohorts, it's the
**opposite**: classical's kappa drops more under identity erasure than
REVE's does. This is reported as a genuine, unresolved split within the
sleep domain -- not evidence for "FMs always trap harder," which the
now-4-dataset sleep record does not support as a universal claim. A
plausible (untested) explanation is DOD's smaller per-cohort N (25/56 vs
Sleep-EDF's 30/HMC's 22 recordings, but many more total epochs per subject
in DOD given full-night 250Hz recording at 12/8 channels) changing which
model's decision surface leans more on subject-specific band-power patterns
vs subject-specific higher-order structure -- worth a dedicated follow-up if
the sleep-domain identity-trap thesis is pursued further, rather than
asserting a direction here.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_sleep_dod.py --which dodh --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_sleep_dod.py --which dodh --model reve --both
scripts/eegfm_t9.sh python scripts/atlas_sleep_dod.py --which dodo --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_sleep_dod.py --which dodo --model reve --both
```
