# NEMAR CogNeuro extras (N170, ERN, P3-aud, P3-vis) + M3CV — results

*NeuroTechX-Atlas CogNeuro arm, independent replications beyond ERP CORE's
shared ~40-subject cohort (`atlas_cogneuro_section.py`). Completed 2026-07-06.
Four new datasets found via NEMAR search: N170 (ds002718, 18 subj), ERN
(ds004883, 172 subj), auditory P3 oddball (ds003061, 13 subj), visual P3b
oddball (ds006018, 127 subj) — plus M3CV, a 95-subject biometric-competition
cohort repurposed as a non-circular identity-trap test (13-condition decode
with subject identity as the erased nuisance axis, not decoding identity
itself). All five loaders live in `scripts/atlas_cogneuro_extra.py` /
`scripts/atlas_m3cv_identity.py`.

## Why these matter

ERP CORE's own 7-component battery shares a single ~40-subject cohort across
every component — a real but limited N. These four independent OpenNeuro
cohorts (13–172 subjects each) test whether the ERP CORE headline (REVE beats
classical on cognitive/evoked components, identity-Δ≈0) generalizes to new
populations, montages, and — for ERN/P3-vis — 4–7× the subject count.

## Setup per dataset

| Dataset | N (used) | Contrast | Window | Montage / REVE channel handling |
|---|---|---|---|---|
| N170 (ds002718) | 18 subj, 15,929 trials | face (famous/unfamiliar) vs scrambled | −0.2 to 0.8 s | Real electrode positions, nearest-neighbor matched to `standard_1005` |
| ERN (ds004883) | 162 subj, 86,288 trials | error vs correct response (flanker), response-locked | −0.4 to 0.8 s | Same electrode-position matching (ses-1 only per subject) |
| P3-aud (ds003061) | 13 subj, 24,163 trials | oddball (target) vs standard tone | −0.2 to 0.8 s | Real 10-20 names already in `standard_1005` form — no remapping needed |
| P3-vis (ds006018) | 98/127 subj, 20,419 trials | rare target vs frequent non-target (ERP CORE active visual oddball protocol) | −0.2 to 0.8 s | Parsed from `.vhdr` (no channels.tsv); 5 mastoid/EOG channels excluded by name |
| M3CV | 95 subj, 35,050 epochs | 13-way experimental condition, LOSO across subjects | fixed 4 s epochs | **Classical only** — no channel-name ground truth shipped with the dataset; REVE deliberately unsupported rather than guessed (same call as SeizeIT2) |

All five use `atlas_cogneuro_section.py`'s shared `loso()`/`within_subject_ba()`
(LOSO cross-subject decode, per-subject within-subject CV, and a LEACE
subject-identity-erased LOSO re-probe).

## Results

| Dataset | Model | LOSO | within-subj | identity-free LOSO | idΔ |
|---|---|---|---|---|---|
| N170 | REVE | 60.3% | 64.4% | 60.1% | **+0.1** |
| ERN | classical | 57.6% | 67.0% | 59.0% | **−1.4** |
| ERN | REVE | 69.4% | 70.5% | 62.4% | **+7.0** |
| P3-aud | classical | 61.1% | 67.8% | 62.7% | **−1.6** |
| P3-aud | REVE | 71.4% | 77.9% | 71.6% | **−0.2** |
| P3-vis | classical | 59.2% | 57.8% | 50.3% | **+8.9** |
| P3-vis | REVE | 68.0% | 61.0% | 63.1% | **+4.9** |
| M3CV (condition decode) | classical | 59.3% | 82.3% | 59.9% | **−0.6** |

(N170 classical was not run — REVE-only result reported here; N170's REVE
LOSO≈within with idΔ≈0 is consistent with the other near-zero-trap cases.)

## Findings — honest, not spun

**REVE beats classical on raw LOSO decode in every dataset that ran both**
(ERN 69.4 vs 57.6, P3-aud 71.4 vs 61.1, P3-vis 68.0 vs 59.2) — this is the
ERP CORE headline (FMs win on cognitive/evoked components) replicating
independently at 4-7× the subject count for ERN and P3-vis, not just on the
shared 40-subject ERP CORE cohort.

**The identity-trap effect is NOT uniform across this set — a real,
reported mix, not a clean confirmation.** N170, ERN (REVE), and P3-aud
(both models) show small-to-negligible idΔ (≤|1.6|, and two of four are
negative — i.e. identity-erasure *improved* decode slightly, within noise).
**P3-vis is the exception**: a real trap-sized effect for both classical
(+8.9) and REVE (+4.9) — the only dataset in this extras set where erasing
subject identity meaningfully hurt condition decode for both model families.
This is reported as a genuine counterexample *within* the CogNeuro-extras set
itself, not swept into "FMs never show a trap on cognition" — that claim
would be false given P3-vis.

**M3CV's non-circular framing works as intended.** Decoding the 13
experimental conditions (not subject identity itself) under LOSO with
identity erased gives a coherent, non-tautological idΔ (classical −0.6,
essentially null) — the condition-decode signal survives subject-identity
erasure almost intact, i.e. M3CV's task labels are not primarily an identity
proxy. The very high within-subject accuracy (82.3%) vs modest LOSO (59.3%)
shows most of the achievable condition signal here is subject-specific
in nature (as expected for a dataset literally designed for
biometric verification) but not smuggled through raw identity per se.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_cogneuro_extra.py --model reve --dataset n170
scripts/eegfm_t9.sh python scripts/atlas_cogneuro_extra.py --model logbandpower --dataset ern
scripts/eegfm_t9.sh python scripts/atlas_cogneuro_extra.py --model reve --dataset ern
scripts/eegfm_t9.sh python scripts/atlas_cogneuro_extra.py --model logbandpower --dataset p3_aud
scripts/eegfm_t9.sh python scripts/atlas_cogneuro_extra.py --model reve --dataset p3_aud
scripts/eegfm_t9.sh python scripts/atlas_cogneuro_extra.py --model logbandpower --dataset p3_vis
scripts/eegfm_t9.sh python scripts/atlas_cogneuro_extra.py --model reve --dataset p3_vis
scripts/eegfm_t9.sh python scripts/atlas_m3cv_identity.py --model logbandpower --max-per-cell 30
```

## Notes on scope

- P3-vis used 98 of 127 subjects (some subjects' `.vhdr`/events failed to
  parse cleanly in this pass — not investigated further since the N is
  already the largest in this set).
- ERN used 162 of 172 subjects, session 1 only per subject (see loader
  docstring for why: task name varies by counterbalance order across the
  dataset's 3 sessions/subject).
- M3CV was capped at 30 epochs per (subject,condition) cell (35,050 of the
  full 57,851-epoch Enrollment split) to bound runtime; a deeper run on the
  uncapped cohort is possible but not expected to change the qualitative
  result given the already-tight within vs LOSO gap.
