# MDD (Mumtaz 2016) depression detection — results

*New task domain: clinical psychiatric diagnosis (MDD vs healthy), and one
of OpenEEGBench's 12. 64 subjects (healthy "HS\*" + MDD "MDDS\*"), binary
classification, 19-channel standard monopolar 10-20 montage, windows mixed
across 3 task conditions (P300, eyesClosed, eyesOpen — reproduces
OpenEEGBench's own task framing rather than a narrower resting-state-only
protocol).*

## Setup

- 46 train / 9 dev / 9 eval subjects, subject-disjoint via
  **stratified** splitting (see caveat below — this dataset genuinely
  needed it).
- Pulled via braindecode's own `BaseConcatDataset.pull_from_hub`
  (`braindecode/mdd_mumtaz2016`, open HF mirror).

## A real bug this loader caught: subject-id-encodes-label

MDD's subject ids are literally `HS<n>` (healthy) / `MDDS<n>` (MDD) — the
label is baked into the id. The naive lexicographic sort-and-slice split
used elsewhere in this program (`subject_splits`) sorts `"HS1" < "HS2" <
... < "MDDS1" < ...` alphabetically, so slicing off the *last* 15% for eval
put **100% MDDS (single-class) subjects in eval** — verified directly.
Balanced-accuracy still reported a plausible-looking 81-89%, but Cohen's κ
was a degenerate exact 0.000 (undefined agreement-above-chance on a
single-class eval set) — the giveaway that something was wrong. Fixed with
`stratified_subject_splits()` (split within each label group, then union),
added to the shared `atlas_hf_braindecode.py` helper since any dataset
where subject id correlates with label is at risk of the same bug.

## Results (eval split, after the stratification fix)

| Model | Config | balanced-acc | κ |
|---|---|---|---|
| classical (log-band-power) | normal | 86.1% | 0.726 |
| classical (log-band-power) | identity-free | 65.7% | 0.312 |
| REVE | normal | 79.2% | 0.591 |
| REVE | identity-free | 50.0% | **0.000** |

Identity-free Δ(eval κ): classical **+0.414**, REVE **+0.591** — the
**largest identity-free Δ recorded anywhere in this program**, surpassing
DCSM's previous record (Δ+0.557, sleep staging).

## Finding — honest, and genuinely important

**Both models show real signal normally** (classical κ 0.726, REVE κ
0.591 — this is not a floor-collapse artifact like Helsinki/CHB-MIT/Siena;
there is a real classifier here before erasure). **REVE then collapses
completely to chance** (κ exactly 0.000, balanced-acc exactly 50.0% — total
collapse, not partial) **under identity erasure**, and classical also drops
substantially (Δ+0.414, its second-largest drop in the program after this).

**Why this is the biggest Δ yet, and probably not a coincidence:** MDD
diagnosis is a **subject-level trait** — constant for a given person across
every recording, unlike sleep stage, seizure state, or arithmetic-engagement,
which all vary *within* a subject over time. For a trait-classification task,
"correctly identifying which subject this is" and "correctly guessing the
label" are almost the same problem by construction — a classifier riding on
subject identity gets the label almost for free, whereas for a state task,
knowing the subject's identity tells you nothing about which time-varying
state they're currently in. This gives a clean, testable hypothesis for the
whole program going forward: **trait-classification tasks (diagnosis,
clinical status) should show systematically larger identity-free Δ than
state-classification tasks (sleep stage, seizure, arithmetic, motor
imagery)** — consistent with DCSM (state task, large but smaller Δ+0.557)
being the next-largest, and every near-zero-Δ result in the program
(Ear-EEG, ISRUC, arithmetic, BCI2020-3) being state tasks. This is the
first dataset in the program to test a pure trait-classification task
directly, and it fits the hypothesis cleanly — but it's one dataset, not
proof; the natural next test is TDBRAIN's planned MDD-vs-ADHD trait audit
(task #60 in the project tracker), which would be a second, independent
trait-classification data point.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_depression_mdd.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_depression_mdd.py --model reve --both
```
