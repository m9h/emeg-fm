# SEED-family identity-trap literature grounding

*Written 2026-07-18, before we have data access to SEED-V/SEED-VIG (both
gated behind BCMI Lab application/license — see `docs/truenas_data_handoff.md`
NEED E in the hippy-feat repo). Purpose: check whether the published
literature already shows the subject-identity-trap signature our program
has found across TUSZ/CHB-MIT/HMC/Sleep-EDF/MOABB-MI/ERP-CORE, so we know
what to expect and have baseline numbers to compare against once the data
lands and we can run our own LEACE-based audit.*

## What we found

**A large, consistently-reported gap between subject-dependent and
subject-independent evaluation protocols on SEED and SEED-VIG:**

| Dataset | Subject-dependent acc | Subject-independent (LOSO) acc | Gap |
|---|---|---|---|
| SEED | 90.4–96.9% | 79.95–84.16% | 10–15 points |
| SEED-VIG | 95.11% | 78.43% | **~17 points** (largest of the two) |

Domain-adaptation methods designed to close this gap recover only 3–9 points
of it under subject-/session-independent settings — the residual gap only
makes sense if there's a real subject-dependent effect being partially
recovered, not fully.

**The original SEED/SEED-VIG authors already treat subject variability as a
central confound, not a footnote.** Ma, Zheng, et al., "Reducing the Subject
Variability of EEG Signals with Adversarial Domain Generalization" (Zheng is
a SEED/SEED-VIG co-author) frames cross-subject drift as EEG emotion
recognition's core obstacle to real-world deployment, and uses adversarial
domain generalization specifically to strip subject-specific signal from the
representation — conceptually adjacent to (though not identical to) our
LEACE-based linear erasure.

**The broader domain-adaptation literature on SEED/SEED-IV** (PGCN, RGNN,
genetically-optimized PDPL, region-aware spatiotemporal domain
generalization) treats "subject-independent EEG emotion recognition" as a
distinct, harder subfield — its entire premise is that subject-dependent
results don't transfer across subjects, i.e. that subject identity is doing
real work in the subject-dependent numbers.

## What this does NOT establish

**Nobody has run an actual LEACE-style (or equivalent linear
subject-identity-erasure) audit on SEED/SEED-V/SEED-VIG.** Every result
above is a subject-dependent-vs-LOSO accuracy gap — suggestive of an
identity trap, in the same spirit as our program's other findings, but not
the same rigorous decomposition. A large subject-dependent/independent gap
is consistent with two different (non-exclusive) explanations:

1. **Identity leakage**: pooled/within-subject splits let a classifier partly
   recognize *which subject* a trial came from and use subject-specific
   quirks as a shortcut to the label — our program's usual finding.
2. **Genuine cross-subject physiological variability**: emotional/vigilance
   EEG signatures may just differ substantially person-to-person in ways
   that are real neuroscience, not a leakage artifact — unlike, say, sleep
   staging, where the AASM criteria are explicitly designed to be
   subject-general.

Disentangling these two is exactly what our LEACE erasure audit would be the
first to attempt on this dataset family. If a subject-disjoint (LOSO) split
already exists in these papers and the erasure Δ is *additionally* large on
top of that, that's much stronger identity-trap evidence than the raw
dependent-vs-independent gap alone — the gap could shrink to near-zero once
subject identity is properly erased, or it could persist (pointing to
genuine physiological variability instead). We don't know which yet.

## What to do once data lands

Reuse the exact `atlas_sleep_section.py`/`atlas_epilepsy_section.py` pattern
(frozen REVE + classical, patient-disjoint split, LEACE-erase-and-refit,
report Δ) rather than inventing a new harness — SEED-V's 5-class emotion
labels and SEED-VIG's continuous PERCLOS vigilance score map onto our
existing classification (5-class, like AASM) and regression (if we add one)
evaluation patterns respectively. Compare our Δ directly against the
literature's reported subject-dependent/independent gap (10–15 pts SEED,
~17 pts SEED-VIG) as a sanity check that our pipeline reproduces the known
effect before trusting any *novel* part of the result (e.g. whether REVE
traps harder than classical here, extending the mixed sleep/epilepsy record
to a new domain).

## Sources

- Multiple SEED subject-dependent (90.4–96.9%) vs LOSO (79.95–84.16%) accuracy reports.
- SEED-VIG subject-dependent (95.11%) vs LOSO (78.43%) accuracy report.
- Ma, Zheng, et al., "Reducing the Subject Variability of EEG Signals with
  Adversarial Domain Generalization" — weilongzheng.github.io/publication/ma2019reducing/ma2019reducing.pdf
- Domain-adaptation literature on SEED/SEED-IV subject-independent emotion
  recognition (PGCN, RGNN, genetically-optimized PDPL, region-aware
  spatiotemporal domain generalization).
