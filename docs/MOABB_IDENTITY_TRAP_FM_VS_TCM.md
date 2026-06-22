# The identity trap is a pooling artifact: FM (REVE) vs classical control (Wang TCM) across MOABB

**TL;DR.** FMScope ("The Identity Trap in EEG Foundation Models") reports that erasing
subject identity from a frozen EEG-FM's features *raises* downstream task accuracy — the
"identity trap." We re-ran that subject-axis erasure across the **entire MOABB benchmark**
(motor imagery, P300/ERP, SSVEP) two ways — the **pooled** protocol the paper uses, and an
**honest per-trial** protocol — for **two feature families**: the **REVE** foundation model and
a classical **Wang Topographic Component Model (TCM)** control. The trap reproduces faithfully
under pooling (REVE traps on **34 datasets**, including **17/17** motor-imagery sets) and is
**near-absent per trial (1/41)**. Critically, the classical TCM control — which encodes far less
subject identity — shows the **same** pooled→per-trial collapse. A pathology that appears in a
near-identity-free control and vanishes when the sample isn't collapsed is a property of the
**metric**, not of foundation models.

This report and the companion manuscript (`paper/paper.tex`, built by `paper/build.sh`) are
generated from the committed audit CSVs in `results/moabb_fmscope/` — every number is tangled by
`paper/generate.py`, nothing hand-transcribed.

## Background: the trap and the metric

FMScope fits LEACE (Belrose 2023) to linearly remove the subject axis from a frozen model's
features, then re-decodes the task. An **identity-free lift** (task balanced accuracy rises after
erasure) is read as evidence the model's skill rode on subject leakage.

The published protocol scores erasure at the **recording** level: one recording per
`(subject, class)`, with that recording's per-trial predictions **mean-pooled** into a single
label. With only a handful of subjects per class, this collapses the sample to a few
high-variance points, and a linear probe's post-erasure accuracy swings wildly — easy to
manufacture a "lift." The **per-trial** protocol instead makes every window its own recording,
groups the stratified-k-fold CV by subject (so no subject is split across folds), and gives
`n ≫ p` — a stable estimate.

## Method

| | |
|---|---|
| **Erasure** | LEACE subject-axis erasure; report task BA before (raw) / after (identity-free), subject-probe BA, and LEACE subject fraction `s_subj`. Gate 0.55; trap = identity-free − raw > 0.02 among interpretable rows. |
| **Pooled** | one recording per `(subject, class)`, predictions mean-pooled (paper method). |
| **Per-trial** | one recording per window, CV grouped by subject (`n ≫ p`). |
| **FM** | REVE block-6 features, published input contract (0.5–99.5 Hz, 200 Hz). |
| **Control** | Wang TCM: per-mode spatial/temporal SVD bases over all trials (unsupervised), per-trial loading `L = BᵀxC`. Far weaker identity encoding than an FM. |
| **Corpus** | every MOABB MI (LeftRightImagery), ERP (P300, binary Target/NonTarget), SSVEP (binary). 24-subject cap to bound memory; broadband fmax auto-clamped below each cohort's native Nyquist. |

## Result 1 — the trap is a pooling artifact

Datasets trapping / interpretable datasets, by paradigm × family × metric:

| Paradigm | REVE pooled | REVE per-trial | TCM pooled-lift | TCM per-trial |
|---|---|---|---|---|
| **MI** | **17/17** | 0/12 | 7/13 | 0/13 |
| **ERP** | **14/22** | 0/23 | 7/22 | 0/22 |
| **SSVEP** | 3/6 | **1/6** | 1/6 | 0/6 |

REVE traps on **34 datasets** under pooling (every single MI set) and only **1/41** per trial.
The classical TCM control reproduces the same collapse with far less identity.

## Result 2 — canonical reproduction (the paper's own MI trap)

| Dataset | REVE pooled | REVE per-trial | TCM pooled | TCM per-trial |
|---|---|---|---|---|
| **BNCI2014-001** (ds004362 class) | 0.667→**0.963** | 0.537→0.539 (n-t) | 0.537→0.648 | 0.513→0.509 (n-t) |
| Schirrmeister2017 | 0.690→**0.964** | 0.546→0.551 (n-t) | 0.821→1.000 | 0.592 (t-c) |
| Stieger2021 | 0.981→1.000 | 0.730→0.710 (t-c) | 0.989→1.000 | 0.790 (t-c) |

BNCI2014-001 reproduces the FMScope paper's own motor-imagery trap exactly: pooled 0.67→0.96
lift, flat per trial.

## Result 3 — identity dominance without a per-trial trap

REVE's representation is heavily subject-dominated — subject fraction **0.92 / 0.86 / 0.81**
(mean, MI / ERP / SSVEP), range 0.55–0.98 — yet shows no robust per-trial trap. And on clean ERP
contrasts the **plain-SVD control out-decodes the foundation model per trial** while carrying a
fraction of the identity:

| ERP dataset | REVE per-trial raw | TCM per-trial raw | REVE subj-frac | TCM subj-BA |
|---|---|---|---|---|
| Huebner2017 | 0.648 | **0.813** | 0.92 | 0.20 |
| ERPCore-ERN | 0.750 | **0.797** | 0.68 | 0.29 |
| Lee2019-ERP | 0.675 | **0.760** | 0.78 | 0.12 |
| BrainInvaders2012 | 0.628 | 0.690 | 0.93 | 0.10 |
| ERPCore-N170 | 0.628 | 0.686 | 0.90 | 0.58 |
| Huebner2018 | **0.635** | 0.549 (n-t) | 0.89 | 0.12 |
| BrainInvaders2014a | **0.575** | 0.505 (n-t) | 0.95 | 0.17 |

(REVE `subject_frac` is a variance share; TCM `subject-BA` is a decode balanced accuracy —
related but distinct metrics; compare directions, not magnitudes.) REVE wins on the *weaker*
datasets where TCM drops below gate; TCM wins on the clean strong contrasts.

## The one exception is small-N noise

The single per-trial REVE trap across all 41 datasets is **Nakanishi2015** — the smallest cohort
(9 subjects, 270 windows), a 0.877→0.899 lift barely over the 0.02 threshold, and **not**
reproduced by the TCM control on the same data (0.840→0.814, a *decrease*). A borderline
small-sample fluctuation, not a robust counterexample.

## Reproducibility

- Data: `results/moabb_fmscope/{tcm_pertrial,tcm_pertrial_erp,tcm_pertrial_ssvep}.csv` (TCM,
  pooled+per-trial cols) and `leaderboard_{leftright,erp,ssvep}{,_pertrial}.csv` (REVE).
- `bash paper/build.sh` re-tangles every macro/table from those CSVs and weaves `paper/paper.pdf`.
- Audit code: `scripts/moabb_tcm_pertrial.py` (TCM) and `scripts/moabb_identity_leaderboard.py`
  (REVE), both with `--paradigm {leftright,erp,ssvep}`, subject-cap, and Nyquist auto-clamp.

## Failures (9, all external or data-level — not contract bugs)

Dead hosts (EPFLP300, BrainInvaders2013a*), corrupt archives (Sosulski2019, Liu2024), empty
class sets after the binary filter (Kojima2024B, RomaniBF2025ERP, Zhou2016, PhysionetMI-Nyquist,
Wang2016). *BI2013a later recovered for TCM once cached.

## Conclusion

Across the full MOABB benchmark — three paradigms, a foundation model and a classical control,
pooled and per-trial erasure — the identity trap behaves as a **pooling artifact**: pervasive
under the published pooled metric (faithfully reproducing the paper's own MI trap), near-absent
(1/41) per trial, and present even in a near-identity-free control. **Per-trial, subject-grouped
erasure is the test that should be reported for identity-trap claims.**
