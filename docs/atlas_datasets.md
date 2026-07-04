# NeuroTechX Atlas — dataset manifest

*The consolidated dataset list for the Atlas: what is wired-and-run, the complete
MOABB universe folded in, and the coverage map against the original NeuroAtlas
(arXiv:2605.14698, 42 datasets). Last updated 2026-07-03.*

Three tiers below: **A** = datasets actually run in an Atlas section; **B** = the
complete MOABB roster (the BCI/ERP/SSVEP universe the identity-free leaderboard can
sweep); **C** = NeuroAtlas coverage comparison; **D** = staged-but-not-run / gated.

---

## A. Atlas sections — datasets wired and run

| Section | Dataset | Accession | n | Access | Metric |
|---|---|---|---|---|---|
| Brain-age | TDBRAIN | Brainclinics (Synapse-class) | 1285 | gated-obtained | R² age |
| Brain-age | LEMON | MPI OpenNeuro-mirrored | ~213 | open | R² age |
| Brain-age | HBN (subset) | FCP-INDI S3 | varies | open | R² age |
| BCI | TrianaGuzman2024 (MI) | **ds005342** | 32 | open (NEMAR) | LOSO norm-BA + id-free Δ |
| BCI | Chailloux2020 (P300) | **ds003190** | 19 | open (NEMAR) | LOSO norm-BA + id-free Δ |
| CogNeuro | ERP CORE (N170/MMN/N2pc/P3/N400/ERN/LRP) | ERP CORE (OSF); also MOABB `ErpCore2021_*` | ~40 | open | per-comp LOSO + within-subj + LEACE |
| Brain-to-Image | Alljoined-1.6M | HuggingFace | — | open | EEG→CLIP top-k retrieval |
| Epilepsy | TUSZ v2.0.3 | Temple (ISIP) | train4667/dev1832/eval865 | gated-obtained | SzCORE + NEDC v6 |
| Sleep | — (NSRR) | — | — | — | ⚪ not built |

**Run today: ~6 distinct datasets across 5 sections.** BCI is only the 2-dataset
MOABB∩NEMAR intersection — see Part B for why, and the full MOABB universe available
to fold in.

---

## B. Complete MOABB roster (folded in) — 148 datasets, ~3,400 subjects

MOABB (which NeuroTechX owns) ships **148 real datasets** (excl. the synthetic
`FakeDataset`). Only **2 are mirrored on OpenNeuro/NEMAR** (TrianaGuzman2024 ds005342,
Chailloux2020 ds003190) — hence the Atlas BCI section's "2". The other 146 are
MOABB-native (Zenodo / lab servers / GIN), auto-downloaded by MOABB, **not** on NEMAR.
The **identity-free MOABB leaderboard** (separate `audit_moabb_fmscope` project) is what
sweeps this whole universe (per-paradigm) with the FMScope/LEACE deconfound; the Atlas
BCI section is the NEMAR-intersection slice.

*Format: `Dataset (n_subjects)`. Counts from `moabb.datasets.utils.dataset_list`.*

**Motor imagery** — 53 datasets, 1318 subjects total:

PhysionetMI (109), Dreyer2023 (87), Stieger2021 (62), Dreyer2023A (60), Lee2019_MI (54), Cho2017 (52), Yang2025 (51), Liu2024 (50), BNCI2020_001 (45), HefmiIch2025 (37), TrianaGuzman2024 (32), GuttmannFlury2025_ME (31), GuttmannFlury2025_MI (31), Rozado2015 (30), Zuo2025 (30), Shin2017A (29), Shin2017B (29), Chang2025 (28), Liu2025 (27), Forenzo2023 (25), Jeong2020 (25), Ma2020 (25), Gao2026 (22), Dreyer2023B (21), BNCI2024_001 (20), BNCI2025_001 (20), Zhou2020 (20), Kumar2024 (18), Yi2025 (18), Brandl2020 (16), Ofner2017 (15), BNCI2014_002 (14), Schirrmeister2017 (14), Wairagkar2018 (14), BNCI2022_001 (13), BNCI2015_001 (12), Tavakolan2017 (12), Zhang2017 (12), BNCI2019_001 (10), BNCI2025_002 (10), GrosseWentrup2009 (10), Weibo2014 (10), BNCI2014_001 (9), BNCI2014_004 (9), BNCI2015_004 (9), AlexMI (8), Kaya2018 (7), Dreyer2023C (6), Wu2020 (6), BNCI2003_004 (5), Zhou2016 (4), Beetl2021_A (3), Beetl2021_B (2)

**P300 / ERP** — 68 datasets, 1407 subjects total:

BI2014a (64), Lee2019_ERP (54), BI2015b (44), BI2015a (43), ErpCore2021_ERN (40), ErpCore2021_LRP (40), ErpCore2021_MMN (40), ErpCore2021_N170 (40), ErpCore2021_N2pc (40), ErpCore2021_N400 (40), ErpCore2021_P3 (40), BI2014b (38), Mainsah2025_Q (36), GuttmannFlury2025_P300 (31), Lee2024_TV (30), BI2012 (25), BI2013a (24), Lee2021Mobile_ERP (24), Mainsah2025_S2 (24), RomaniBF2025ERP (22), BNCI2015_009 (21), Cattan2019_VR (21), FakeVirtualRealityDataset (21), Mainsah2025_M (21), Mainsah2025_G (20), Mainsah2025_J (20), Mainsah2025_R (20), Chailloux2020 (19), Mainsah2025_B (19), Mainsah2025_C (19), Mainsah2025_P (19), BNCI2020_002 (18), Mainsah2025_O (18), Mainsah2025_D (17), BNCI2015_007 (16), Mainsah2025_H (16), BNCI2016_002 (15), Kojima2024B (15), Lee2024_DL (15), Lee2024_EL (15), Simoes2020 (15), Zhang2025 (15), Lee2024_BS (14), Zheng2020 (14), BNCI2015_008 (13), Huebner2017 (13), Mainsah2025_A (13), Mainsah2025_I (13), Sosulski2019 (13), BNCI2015_010 (12), Huebner2018 (12), BNCI2015_006 (11), Kojima2024A (11), Mainsah2025_L (11), BNCI2014_009 (10), BNCI2015_003 (10), BNCI2015_012 (10), Kaneshiro2015 (10), Lee2024_AC (10), Mainsah2025_F (10), Mainsah2025_S1 (10), Speier2017 (10), BNCI2014_008 (8), EPFLP300 (8), Mainsah2025_E (8), Mainsah2025_N (8), BNCI2015_013 (6), Mainsah2025_K (5)

**SSVEP** — 16 datasets, 509 subjects total:

Liu2022EldBETA (100), Liu2020BETA (70), Dong2023 (59), Lee2019_SSVEP (54), Kim2025BetaRange (40), Wang2016 (34), GuttmannFlury2025_SSVEP (31), Han2024Fatigue (24), Lee2021Mobile_SSVEP (23), Chen2017SingleFlicker (12), Kalunga2016 (12), MAMEM1 (11), MAMEM2 (11), MAMEM3 (11), Nakanishi2015 (9), Wang2021Combined (8)

**c-VEP** — 8 datasets, 122 subjects total:

Thielen2021 (30), MartinezCagigal2023Checker (16), MartinezCagigal2023Pary (16), CastillosBurstVEP100 (12), CastillosBurstVEP40 (12), CastillosCVEP100 (12), CastillosCVEP40 (12), Thielen2015 (12)

**Resting-state** — 3 datasets, 46 subjects total:

Rodrigues2017 (19), Hinss2021 (15), Cattan2019_PHMD (12)

> Note: MOABB also exposes **ERP CORE** as `ErpCore2021_*` (7 components) — our CogNeuro
> section currently loads ERP CORE via its own OSF loader; it could be driven through the
> MOABB path for consistency. `Lee2019` appears in all of MI/ERP/SSVEP.

---

## C. NeuroAtlas dataset list (mined from the paper) + coverage cross-check

NeuroAtlas (arXiv:2605.14698): **42 datasets, ∼260k hours** = **7 epilepsy + 15 sleep +
17 BCI** (+ brain-age computed on the sleep/PhysioNet cohorts). The abstract omits the
table; the following is mined from the body text, figures, and captions (≈37 of 42
individually named — the rest sit unlabeled in Appendix B / Fig. 2–3 heatmap cells).

**Epilepsy (7)** — "∼1,410 patients, ∼58,374 h, ∼9,019 annotated seizures":
Helsinki, SeizeIT1, SeizeIT2, **TUSZ**, Bonn, **TUAB** (Temple Abnormal), NMT.

**Sleep (15)** — "∼15.8k patients, ∼201k h" (∼7 named): Sleep-EDF (Expanded), ISRUC-Sleep,
DREEM, HMC (Haaglanden Medisch Centrum), MASS (Montreal Archive), DCSM, + PhysioNet-2026;
remaining ∼7 unnamed in body text.

**Brain-age** — computed on the sleep cohorts + a PhysioNet-2026 healthy/CI holdout;
Sun et al. as the task-specific baseline (no separate corpora).

**BCI (17)** — "adding also two cognitive tasks and one emotion recognition task":
MI ×6 (PhysioNet-MI + 5 unnamed), P300/ERP (Brain Invaders **BI2014a**, Hoffmann P300),
SSVEP (Wang 40-class speller + 1), cognitive ×2, emotion (**DREAMER**).
Supervised-pretraining aside: Siena Scalp EEG.
**NOT ERP CORE** — verified 2026-07-03: the strings "ERP CORE"/N170/N2pc/N400/ERN/LRP appear
NOWHERE in the paper; Kappenman et al. 2021 is in the reference list but never used. Their
ERP arm is plain P300 (BI2014a), so our 7-component ERP-CORE CogNeuro section is genuinely
additive. (An earlier extraction wrongly listed "ERP CORE" as theirs — a citation-vs-dataset
confusion, corrected here.)

### Dataset-for-dataset cross-check (theirs → ours)

| NeuroAtlas dataset | Domain | Do we have it? |
|---|---|---|
| **TUSZ** | epilepsy | ✅ **have + done** (dual-scored SzCORE + NEDC v6) |
| TUAB | epilepsy/abnormal | 🟡 TUEG v2.0.2 staged (same Temple family) |
| Bonn / NMT / Helsinki / SeizeIT1 / SeizeIT2 | epilepsy | ❌ not staged |
| Sleep-EDF | sleep | 🟡 covered by neuralfetch loader; section not built |
| ISRUC / DREEM / HMC / MASS / DCSM + NSRR | sleep | ❌/🟡 NSRR staged; these specific sets not wired |
| PhysioNet-MI | MI | ✅ **in MOABB** (`PhysionetMI`, 109) |
| Brain Invaders **BI2014a** | P300 | ✅ **in MOABB** (`BI2014a`, 64) |
| Wang 40-class SSVEP | SSVEP | ✅ **in MOABB** (`Wang2016` 34 / `Kim2025BetaRange` 40) |
| Hoffmann P300 | P300 | ❌ (MOABB has `EPFLP300` — related Hoffmann-lab set) |
| DREAMER (emotion) | affect | ❌ not in MOABB |
| brain-age (their sleep-based) | brain-age | ✅ **have via TDBRAIN + LEMON + HBN** (different corpora, same task) |

**Takeaway:** MOABB already contains **their key BCI datasets** (PhysioNet-MI, BI2014a,
ERP CORE, Wang SSVEP) — so their entire BCI arm is reproducible from our 148-dataset roster
with no new staging. The gaps are their **non-Temple epilepsy** sets (Bonn/NMT/SeizeIT) and
the **sleep** cohorts (needs the NSRR/DREEM section built).

### Where NeuroAtlas's 42 actually live (OpenNeuro vs other open vs gated)

"Downloadable from OpenNeuro" is a **narrow** set — most NeuroAtlas datasets live on
PhysioNet / Zenodo / OSF / Temple / dedicated portals, not OpenNeuro. But nearly all are
**openly obtainable** somewhere, and MOABB auto-fetches its BCI ones from their native repos.

| Dataset | Domain | On OpenNeuro? | Open source (where) |
|---|---|---|---|
| **PhysioNet-MI** | MI | ✅ **ds004362** | also PhysioNet `eegmmidb`; **we already reproduced ds004362 from-raw** |
| **SeizeIT2** | epilepsy | ✅ **ds005873** | OpenNeuro (125 pt, 883 focal sz) — pullable |
| SeizeIT1 | epilepsy | 🟡 likely (SeizeIT2 sibling; unverified ds) | KU Leuven RDR |
| BI2014a | P300 | ✗ | ✅ Zenodo (via MOABB `BI2014a`) |
| Wang 40-class SSVEP | SSVEP | ✗ | ✅ Tsinghua portal (via MOABB `Wang2016`) |
| Hoffmann P300 | P300 | ✗ | ✅ EPFL (MOABB has sibling `EPFLP300`) |
| DREAMER | emotion | ✗ | 🟡 Zenodo (application) |
| Sleep-EDF / HMC | sleep | ✗ | ✅ PhysioNet |
| ISRUC / DREEM / DCSM | sleep | ✗ | ✅ dedicated portals (open) |
| MASS | sleep | ✗ | 🟡 application |
| Bonn / NMT | epilepsy | ✗ | ✅ Univ-Bonn / Zenodo |
| Helsinki | epilepsy | ✗ | ✅ Zenodo (neonatal sz) |
| **TUSZ / TUAB** | epilepsy | ✗ | Temple ISIP (open, no-auth pull) — **TUSZ done; TUAB=TUEG staged** |
| Siena | epilepsy (pretrain) | ✗ | ✅ PhysioNet |
| PhysioNet-2026 | brain-age | ✗ | ✅ PhysioNet |

**Count on OpenNeuro specifically: ~2–3** (PhysioNet-MI `ds004362` ✅ ours, SeizeIT2
`ds005873`, likely SeizeIT1). **Count openly obtainable overall: ~35+** of the 42 (the
rest via PhysioNet/Zenodo/OSF/Temple; only MASS + DREAMER need an application). MOABB
already auto-pulls their whole BCI arm. So the real "download list" to add for a full
head-to-head is small: SeizeIT2 (OpenNeuro), the non-Temple epilepsy sets (Bonn/NMT/
Helsinki/Siena via Zenodo/PhysioNet), and the sleep cohorts (PhysioNet/portals).

### What we add that NeuroAtlas lacks

| Axis | Ours | Theirs |
|---|---|---|
| **Cognitive/ERP** (ERP CORE 7-comp, Luck battery) | ✅ | ✗ — only plain P300 (BI2014a) as BCI; ERP CORE cited-but-unused, no cognitive-component analysis |
| **Brain-to-image** (Alljoined EEG→CLIP) | ✅ | ✗ |
| **Identity-free (LEACE) confound axis** | ✅ every section | ✗ (only an oculomotor confound) |
| **Open-data footprint** | MOABB 148 + NEMAR ∼250 BIDS | 42, several clinically gated |

---

## D. Staged / gated / expansion (not yet run)

- **Brain-age expansion:** CHBP (Synapse), TUEG v2.0.2 (1.6 TB), Cam-CAN MEG (gated app),
  NSRR sleep EEG (~28–30 k scalp-EEG, per-cohort DUA). See the brain-age benchmark repo.
- **NEMAR/OpenNeuro pool:** ~250 open BIDS EEG datasets — only a handful are wired; the
  "entire NEMAR" ambition is not yet realized beyond the 2 MOABB∩NEMAR BCI sets.
- **Full MOABB sweep:** the 148 datasets above are available to the identity-free
  leaderboard but only the MI (LeftRightImagery) slice has been swept end-to-end.

Sources: `moabb.datasets.utils.dataset_list` (148); NeuroAtlas arXiv:2605.14698 (42, abstract);
per-section results in `docs/EEG_FM_ATLAS.md`.
