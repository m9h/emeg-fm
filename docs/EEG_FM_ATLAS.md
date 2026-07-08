# EEG-FM work in `emeg-fm` — a map for the next agent

*Canonical onboarding index. If you're an agent picking up the EEG foundation-model
work, read this first, then the linked docs. Last major build: 2026-07-02.*

## What this is

The **NeuroTechX Atlas** — the *easiest-to-run reproduction + extension of NeuroAtlas*
(Kontras et al. 2026, arXiv:2605.14698), an open EEG-foundation-model benchmark across
multiple clinical/cognitive domains, with a **confound axis NeuroAtlas lacks** (our
subject/patient **identity-free** analysis).

Three design pillars:
1. **Open, self-downloading data** — **NEMAR/OpenNeuro** (BIDS) + **MOABB** (NeuroTechX owns
   it) + a few gated-but-obtained sets (TDBRAIN, TUSZ). No huge staging burden.
2. **One montage-agnostic extractor** covers the whole EEG-FM zoo (the braindecode keystone,
   below) — adding a model is a table row, not a new adapter.
3. **NeuroAtlas's metrics + our identity-free Δ** (LEACE erasure of subject/patient identity)
   — because much EEG "decoding" is subject re-identification.

## Quick start

Everything runs in the **NGC 26.06 container** via `scripts/eegfm_t9.sh` (torch + transformers
+ braindecode 1.6.1 + the EEG-FM zoo; PYTHONPATH wires `/mnt/t9/{moabblibs,eegfm_libs_2606,
tsfmlibs,tokfix}` + emeg-fm + fmscope). Example:
```
scripts/eegfm_t9.sh python scripts/atlas_cogneuro_section.py --model reve
```
GPU is 1× GB10 on the DGX; heavy data lives on node-local `/mnt/t9`, never `/data` NFS for
many-small-file I/O. `leace.py` is copied to `/mnt/t9` for in-container import.

## The braindecode keystone (why one extractor covers the zoo)

`docs/braindecode_eegfm_extraction.md` — braindecode 1.6.1 ships the zoo as first-class
`braindecode.models.*` (BIOT/BENDR/CBraMod/LUNA/LaBraM/EEGPT/REVE + `Interpolated*`; 1.6.1 adds
EEGDINO/STEEGFormer/MVPFormer/InterpolatedEEGPT/TCFormer — EEGDINO is wired in). Three
properties → one ~150-line extractor: `from_pretrained(hf_id)` (uniform load) + a shared
`final_layer` (hook its input = pooled embedding, ANY model) + `Interpolated*` (`chs_info` →
montage-adapt to the pretrained layout). **Gotchas:** the WeightWatcher `_state_dict_wrapper`
loaders are analysis-only (no real forward); `Interpolated*` SVD-fails on *small* BCI montages
(≤~18 ch) → for those use montage-flexible models (TS-FMs, REVE-3D-coords); custom-code models
(ZUNA) keep a hand `HFModelAdapter` in `emeg_fm/eeg_fm.py`.

**Full dataset manifest:** `docs/atlas_datasets.md` — what's wired-and-run per section,
the complete **148-dataset MOABB** roster folded in, and the coverage map vs NeuroAtlas's 42.

## The Atlas sections (script · data · metric · key result)

| Domain | Script | Data (open) | Metric | Status / headline |
|---|---|---|---|---|
| **Brain-age** | `ts_fm_brain_age.py`, `eeg_fm_brain_age.py`, `braindecode_fm_brain_age.py` | TDBRAIN 1285 / LEMON / HBN | R² age (10-fold, seed 42) | ✅ **classical 0.74 beats ALL 9 FMs**; EEG-FMs & TS-FMs interleaved (Mantis .69 > LaBraM .67 > MOMENT .65 > BIOT .64 > LuMamba .63 > Chronos .60 > REVE .59 > ZUNA .57) |
| **BCI** | `atlas_bci_section.py` | MOABB∩NEMAR = 2 (TrianaGuzman2024 MI ds005342, Chailloux2020 P300 ds003190) | LOSO norm-BA + identity-free Δ | ✅ classical MI 59.8%, REVE 54% — FMs weak on MI |
| **CogNeuro** | `atlas_cogneuro_section.py` (+ `erpcore_luck_parity.py` loader) | ERP CORE 7 components (same ~40 subj) | per-comp LOSO + within-subject + LEACE | ✅ **REVE mean 64.1% vs classical 54.4% over all 7 components, identity-Δ≈0** — FMs win on cognition, no trap |
| **CogNeuro-extras** | `atlas_cogneuro_extra.py` + `atlas_m3cv_identity.py` | N170 (ds002718, 18 subj), ERN (ds004883, 162 subj), P3-aud (ds003061, 13 subj), P3-vis (ds006018, 98 subj), M3CV (95 subj, 13-cond decode) | LOSO + within-subject + LEACE | ✅ **REVE beats classical everywhere both ran** (ERN 69.4→57.6, P3-aud 71.4→61.1, P3-vis 68.0→59.2) — independent 4-7× replication of the ERP CORE headline at much larger N. Identity-trap is **not uniform**: N170/ERN/P3-aud show idΔ≈0, but **P3-vis shows a real trap for both models** (classical +8.9, REVE +4.9) — a genuine within-domain counterexample, reported honestly. M3CV's non-circular condition-decode framing (identity as erased nuisance, not the decode target) gives idΔ≈0. `docs/nemar_cogneuro_extras_results.md` |
| **Brain-to-Image** | `atlas_brain2image_section.py` (+ `extract_alljoined_reve.py`) | Alljoined EEG↔image | EEG→CLIP top-k retrieval | ✅ REVE top-1 ~2× chance (smoke) — FMs' natural-vision domain |
| **Epilepsy** | `atlas_epilepsy_section.py` + `epilepsy_scorer.py` + `nedc_crosscheck.py` | TUSZ v2.0.3 (train4667/dev1832/eval865) + **SeizeIT2** (ds005873, wearable, 89/18/18 subj) + **Helsinki neonatal** (Zenodo 2547147, 49/15/15 neonates) | SzCORE sens@1/10 FP/24h + F1 (+ official NEDC v6 OVLP/TAES cross-check) + patient-id-free Δ | ✅ **both FMs (REVE, EEGDINO) COLLAPSE to 0% usable sens when patient-identity erased on TUSZ; only classical survives (40.9→25.0%)**. Classical Δ REPLICATES on all 3 epilepsy sets (TUSZ+0.044, SeizeIT2+0.019, Helsinki+0.067) — robust 4-domain signature incl. sleep. REVE on Helsinki (small N=15 eval) is genuinely INCONCLUSIVE, honestly reported as such, not spun. `docs/epilepsy_tusz_results.md`, `docs/helsinki_neonatal_results.md` |
| **Sleep** | `atlas_sleep_section.py` + `atlas_sleep_hmc.py` + `atlas_sleep_dod.py` | Sleep-EDF Expanded (197 recs) + **HMC** (151 recs, NeuroAtlas-named) + **Dreem DOD-H/DOD-O** (25+56 subj) | Balanced-acc + Cohen's κ (5-class AASM) + patient-id-free Δ | ✅ **HMC = cleanest identity-trap confirmation yet: REVE WINS on raw perf (κ 0.898 > classical 0.859) yet COLLAPSES hardest under identity erasure (Δ+0.490 vs classical Δ+0.085)**. Sleep-EDF: classical κ 0.551→0.508 (Δ+0.043); REVE 0.459→0.202 (Δ+0.257). **Dreem DOD REVERSES the direction**: classical shows the larger κ drop on both DOD-H (Δ+0.233 vs REVE Δ+0.121) and DOD-O (Δ+0.165 vs REVE Δ+0.100) — reported as a genuine unresolved split across the 4-dataset sleep record, not smoothed into a universal claim. `docs/sleep_edf_results.md`, `docs/hmc_sleep_results.md`, `docs/dreem_dod_sleep_results.md` |

## The thesis (what the results say)

- **FMs earn their keep on evoked/cognitive responses** (ERP CORE, N170) but **classical usually
  wins on spectral tasks** (brain-age, MI, Sleep-EDF staging) — though NOT universally: on HMC
  sleep staging REVE genuinely WINS on raw performance (κ 0.898 vs classical 0.859). Naturalistic
  vision (Alljoined) is where FMs should win most.
- **The identity trap is now confirmed across FIVE datasets in THREE domains, not one-off**:
  epilepsy (TUSZ both FMs→0% sens; SeizeIT2 classical Δ+0.019; Helsinki classical Δ+0.067,
  REVE inconclusive small-N) and sleep (Sleep-EDF REVE Δ+0.257 severe; **HMC REVE Δ+0.490 —
  the CLEANEST confirmation: REVE WINS on raw perf yet collapses hardest under identity
  erasure**, proving the trap isn't just "the FM was already losing"). Classical Δ is a TIGHT,
  consistent cluster across all 5 (+0.019 to +0.085) — the repeated empirical signature, not a
  single result. See `docs/hmc_sleep_results.md` for the strongest single case.
- **The identity trap is a spectrum**: clean on ERP CORE (Δ≈0, same-subject control) → grows on
  Alljoined (naturalistic) / HBN-task (developmental) → **DRAMATIC on TUSZ epilepsy (CONFIRMED
  2026-07-03): both FMs (REVE, EEGDINO) collapse to 0% usable sensitivity when patient identity is
  LEACE-erased, while classical survives (40.9→25.0%)** — the FMs' seizure "detection" rides on
  patient-specific signatures, not seizure physiology. Making the trap *appear vs not* across
  domains is the differentiated result NeuroAtlas can't show (they only do an oculomotor confound).
  Foundations: `docs/epilepsy_tusz_results.md`, `docs/MOABB_IDENTITY_TRAP_FM_VS_TCM.md`, FMScope
  (arXiv:2606.06647), the vendored `fmscope/` package (`audit_cell`, LEACE).

## Models

- **EEG-FMs (braindecode-native):** REVE, LaBraM, BIOT, BENDR, CBraMod, LUNA, EEGPT + LuMamba
  (dedicated `lumamba_brain_age.py`), ZUNA (custom-code adapter). Container/license inventory:
  the WeightWatcher analysis (`docs/WEIGHTWATCHER_EEGFM_ANALYSIS.md`, `analyze_eegfm_weightwatcher.py`).
- **Generic TS-FMs:** MOMENT, Mantis, Chronos-Bolt (`ts_fm_brain_age.py` adapters, in `/mnt/t9/tsfmlibs`).
  Next: Chronos-2 (native multivariate), Time-MoE, TimesFM-2.5 (survey: arXiv:2510.27522).
- **Classical anchors:** coffeine filterbank-Riemann + NEOBA (brain-age, in `meeg-brain-age-benchmark-paper`);
  log-band-power (BCI/CogNeuro); CatBoost/HistGBM on expert features (epilepsy, `epilepsy_scorer.train_gbm`).

## Related & external anchors

- **White paper** (living survey + benchmark + confound critique): `docs/neurotechx_dl_eeg_whitepaper.md`.
- **Brain-age benchmark** repo: `~/dev/meeg-brain-age-benchmark-paper` (coffeine/NEOBA/Deep4Net,
  TDBRAIN wired in as `config_tdbrain_eeg.py`).
- **FMScope reproduction**: vendored `fmscope/`.
- **External:** NeuroAtlas (arXiv:2605.14698); the Sci-Rep 2026 transparent-TUSZ seizure benchmark
  (s41598-026-41358-w) — the epilepsy protocol anchor; generic-TS-FMs-on-EEG (arXiv:2510.27522).
- **Data staging** requests go to the truenas agent via `hippy-feat/docs/truenas_data_handoff.md`.

## TODO (as of 2026-07-02)

Finish BCI zoo+TS-FMs+P300; Alljoined full test-set + cross-subject/identity-free; HBN-task
CogNeuro (developmental identity stress-test); run TUSZ epilepsy when download completes; package
a one-command `neurotechx-atlas` runner + README; add Pearson-r (NeuroAtlas-comparable) + brain-age-gap.
