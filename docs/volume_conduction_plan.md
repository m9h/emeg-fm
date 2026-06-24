# HBN EEG volume-conduction / structure–function — plan & data map

*Compiled 2026-06-23. Tests whether EEG's age signal is anatomy-redundant (volume conduction) vs
EEG-unique (neural), and builds toward a forward-model causal test. Companion to the zeta-law /
data-spectrum work in `wwj`.*

## Status (this commit)
**Tier-1 analysis core: DONE + TDD.** `emeg_fm/variance_partition.py` (pure numpy) +
`tests/test_variance_partition.py` (green). The data assembly + tiers 2–3 are cluster jobs, scoped
below with exact commands.

## Data map (verified on disk)
- **EEG — DONE.** REVE-base block-6 embeddings at `/mnt/t9/reve_hbn_emb.npz` — `X (2537, 512)` +
  `ages (2537,)`, RestingState, brain-age R²=0.605. **No subject IDs stored** but order =
  `sorted(participant_id)` over the 2537 subjects with `proc-autoreject_epo.fif` + age
  (deterministically reconstructable, no compute). Pipeline: `scripts/reve_brain_age*.{py,sbatch}`.
- **DWI — MUST-GENERATE (cheap, ready).** `/data/raw/hbn-qsiprep/` (2136 subj): preproc DWI
  `*_space-T1w_desc-preproc_dwi.nii.gz` + `.bval/.bvec` + mrtrix `.b` + mask; shells b=0/1000/2000.
  **No FA/MD shipped.** mrtrix 3.0.4 + FSL 6.0.7.19 installed → generation runs out of the box
  (~1–3 min/subj; ~few CPU-hrs for the cohort):
  ```
  dwi2tensor <dwi>.nii.gz -grad <dwi>.b -mask <mask>.nii.gz tensor.mif
  tensor2metric tensor.mif -fa fa.nii.gz -adc md.nii.gz
  ```
- **VBM — PARTIAL.** No FSL-VBM/CAT12/SPM. FreeSurfer only N=29 (too few; 0 of the 5 prototype subj).
  **Use qsiprep GM/WM/CSF `*_label-{GM,WM,CSF}_probseg.nii.gz` in MNI (2136 subj)** as the VBM proxy
  (voxel GM-density or atlas-ROI GM volumes).
- **Cohort:** EEG∩DWI∩T1 = **1534** (865 also have the manifest's psychopathology factors). Age is
  not the binding constraint (REVE npz carries ages).
- **Forward model — MUST-INSTALL.** SimNIBS/charm are source checkouts only (not callable). Install
  SimNIBS (bundles charm/SAMSEG) before tier 3.

## Tier 1 — variance partition (EEG age ~ anatomy)  [core DONE]
`variance_partition(E, A, y)` → subject-level CV ridge commonality:
`redundant_fraction` (EEG age-signal reproducible from anatomy — *consistent with* conduction) and
`eeg_unique_fraction` (age info anatomy can't reproduce — candidate neural). Rows are subjects ⇒
k-fold = subject-level CV (no pseudoreplication).
**Remaining (data assembly, then run):**
1. Reconstruct EEG IDs: `sorted` participant list over the 2537 autoreject-epo subjects → align to `X`.
2. Generate DWI FA/MD (command above) over EEG∩DWI subjects; reduce to per-subject features
   (atlas-ROI means of FA/MD) → `A_dwi`.
3. GM features from qsiprep GM-probseg (atlas-ROI GM volumes) → `A_vbm`. `A = [A_vbm ⊕ A_dwi]`.
4. `variance_partition(X_eeg, A, ages)` on the ~1534 cohort. Headline: % of EEG brain-age that is
   anatomy-redundant.

## Tier 2 — joint VBM/DWI structural embedding  [BUILT]
`scripts/build_structural_embedding.py` → `subject_structural_features` (block-pooled MNI GM-probseg ⊕
global FA/MD scalars) over the EEG cohort → `structural_emb.npz` (X, ids), reusable by E1/E2/E4.
Depends on the tier-1 DWI generation.

## E4 — cross-modal EEG↔structural  [BUILT, core tested]
`emeg_fm/cross_modal.py` (vendored twin of `wwj`'s e4; `tests/test_cross_modal.py` green) +
`scripts/run_e4.py`: REVE EEG ⊗ structural → canonical-correlation spectrum, **full and
age-residualized** (conduction removed) → shared-subspace dimension beyond conduction. The real
EEG↔sMRI structure–function number, once the structural embedding exists.

## Tier 3 — forward-model prototype (5 subjects)  [DONE — pipeline works end-to-end]
The **causal** conduction test: does the anatomy-derived lead field reproduce the EEG age effect?
Per subject: CHARM head segmentation (raw T1, `hbn-bids/.../acq-HCP_T1w`) → SimNIBS EEG leadfield
(`scripts/tier3_leadfield_prototype.py`). **All 5 prototype subjects have a valid GM-volume leadfield**
(75 elec × ~0.9–1.35M GM tets × 3, V/m): `sub-NDARAA948VFH` (7.98), `sub-NDARAB458VK9` (12.84),
`sub-NDARAC349YUC` (10.05), `sub-NDARAC853DTE` (10.23), `sub-NDARAD224CRB` (8.48).

SimNIBS 4.1 ships **x86_64** binaries; this box is aarch64. Workarounds (see `docs/simnibs_install_notes.md`):
libstdc++ preload for petsc4py; **two-step CHARM** (`charm T1` writes the label image then the x86_64
CAT surface step crashes — tolerated — followed by `charm --mesh`); **mmg 5.8.0 rebuilt from source**
for the mesh optimiser; leadfield with `interpolation=None` + `tissues=[2]` (GM volume — the default
needs the disabled central surfaces / lands on eye-balls). Conductivity is scalar (no aarch64 `dwi2cond`);
cap is the bundled 10-10 (EGI-128 is a follow-up).

**Leadfield doesn't scale raw** (~2 GB/subj ⇒ ~3.7 TB at n≈1534). `emeg_fm/leadfield.py` +
`scripts/build_leadfield_descriptors.py` reduce each to a compact per-subject descriptor (per-electrode
block-pooled GM field, 75·grid³ ≈ 4800-d, + per-electrode RMS gain); `--discard-raw` is the scaling mode.
n=5 age trend is illustrative only (far too few for a real test). To run the actual tier-3 test: scale
the summary-only pipeline over the cohort, then variance-partition the leadfield descriptor vs the EEG
age signal (same machinery as tier-1).

## Honest caveats
- **Correlation ≠ causation.** The tier-1 redundancy split is *consistent with* conduction, not proof:
  age is a common cause, so a neural age effect that merely tracks age also reads as redundant. A large
  `eeg_unique` is positive evidence of EEG-specific signal; conduction is confirmed only by tier 3.
- **Anchor on age.** Psychopathology is a cross-modal wash (EEG competition + the fMRI zeta-law run);
  age is the signal-bearing target here. The framework *quantifies* the wash rather than chasing it.
- **Subject-level CV** throughout (rows = subjects, so k-fold is already subject-level).

## Run order
1. (done) tier-1/tier-2/E4 cores + tests; EEG-ID reconstruction validated (2537==2537).
2. `python scripts/gen_dwi_scalars.py` — DWI FA/MD (cluster, ~few CPU-hrs).
3. `python scripts/run_tier1.py` — tier-1 variance-partition headline (anatomy-redundant vs EEG-unique).
4. `python scripts/build_structural_embedding.py` — tier-2 structural_emb.npz.
5. `python scripts/run_e4.py` — EEG↔structural cross-modal spectrum (full + age-residualized).
6. (done) `python scripts/tier3_leadfield_prototype.py` (5 subjects) — GM-volume leadfields.
7. `python scripts/build_leadfield_descriptors.py [--discard-raw]` — compact per-subject leadfield
   descriptor (storable at cohort scale); prints the n=5 leadfield-vs-age trend (illustrative).
