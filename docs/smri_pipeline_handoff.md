# Handoff: structural-MRI (sMRI) processing — emeg-fm volume-conduction

Repo: `/home/mhough/dev/emeg-fm`  ·  branch: `volume-conduction`
State verified: 2026-06-27 (a few figures below correct stale notes).

## Why it exists (downstream consumers)
The sMRI pipeline feeds three analyses in the EEG volume-conduction study:
1. **Morphometry features** — per-region DKT+aseg volumes → the "anatomy" block `A` in the tier-1
   structure↔function variance partition (replacing the crude block-pooled qsiprep GM-probseg
   features).
2. **Surface-source EEG leadfields** — FreeSurfer-style central-GM cortical surfaces → `charm
   --fs-dir` → the standard EEG source space (vs the cheap GM-*volume* leadfield fallback).
3. **The Thompson "zeta in data distributions" analysis** — the morphometry matrix is a clean,
   full-rank *structural* data distribution whose empirical spectrum can be fit for heavy-tailed
   (zeta / power-law) structure. See the dedicated section at the bottom — this is the thread being
   driven by a separate agent.

## The key design: a two-track split forced by architecture
FastSurfer has two halves with different hardware needs, so the pipeline is split across two boxes
that share the NAS at `/data` (no data movement between them):

| Track | Box | What runs | Output |
|---|---|---|---|
| **Seg (morphometry)** | DGX Spark (aarch64, GB10 GPU) | FastSurfer `--seg_only` — the CNN seg is **aarch64/GPU-native**, ~2 min/subj, ~4–8 GB RAM | `aseg+DKT.VINN.stats` → per-region volumes |
| **Surfaces (leadfield)** | Legion (x86, fedora, 31 GB) | FastSurfer **full** pipeline — recon-surf needs **FreeSurfer 7.4.1 = x86-only** | `lh/rh.pial`, `.white` cortical surfaces |

Hard rule: **don't try surfaces on the DGX** (no x86 FreeSurfer); **don't move the seg track to x86**
(it's the cheap GPU-native half).

## Current state (verified 2026-06-27)
- **Cohort**: HBN BIDS at `/data/raw/hbn-bids`, **2617 subjects**.
- **Seg cohort**: **619 fully complete** (`aseg+DKT.VINN.stats` present), ~1239 started/partial.
  **Idempotent & resumable** — the runner skips any subject whose `.VINN.stats` exists, so just
  re-run to continue.
- **`/data/derivatives/volume_conduction/morphometry.npz`**: still only the **5-subject prototype**
  — `X=(5, 100)` (100 DKT+aseg regions), plus `regions`, `ids`, `ages`. ⚠️ **Needs rebuilding** from
  the 619 completed segs via `build_morphometry.py`.
- **Surface track**: ~4–5 prototype subjects (`fastsurfer/`, 4 with `lh.white`); `surface_forward/`
  holds 4 surface leadfields. Validated end-to-end.
- **GM-volume leadfields** (aarch64 CHARM fallback): 5 subjects in `forward/`.
- **Surface vs volume**: per-electrode RMS gain correlates **r = 0.90** → the cheap GM-volume
  leadfield is a strong proxy for the surface gold standard.

## Key files (emeg-fm, branch `volume-conduction`)
- `emeg_fm/morphometry.py` — `parse_volume_stats` (.stats → {region: Volume_mm3}, col 3=vol /
  col 4=name), `assemble_morphometry` (→ X, regions, ids; 0-fill missing), `normalize_by_total`
  (composition fractions; **omit** to keep absolute volumes, which carry the head-size/age signal
  relevant to conduction). Pure parse+numpy, unit-tested.
- `scripts/run_fastsurfer_seg.sh` — **DGX seg batch**. `docker run --gpus all … fastsurfer:grace
  --seg_only --threads 8`, reads `/data/raw/hbn-bids`, writes `…/fastsurfer_seg`. Idempotent on
  `aseg+DKT.VINN.stats`. Supports `SUBJ_FILE=`, explicit subject args, or `LIMIT=N`.
- `scripts/fastsurfer_run_one_podman.sh` — **Legion surface runner** (one subject). Rootless
  `podman … deepmi/fastsurfer:latest --vox_size 1 --3T --threads 4`. Writes to local
  `/home/mhough/fs_work` then `rsync` to `…/fastsurfer`; drops a `<sid>.done` sentinel with the rc.
- `scripts/build_morphometry.py [--compare]` — assembles `morphometry.npz`, aligns to the REVE EEG
  cohort + ages; `--compare` runs the tier-1 variance partition with morphometry as block `A`
  head-to-head vs the block-pooled `structural_emb.npz` on the common subjects.
- `scripts/tier3_surface_leadfield.py` — `charm <id> T1 --fs-dir <fsdir> --forcerun` → mesh →
  leadfield with `interpolation="middle gm"`.
- `scripts/compare_surface_volume.py` — the surface-vs-volume per-electrode gain comparison (r=0.90).
- `docs/volume_conduction_plan.md`, `docs/simnibs_install_notes.md`, `docs/donoho_denoise.md`.

## Gotchas the next agent MUST know (hard-won)
- **Legion is the user's interactive 31 GB box** — `--memory=16g --memory-swap=30g` is **MANDATORY**
  and `--parallel` is banned: an unbounded `--parallel` recon-surf OOM-killed the host's terminals
  once. Use `--threads 4`, sequential hemispheres. The cap makes overflow die *inside* the container
  (graceful, rc≠0 sentinel) instead of taking down host processes.
- **Legion podman quirks**: the CDI spec must match the installed NVIDIA driver (`nvidia-ctk cdi
  generate` if mismatched); `--security-opt=label=disable` (NVML perms), `--userns=keep-id --user
  $(id -u):$(id -g)` (the image refuses a nonroot uid otherwise). Only `~/.ssh/id_ed25519` is
  authorized on the Legion (the `dgx-ssh-key` is passphrase-encrypted).
- **`--vox_size 1`** — the 0.8 mm conform produced a 320-vs-321 dimension mismatch.
- **Bad scans**: a few subjects (e.g. the original "subject-1") trigger an `mri_normalize` pathology
  (>16–25 GB on a normal 256³) — skip them.
- **SimNIBS CHARM on aarch64** (the GM-volume track) ships x86 binaries → see
  `docs/simnibs_install_notes.md`: mmg source-build + **two-step charm** (segment tolerating the CAT
  surface crash, then `--mesh`) + `interpolation=None; tissues=[2]` for the GM-volume leadfield. The
  **surface route sidesteps all of this** via `charm --fs-dir` (uses FastSurfer surfaces, bypassing
  the broken x86 CAT step at `charm_main.py:334`).
- **The DGX is a shared 9-user box** — be considerate of CPU/GPU; **do not touch other agents'
  task-EEG data or tmux sessions**. **Commit only these pipeline files; never touch the user's WIP**
  (paper/, moabb, weightwatcher, etc.).

## Immediate next steps
1. **Resume the seg cohort** — re-run `bash scripts/run_fastsurfer_seg.sh` on the DGX (idempotent;
   ~619 → 2617 remaining).
2. **Rebuild `morphometry.npz`** from the ~619 completed segs and run
   `python scripts/build_morphometry.py --compare` for the morphometry-vs-block-pooled tier-1
   head-to-head (the npz is currently only the 5-subject prototype).
3. (Optional) Scale surface leadfields beyond the 4-subject prototype on the Legion — slow, x86-only;
   only needed if the r=0.90 volume proxy proves insufficient.

---

## How the morphometry feeds the Thompson "zeta in data distributions" analysis

The zeta thread asks whether the empirical spectra of our *data* matrices are **heavy-tailed
power-laws** (the HT-SR / "zeta-law" regime — an empirical spectral density with a power-law tail of
exponent α, the data-side analogue of the weight-spectrum self-regularization in WeightWatcher),
rather than a clean spiked covariance (a few signal spikes above a Marchenko–Pastur noise bulk). The
distinction is not cosmetic — it dictates the correct estimator (see `docs/donoho_denoise.md`).

**What we already found (the motivation):**
- The **REVE EEG embeddings are power-law**: top-50 log–log singular-value slope ≈ **−2.2** (α ≈ 2,
  the HT-SR *critical* heavy-tail), cond ≈ 2e8, no clean MP bulk edge. This is squarely the zeta-law
  regime, and it broke the spiked-model (Donoho) whitener in the cross-modal CCA — the permutation
  null is the honest tool there, not rank-reduced denoising.
- The **block-pooled structural features were unusable** for this question: rank-deficient, median
  eigenvalue ≈ 1e-19 (~267 near-constant out-of-brain GM columns), so the MP-median σ² estimate
  degenerates and any spectral/zeta fit is meaningless.

**Where the morphometry comes in:** the DKT+aseg morphometry matrix (subjects × ~100 interpretable
regional volumes) is the **clean, full-rank structural data distribution** the zeta analysis needs —
no out-of-brain near-constant columns, every column a named anatomical volume. Concretely it lets the
Thompson analysis:
1. **Fit the structural-side spectrum** (covariance eigenvalues / singular values of `X`, and of
   `normalize_by_total(X)`) for a power-law exponent and compare it to the EEG α ≈ 2 — i.e. is the
   *structural* data distribution also critically heavy-tailed, or genuinely spiked? This is the
   structural counterpart of the EEG REVE measurement.
2. **Replace the rank-deficient block-pooled block** so the cross-modal EEG↔structure coupling can be
   whitened with the *right* tool: the zeta/Donoho diagnostic on each block (check the log–log slope
   / for a visible MP edge first) decides permutation-null vs spiked-model denoising per block.
3. **Tie the spectrum to interpretable anatomy** — because the morphometry columns are named regions
   (not opaque pooled blocks), a heavy-tailed structural spectrum can be read back to *which*
   regional volumes drive the tail, unlike the embedding case.

Practical note for whoever runs it: build the full feature matrix first
(`scripts/build_morphometry.py` over the 619 completed segs → rebuilt `morphometry.npz`), then run
the zeta/power-law fit on `X` (and the by-total-normalized variant). The shared denoise/spectral
core lives in `emeg_fm/denoise.py` (Gavish–Donoho threshold, MP-median σ², BBP spike map) and is
documented in `docs/donoho_denoise.md`; the **applicability test is whether the spectrum has a real
bulk edge at all** — if it's a continuous power-law (as the EEG side is), report it as zeta-law and
use the permutation null rather than forcing a spiked-model fit. The zeta/Thompson thread is owned by
a separate agent — coordinate before changing `denoise.py` or `cross_modal.py`.

---

## RESULT (2026-06-28): the structural zeta test was run — structure is SPIKED, not zeta

The test above has now been executed at fine parcellation, and the verdict is decisive and stable.

**The data path that made it possible (no fMRIPrep / no surface compute needed).** There is no large
HBN FreeSurfer/fMRIPrep/4S release to download (functional pipeline is C-PAC; no surface release at
scale). But two anonymous FCP-INDI downloads suffice:
- **HBN-POD2 QSIPrep** (`s3://fcp-indi/.../BIDS_curated/derivatives/qsiprep/`, 2136 subjects) ships
  `space-MNI152NLin2009cAsym_label-GM_probseg` — GM **already in atlas space**, no warp.
- **4S atlas** (PennLINC AtlasPack) in MNI152NLin2009cAsym at 156–1056 parcels. ⚠️ AtlasPack is a
  **git-annex/DataLad** dataset on an **OSF** special remote; a plain `git clone` leaves the niftis as
  broken symlinks. Materialize with `pip install datalad-osf` → `git annex enableremote osf-storage`
  → `git annex get tpl-MNI152NLin2009cAsym_atlas-4S*Parcels_res-01_dseg.nii.gz`. (`pip install
  git-annex-remote-osf` does NOT exist — the package is `datalad-osf`.)
- Extractor: **`scripts/build_4s_morphometry.py --res {456,1056}`** — per-parcel GM volume =
  `bincount(GM_probseg within parcel)`; gives X = (2136 subjects × n_parcels), no surfaces.

**The verdict (cross-subject regional-GM-volume covariance, z-scored columns):**

| features | n × p | β=p/n | top-mode var | eigs above MP edge | top-50 SV slope |
|---|---|---|---|---|---|
| aseg+DKT volumes | 857 × 95 | 0.11 | 60–76% | 4–5 / 95 (5%) | −0.7 |
| 4S456 GM | 2136 × 456 | 0.21 | 43–55% | 14–15 / 456 (3%) | −0.6/−0.8 |
| 4S1056 GM | 2136 × 1056 | 0.49 | 37–55% | 19–25 / 1056 (2%) | −0.6/−0.8 |
| *EEG REVE (contrast)* | | | — | ~32% | **−2.2** |

Across p = 95 → 456 → 1056 the genuine structured modes stay at **~15–25** while every added parcel is
MP noise — the textbook **spiked-covariance** signature (fixed-rank signal + growing noise bulk), the
*opposite* of a power law. **Structure is low-rank-signal + MP-noise; the zeta/heavy-tail law is a
property of the functional (EEG) embeddings, not the structural data.** Finer parcellation did not
reveal a hidden tail, so the earlier p=95 "spiked" call was not a resolution artifact.

**Methodological catch for the Thompson thread:** the *full-spectrum* log-GM eigenvalue slope is
≈ −2.1 to −2.3 — superficially right at α≈2 — but that is the spike→bulk geometry, **not** a scale-free
cascade. The decisive diagnostic is the **MP-bulk fraction** (here 97–98% of eigenvalues are noise),
not a naive full-slope fit (which would falsely declare zeta). Trust the MP-edge / bulk-fraction test.

**Scope:** this is regional-GM-volume *cross-subject covariance* — the structural analog of the EEG
embedding spectrum. It does NOT speak to per-subject **connectome** ESDs (a different object needing
connectivity, not morphometry); that remains open. Probe script: the spectral fit lives at
`scratchpad/zeta_4s.py` (reusable; loads `morphometry_4s{456,1056}.npz`).
