# Handoff: ERP additive-vs-phase-reset (MMN) → the smni-eeg / dynamics agent

**Owner:** the **smni-eeg agent** (dynamics / generative-modelling lane). This is a DCM /
neural-mass adjudication question — its core (the *resolution*) is generative, so it lives
with the dynamics thread, not the EEG-FM/white-paper lane. Origin: user routed it here
2026-07-01 ("might best be a smni-eeg project"). Background + framing:
memory `project_erp_additive_vs_phasereset`; data README `/data/datasets/spm_dcm/README.md`.

## The question
Additive (independent phasic burst on background EEG) vs phase-reset (event realigns ongoing
rhythms) accounts of ERP generation. **Yeung, Bogacz, Holroyd, Cohen 2004** (Psychophysiology
41:822) + **Yeung et al. 2007** (ERN; 44:39): ITC and "no total-power increase" **cannot
distinguish** the two — a purely additive fixed-latency peak in power-matched noise forces
phase clustering → artifactual ITC↑. (Also Bishop & Hardiman 2010 single-trial MMN; Cavanagh
& Frank 2010 frontal-theta↔RL prediction errors.) This is the 2004 ancestor of our
identity-trap / metric-confound thesis.

## Feasibility scouting — DONE (2026-07-01), don't repeat
Everything needed is in hand; the *demonstration* half is ~1–2 h, a single script.
- **MMN data STAGED (node-local):** `/mnt/t9/spm_dcm/eeg_mmn/eeg_mmn/`
  - `subject1.bdf` — 144 ch @ 512 Hz, 915 s. Triggers on the `Status` channel:
    **65152 ×480 (standard), 65216 ×120 (deviant)** — textbook 4:1 oddball. Epoch →
    deviant−standard = the MMN.
  - `maeMdfspm8_subject1.{mat,dat}` — the **preprocessed epoched** SPM file (DCM-ERP loads this).
  - `DCM_subject1.mat` — a **reference DCM-ERP model spec** (seeds the generative fit).
  - zips persist on `/data/datasets/spm_dcm/` (extraction stays on `/mnt/t9`; no NFS small-file unzip).
- **ITC / JTFA in hand:** MNE 1.12.1 (`~/dev/neurojax/.venv-models`) gives ITC in one call —
  `mne.time_frequency.tfr_morlet(..., return_itc=True)` and reads the BDF directly. Our JAX
  stack: `neurojax/src/neurojax/analysis/timefreq.py` + `superlet.py` (Morlet/Superlet);
  ITC = |mean over trials of unit-phase analytic signal| (2-line add if not present),
  cross-check vs MNE as ground truth → also validates the neurojax JTFA tooling.
- **Additive-only sim:** fixed-latency deterministic peak + power-spectrum-matched 1/f noise,
  **zero** phase-reset (~20 lines). This is the Yeung-2004 stimulus.

## Deliverables
1. **(fast, demo)** Reproduce Yeung-2004: additive-only sim → run ITC/TF → exhibit
   artifactual ITC↑ + flat total-power (the false phase-reset signature).
2. **(fast, demo)** Same pipeline on the **real staged MMN** → show the ambiguity on actual data.
3. **(the real work — smni-eeg core) Resolution via the identifiability triad:**
   (a) single-trial amplitude distributions (additive shifts the mean; phase-reset preserves
   amplitude), (b) pre-stimulus power dependence (phase-reset predicts post-event amplitude
   tracks pre-stim oscillation amplitude), (c) **generative DCM / neural-mass fit to single
   trials** (neurojax dynamics + reference `DCM_subject1.mat`; ties to the SPM25 DCM comparison).

## White-paper hook (EEG-FM lane — mine)
If (1)+(2) yield a figure, it feeds `docs/neurotechx_dl_eeg_whitepaper.md` §4 (the confound
crisis) as the *2004 ancestor* of the identity-trap thesis, reproduced on real data. **I'll
integrate that on your signal** — the analysis is yours; the §4 write-up is the handoff back.

## Constraints
Node-local `/mnt/t9` for compute; don't extract small files to `/data` NFS. Coordinate with
the SPM25 DCM comparison already on this dataset.
