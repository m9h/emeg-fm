# AGENTS.md — start here

This repo (`emeg-fm`) holds the **NeuroTechX Atlas**: an open, identity-aware EEG
foundation-model benchmark (the easiest-to-run reproduction+extension of NeuroAtlas).

**To learn the whole EEG-FM program, read [`docs/EEG_FM_ATLAS.md`](docs/EEG_FM_ATLAS.md)** —
it maps the vision, the one-extractor architecture, every Atlas section (brain-age / BCI /
CogNeuro / Brain-to-Image / epilepsy), the models, the key results, and how to run things.

Key entry points:
- **How to run:** everything goes through the NGC 26.06 container via `scripts/eegfm_t9.sh`.
- **Why one extractor covers the model zoo:** `docs/braindecode_eegfm_extraction.md`.
- **The confound thesis (our differentiator):** `docs/MOABB_IDENTITY_TRAP_FM_VS_TCM.md` + the
  vendored `fmscope/` package (LEACE, `audit_cell`).
- **Living survey/benchmark write-up:** `docs/neurotechx_dl_eeg_whitepaper.md`.
- **Data staging requests:** `../hippy-feat/docs/truenas_data_handoff.md` (the truenas agent).

Conventions: heavy data on node-local `/mnt/t9` (never `/data` NFS for many-small-file I/O);
GPU work in the 26.06 container; add a model = one registry row (see the `*_brain_age.py` and
`atlas_*_section.py` `EEGFM`/`TSFM`/`BD_SPEC` tables).
