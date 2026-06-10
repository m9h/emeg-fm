# Vendored: FMScope

This directory is a vendored copy of **FMScope** — the diagnostic toolkit from
*"The Identity Trap in EEG Foundation Models"* (arXiv 2606.06647).

- **Upstream:** https://github.com/Jimmy110101013/fmscope
- **License:** MIT (see `LICENSE`)
- **Vendored at:** depth-1 clone, nested `.git`/`.github` removed.

It is included unmodified so emeg-fm can run the five frozen-representation
identity-leakage diagnostics (variance decomposition + random-Gaussian null,
LEACE subject-axis erasure, FOOOF aperiodic ablation, layer-wise probing,
direction consistency) against our own E/MEG foundation-model adapters.

The bridge from emeg-fm's REVE/LUNA adapters and the Alljoined cohort into
FMScope's `FMExtractor` / `CohortAdapter` contracts lives in
[`../emeg_fm/fmscope_bridge.py`](../emeg_fm/fmscope_bridge.py) — not in this
vendored tree, so upstream stays a clean drop-in for future syncs.
