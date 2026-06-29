# Deep Learning and Foundation Models for EEG: A 2026 Systematic Review & Benchmark

### A NeuroTechX White Paper — the successor to Roy et al. (2019)

> **Living document.** Maintained by the EMEG-FM research cron (`7d478ae5`) and
> deepened by periodic multi-agent survey sweeps. Status: **v0.1 skeleton**.
> Last deepened: 2026-06-29.
>
> Sections marked **[SWEEP]** are populated by the literature-survey sweep; sections
> marked **[OURS]** are grounded in original NeuroTechX work (benchmarks + audits)
> and are the reason this is a *white paper*, not another survey.

---

## Abstract

[Roy et al. (2019)](https://iopscience.iop.org/article/10.1088/1741-2552/ab260c)
systematically reviewed **154 deep-learning-on-EEG papers (Jan 2010–Jul 2018)** —
the field's canonical reference, led by NeuroTechX co-founder Yannick Roy. It
predates the entire transformer / self-supervised / foundation-model era. This
white paper updates the review for **2019–2026**, adds an **original head-to-head
benchmark** of the current EEG foundation models, and — uniquely — a
**methodological audit of the subject-identity confound** that inflates many
reported EEG-DL results, with prescriptive evaluation guidance. It is positioned to
be the authoritative update because NeuroTechX (a) carries the Roy-review lineage
and (b) owns MOABB, the field's BCI benchmark standard.

---

## 1. Where Roy left off (2019)

- The 154-paper baseline; domains: epilepsy/seizure, sleep, BCI (motor imagery,
  P300, SSVEP), cognitive/affective monitoring.
- Roy's four open questions, still live: **(i) reproducibility** (code/data sharing
  was rare), **(ii) deep-vs-classical** (DL often failed to beat shallow baselines),
  **(iii) data scarcity**, **(iv) no standard benchmark**.
- This white paper answers (ii)–(iv) head-on and confronts (i) directly in §4.

## 2. The paradigm shift, 2019→2026 **[SWEEP]**

- Transformers + SSL pretraining → **EEG foundation models**; state-space models
  (Mamba/S4) emerging; masked-autoencoding the dominant SSL objective.
- **Model-zoo table** (populated by the sweep): LaBraM, BENDR, BIOT, EEGPT,
  CBraMod, REVE, LUNA, Brant, NeuroGPT — arch / pretraining corpus / params / ref.
- Per-domain advances since Roy: epilepsy, sleep, BCI, affective, **+ clinical/
  disease** (e.g. Alzheimer's: LEAD), **+ multimodal** (new since 2019).
- The competing recent surveys ([arXiv 2602.03269](https://arxiv.org/abs/2602.03269),
  [2601.17883](https://arxiv.org/pdf/2601.17883),
  [EEG-FM-Bench 2508.17742](https://arxiv.org/pdf/2508.17742)) are narrow or
  benchmark-only — none combines survey + original benchmark + confound critique.

## 3. Benchmark — head-to-head **[OURS]**

- **WeightWatcher HT-SR α rankings** across REVE / LaBraM / LUNA / BENDR / BIOT:
  REVE/LaBraM/LUNA well-trained (<11% layers α<2); BENDR severely under-trained
  (68% α<2) despite being largest. Per-block breakdown identifies LoRA targets.
- **Downstream** on MOABB + NeuralBench (our runs) — the standardized task suite.
- **FM vs classical (coffeine) vs handcrafted (NEOBA)** on brain-age — directly
  answering Roy's deep-vs-classical question on identical CV splits.

## 4. The reproducibility / confound crisis **[OURS — the original contribution]**

- The **subject-identity confound ("identity trap")**: pooled (subject,condition)
  erasure can make a model look like it decodes the *task* when it is partly
  decoding *who the subject is*. Faithfully reproduces the FMScope trap
  (BNCI2014_001 0.67→0.96; ds004362 0.73→0.99).
- **Pooled vs per-trial erasure** as the stricter, complementary test; the
  deconfounded **"identity-free" leaderboard** across the MOABB datasets.
- ERP CORE (Luck/ERPLAB protocol) and Wang TCM per-trial audits extend it to ERP/
  SSVEP paradigms.
- → **Prescriptive evaluation guidance** (subject-grouped CV, per-trial erasure,
  deconfounding) — the guidance Roy 2019 could not yet give.

## 5. Frontiers **[OURS + SWEEP]**

- **Multimodal:** cross-subject **EEG→fMRI prediction on HBN** (non-simultaneous →
  cross-subject only; `examples/eeg_to_fmri_hbn.py`); EEG→MEG individual source
  imaging on WAND; the Valdés-Sosa generative-fusion lineage.
- **Clinical:** brain-age (NEOBA OSF/ODC, REVE), Alzheimer's (LEAD).
- **Real-time / generative:** EEG-to-image (Alljoined), streaming FM inference.

## 6. Recommendations

- A standard benchmark + **deconfounded evaluation protocol** the field can adopt.
- A **reporting checklist**: confound controls, subject-grouped CV, per-trial
  erasure, leakage diagnostics.
- Open data + reproducible pipelines (the MOABB/NeuralBench model).

---

## References

- Roy, Banville, Albuquerque, Gramfort, Falk, Faubert (2019). *Deep learning-based
  electroencephalography analysis: a systematic review.* J. Neural Eng. 16(5)
  051001. doi:10.1088/1741-2552/ab260c
- [Sweep populates the 2019–2026 corpus + the model-zoo references.]
- NeuroTechX work: MOABB; the FMScope identity-trap audit; NEOBA; REVE/LaBraM
  WeightWatcher comparison; EEG→fMRI HBN; WAND MEG source imaging.

---

### Maintenance log

- **v0.1 (2026-06-29):** skeleton + NeuroTechX positioning. Comprehensive
  literature-survey sweep launched to populate §2 + the model zoo.
