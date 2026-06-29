# Deep Learning and Foundation Models for EEG: A 2026 Systematic Review & Benchmark

### A NeuroTechX White Paper — the successor to Roy et al. (2019)

> **Living document.** Maintained by the EMEG-FM research cron and deepened by
> periodic multi-agent survey sweeps. Status: **v0.4 — survey + first empirical result** (§2/zoo
> from sweep wf_2c2b3461). Last deepened: 2026-06-29.
>
> **[OURS]** marks sections grounded in original NeuroTechX work (benchmarks +
> audits) — the reason this is a *white paper*, not another survey.

---

## Abstract

[Roy et al. (2019)](https://iopscience.iop.org/article/10.1088/1741-2552/ab260c)
systematically reviewed **154 deep-learning-on-EEG papers (Jan 2010–Jul 2018)** —
the field's canonical reference, led by NeuroTechX co-founder Yannick Roy. It
predates the entire transformer / self-supervised / foundation-model era. The seven
years since are **a paradigm shift, not an increment**: architectures flipped from
CNNs to self-supervised transformers (and now state-space models); corpora grew 3–4
orders of magnitude (REVE: 60,000+ h / 25,000 subjects); and — decisively — a
**reproducibility-and-confound reckoning** turned Roy's rigor lens back on the FM era
itself. This white paper updates the review for **2019–2026**, adds an **original
benchmark** (MOABB + WeightWatcher), and a **methodological audit of the subject-
identity confound** (FMScope) with prescriptive evaluation guidance. The honest 2026
summary: foundation models are the headline change, but for practical decoding they
beat lightweight specialists by only ~1% at >2000× the parameters, Riemannian
baselines remain the robust reference, and **evaluation hygiene now matters more than
architecture** — precisely the gap NeuroTechX is positioned to fill.

---

## 1. Where Roy left off (2019)

The 154-paper baseline: a field almost entirely **(a)** supervised and task-specific
(from-scratch CNNs ~40%, RNN/AE/RBM hybrids); **(b)** trained on single labeled
datasets of tens-to-hundreds of subjects; **(c)** hobbled by small data, ~6% code/
data sharing, weak baselines, high preprocessing variance. Roy could ask whether DL
beat shallow baselines; he *could not* ask whether large-scale pretraining helps —
no self-supervised "foundation" EEG model existed. His four open questions
(reproducibility, deep-vs-classical, data scarcity, no benchmark) are exactly what
2019–2026 thrashed on; this paper answers (ii)–(iv) and confronts (i) head-on (§4).

## 2. The paradigm shift, 2019→2026

**Three intertwined shifts.** **(1) Architecture flipped twice:** CNN → Transformer
(BENDR 2021 ported wav2vec-2.0 contrastive SSL to TUEG; the post-LaBraM ICLR-2024
wave made masked/VQ-token transformers the default) → an emerging **state-space**
(Mamba/S4) branch (EEGMamba, FEMBA) chasing linear-in-length scaling. **(2) Self-
supervision + scale** replaced bespoke supervised training: masked reconstruction,
BEiT-style VQ-token prediction (LaBraM, NeuroLM), and latent-prediction/JEPA
(EEG2Rep). The defining EEG-specific design problem became **channel/montage
heterogeneity**, with a clear lineage: channel-as-token (BIOT) → conditional PE
(CBraMod) → 4D coordinate PE (REVE) → learned-query latent compression (LUNA).
**(3) A reproducibility-and-confound reckoning** (§4): segment-wise CV leakage and
the identity trap show many post-2019 "wins" are partly subject re-identification.

### EEG foundation-model zoo (scalp + intracranial)

| Model | Year | Architecture / SSL | Pretraining corpus | Params | Ref |
|---|---|---|---|---|---|
| BENDR | 2021 | wav2vec-2.0 conv+Transformer; contrastive masked-span | TUEG ~1.5 TB | ~10s M | arXiv:2101.12037 |
| BIOT | 2023 | channel-as-token + linear Transformer | cross-dataset | ~3.3M | arXiv:2305.10351 |
| Brant | 2023 | dual time+freq Transformer (SEEG) | 1.01 TB SEEG | ~500M | NeurIPS 2023 |
| Neuro-GPT | 2023 | conformer enc + GPT dec; autoregressive | TUEG ~15k subj | — | arXiv:2311.03764 |
| LaBraM | 2024 | channel-patch Transformer + VQ tokenizer (BEiT) | ~2,500 h / ~20 ds | 5.8–369M | arXiv:2405.18765 |
| EEGPT | 2024 | hierarchical spatial/temporal; dual MAE + align | multi-dataset | 10M | NeurIPS 2024 |
| EEG2Rep | 2024 | JEPA latent-prediction | latent SSL | compact | arXiv:2402.17772 |
| EEGMamba | 2024 | bidirectional Mamba SSM + MoE | 8-task | — | arXiv:2407.20254 |
| Brant-2 | 2024 | time+freq Transformer (SEEG+EEG) | ~4 TB / >15k subj | ~1B | arXiv:2402.10251 |
| NeuroLM | 2025 | text-aligned VQ + GPT-2 LLM; instruction-tuned | ~25,000 h | ≤1.7B | arXiv:2409.00101 |
| CBraMod | 2025 | criss-cross Transformer + conditional PE | TUEG ~9k h | ~4M | arXiv:2412.07236 |
| **REVE** | 2025 | linear patch + Transformer + **4D Fourier PE**; MAE | **60,000+ h / 25k subj / 92 ds** | multi | arXiv:2510.21585 |
| LUNA | 2025 | learned-query latent compression; montage-agnostic | TUEG+Siena ~21k h | — | arXiv:2510.22257 |
| U-Sleep | 2021 | fully-conv U-Net; montage-agnostic hypnograms | 15,660 subj | — | npj Digit Med 4:72 |
| SleepFM | 2024/26 | multimodal leave-one-out contrastive (PSG) | ~585,000 h | — | Nat Med 2026 |
| LEAD | 2025 | Alzheimer's-dedicated multi-scale FM | 2,255 subj / 16 ds | — | arXiv:2502.01678 |

*(Cross-modal decoders — NICE, ATM, DeWave, EEG2TEXT, Brain2Qwerty, EEG2Video,
MindEye2 — are in §5. Next-gen variants: EEGFormer / FoME / CSBrain / Uni-NTFM.)*

### Per-domain advances since Roy

- **BCI:** EEGNet/ShallowConvNet became universal baselines; **MOABB** (Chevallier
  2024, 30 pipelines × 36 datasets) settled the Riemannian-vs-DL verdict (tangent-
  space the robust default); **Euclidean Alignment** the cross-subject workhorse.
- **Sleep** (cleanest lens): supervised seq-models saturated near the inter-rater
  ceiling (~0.80 κ); U-Sleep generalized across 15,660 subjects; SleepFM reframes
  staging as systemic-disease prognosis; Dreem headband got FDA 510(k).
- **Clinical:** weak-supervision seizure detection (AUROC 0.93–0.94); wearable/
  sub-scalp forecasting on circadian/multidien cycles; first dementia FM (LEAD);
  EEG brain-age matured to a candidate biomarker (Engemann 2022).

## 3. Benchmark — head-to-head **[OURS]**

- **WeightWatcher HT-SR α rankings** (data-free, orthogonal to contested leaderboard
  accuracy): REVE/LaBraM/LUNA well-trained (<11% layers α<2); BENDR severely
  under-trained (68% α<2) despite being largest. Per-block α<2 predicts exactly the
  attention matrices where LoRA fine-tuning pays off.
- **Downstream** on MOABB + NeuralBench (our runs) — the standardized, leakage-
  controlled task suite.
- **FM vs classical (coffeine) vs handcrafted (NEOBA)** on brain-age, identical CV —
  directly answering Roy's deep-vs-classical question.

## 4. The reproducibility / confound crisis **[OURS — the original contribution]**

The field scaled, then turned a rigor lens on itself — and the lens is ours.
- **Segment-wise CV leakage** (Brookshire et al. 2024): random-segment CV inflates
  accuracy to **99.8%** that **collapses to ~53% (chance)** under subject-disjoint
  splits — many post-2019 "wins" are subject re-identification.
- **The identity trap** (FMScope, [arXiv:2606.06647](https://arxiv.org/abs/2606.06647)):
  even subject-disjoint CV does not close the gap — frozen FM representations carry
  **13–89× null subject-variance in 12/12 model-dataset pairs** (LaBraM/CBraMod/REVE
  × 4 datasets), **worsening under fine-tuning**, partly a removable linear axis
  carried by the **aperiodic 1/f** carrier.
- **Our recipe:** pooled (subject,condition) erasure reproduces the paper-method
  result byte-exact (BNCI2014_001 0.67→0.96); a stricter **per-trial** test is
  reported alongside; the deconfounded **"identity-free" leaderboard** certifies any
  MOABB pipeline or EEG-FM. This is the prescriptive guidance Roy 2019 could not give.

## 5. Frontiers **[OURS + survey]**

- **Cross-modal generative decoding:** CLIP-aligned EEG→image (NICE arXiv:2308.13234,
  ATM); brain-to-text (DeWave, EEG2TEXT, Meta's **Brain2Qwerty** — MEG CER 32% vs EEG
  67%); EEG→video (EEG2Video); MindEye2 sets the fMRI-side bar. **Caveat the survey
  flags:** teacher-forcing inflates EEG-to-text >3×; noise baselines often rival real
  EEG — the same evaluation-hygiene problem as §4.
- **Multimodal (OURS):** cross-subject **EEG→fMRI on HBN** — sweep wf_e512e1b2
  verified ~1,224 subjects, non-simultaneous → **11 ranked cross-subject cases**
  (`examples/eeg_to_fmri_hbn.py`); the Calhas-2025 "two strategies" fork (bespoke NN
  vs frozen-FM adapter) on HBN's FreeSurfer/C-PAC derivatives — the very cohort
  behind the NeurIPS-2025 EEG Foundation Challenge.

  > **First empirical result (case #2, n=804).** Frozen **REVE resting-EEG embedding
  > → CC200 resting FC**, cross-subject. The trustworthy, rank-robust measure is
  > unambiguous: the **out-of-sample ridge ΔR² of EEG over age+sex+meanFD+site is ≈0**
  > (−0.025 at an eyeballed 50-component cut, −0.018 at the principled **Gavish–Donoho
  > rank** of 137/202). The in-sample CCA canonical correlation, by contrast, is
  > **dimensionality-fragile**: at the ad-hoc k=50 the raw r₁=0.69 looked nominally
  > significant (perm-p=0.04) but did not survive deconfounding (0.51, p=0.30); under
  > the principled GD rank the canonical r *saturates* (~0.89 — CCA overfitting 137+202
  > dims against n=804) and neither raw nor deconfounded is significant (p≈0.19/0.16).
  > **Verdict: no cross-subject EEG→FC signal** — and, fittingly, our *own* pipeline
  > demonstrates the analytical-flexibility lesson (§7) in miniature: a "significant"
  > CCA p that evaporates under a principled dimensionality rule, with only the held-
  > out ΔR² telling the stable truth. FC reliability 0.555.
  > (`scripts/run_eeg_to_fmri_hbn_case2.py`; Gavish–Donoho rank via the `smni-cmi`
  > stack; caveat: subject- not family-blocked.)
- **Source space (OURS):** **WAND MEG** individual source imaging via the
  **Valdés-Sosa / CiftiStorm / VARETA** lineage being ported into neurojax's
  differentiable EMEG-Recon — the bridge to unify interpretable model-driven
  biophysical fusion with scalable data-driven DL (still *unmerged* field-wide).
- **Real-time:** Alljoined-1.6M REVE→CLIP (above chance on sub-01) as the streaming
  FM-inference testbed.

## 6. Open problems (field-wide)

1. **FMs don't reliably beat shallow/Riemannian baselines** once evaluated fairly
   (~0.9–1.2% gain at >2000× params; frozen linear probes near chance).
2. **No EEG scaling law** — more hours/params don't consistently help; mechanism
   unexplained.
3. **The subject-identity confound is the deepest open problem** (see §4).
4. **Evaluation fragmentation + leakage** — inconsistent splits/preprocessing/metrics.
5. **Montage/topology heterogeneity** improving but not robustly solved.
6. **Cross-site/device/population shift + clinical external validation** weak.
7. **Which SSL objective transfers** (MAE vs VQ-token vs JEPA) unsettled; Transformer-
   vs-state-space unresolved.
8. **Confounds beyond identity** (age, sex, site, 1/f) not standardized away.
9. **Interpretability / data-free quality auditing** (HT-SR, SAE) early but promising.
10. **Multimodal/source grounding nascent** — no joint EEG-MEG-fMRI FM; model-driven
    and data-driven paradigms unmerged; real-time streaming rarely demonstrated.

## 7. Recommendations — a deconfounded evaluation protocol

- **Subject-disjoint (and nested N-LNSO) CV by default**, reported explicitly; never
  segment-wise.
- **Identity-erasure audit as standard:** pooled + per-trial erasure; an identity-
  free control that must still beat chance.
- **Deconfound** age/sex/site and the aperiodic 1/f carrier; report incremental ΔR²
  over the confound baseline, not raw association.
- **Report** split provenance, code/checkpoints, subject-level variance + significance.
- **Pair accuracy with a data-free weight-quality lens** (WeightWatcher α).

## 8. Why NeuroTechX (and not another survey)

NeuroTechX uniquely ships **both the benchmark and the deconfounding audit**:
- **Owns MOABB** (Chevallier 2024, 3,500+ subjects) — the natural host for a
  deconfounded "identity-free" leaderboard column.
- **FMScope** operationalizes the field's headline open problem (the FM-era
  generalization of Brookshire 2024).
- **WeightWatcher** adds a label-free weight-spectrum quality axis.
- **NEOBA** brain-age is the clinical-translation arm (classical-vs-FM on identical
  CV + multi-site robustness).
- **EEG→fMRI (HBN) + WAND MEG** extend the audit from sensor-space to source-space
  via the Valdés-Sosa lineage.

> Roy 2019 told the field to share code and worry about reproducibility. 2019–2026
> shows that **even with shared code and subject-disjoint splits, subject-identity
> confounds quietly inflate results.** This white paper is the evaluation-and-
> mechanism complement the model-building literature lacks.

---

## References

- **Roy et al. (2019)** *Deep learning-based electroencephalography analysis: a
  systematic review.* J. Neural Eng. 16(5) 051001. doi:10.1088/1741-2552/ab260c
- **MOABB:** Chevallier et al. (2024) arXiv:2404.15319 · **FMScope identity trap:**
  arXiv:2606.06647 · **segment-leakage:** Brookshire et al. (2024)
- **Capability audits:** arXiv:2507.01196, 2502.21086 · **synthesis review:**
  arXiv:2601.17883 · **benchmarks:** EEG-FM-Bench arXiv:2508.17742, Brain4FMs,
  AdaBrain-Bench
- Model refs inline in the §2 zoo + §5. NeuroTechX work: MOABB; FMScope; NEOBA;
  REVE/LaBraM WeightWatcher; EEG→fMRI HBN; WAND MEG source imaging.

---

### Maintenance log

- **v0.4 (2026-06-29):** §5 case #2 hardened with the principled **Gavish–Donoho
  rank** (`smni-cmi gavish_donoho_rank`, 137/202 vs ad-hoc 50). Lesson: the in-sample
  CCA p is dimensionality-fragile (the one "significant" raw p=0.04 at k=50 evaporates
  at GD rank, p≈0.19); only the rank-robust held-out ΔR²≈0 is trustworthy → still a
  clean no-signal verdict, now also an in-house demonstration of the §7 analytical-
  flexibility point.
- **v0.3 (2026-06-29):** §5 first EMPIRICAL EEG→fMRI result (case #2, n=804): raw
  cross-subject REVE→FC CCA r₁=0.69 (p=0.04) collapses to 0.51 (p=0.30, n.s.) after
  deconfounding age+sex+meanFD+site; ridge ΔR²≈0. Honest confound-mediated null.
  (`scripts/run_eeg_to_fmri_hbn_case2.py`; ids recovered + age-validated to 1e-15;
  C-PAC CC200 FC for 807 EEG∩C-PAC subjects.)
- **v0.2 (2026-06-29):** §2 paradigm-shift narrative + 16-model zoo + per-domain
  advances populated from the 7-lens survey sweep (wf_2c2b3461, all lenses
  substantive); §4 sharpened with Brookshire/FMScope numbers; §5 frontiers + §6
  open problems + §7 protocol + §8 differentiators added. EEG→fMRI arm: 11 ranked
  cases (sweep wf_e512e1b2).
- **v0.1 (2026-06-29):** skeleton + NeuroTechX positioning.
