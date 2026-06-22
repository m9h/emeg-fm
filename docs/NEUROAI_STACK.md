# Adopting Meta's NeuroAI stack (neuralfetch / neuralset / neuralbench / exca)

Meta's [neuroai](https://facebookresearch.github.io/neuroai/) stack is four packages.
This is our adoption map: what each gives us, what we already use, and the scaffolding
landed here. **Strategic frame:** the neuroai stack is pure *infrastructure* leverage;
our novel contributions (the FMScope identity-trap audit, the JAX TopK SAE) sit on top.

| package | role | our use |
|---|---|---|
| **neuralset** | data → AI-ready tensors (transforms/extractors) | **already the contract** we mirror (`emeg_fm/alljoined.py::ReveInputNorm` ≡ `reve.yaml` preprocessing, reconciled 2026-06-10) |
| **neuralfetch** | fetch 132 curated studies → BIDS/tensors | **NEW: `scripts/neuralfetch_stage.py`** (see below) |
| **neuralbench** | eval harness (40+ tasks × 14 models, EEG/MEG/fMRI) | anchor for our WW/LoRA/SAE work; **adopt as eval harness** for downstream/brain-age |
| **exca** | uid-cache + Slurm submission + Pydantic config | not adopted — its bare-Slurm submit clashes with our docker-in-sbatch; caching usable standalone |

## neuralfetch — dataset staging (landed)

`scripts/neuralfetch_stage.py` + `scripts/neuralfetch_stage.sbatch`. Light deps
(no torch/jax); installed to a `--target` dir on /mnt/t9 inside the Docker NGC
container (never `pip --user`), output to /mnt/t9 (never /data NFS).

```bash
# what does it cover vs our roadmap?
sbatch scripts/neuralfetch_stage.sbatch --coverage
# stage one study (gated sources still need creds in-container)
sbatch scripts/neuralfetch_stage.sbatch --study Obeid2016Tueg --query '{"target":"age"}' --out /mnt/t9/tueg
```

**Coverage vs our staging roadmap** (live, 132 studies):

| dataset | covered | study | task |
|---|---|---|---|
| TUEG | ✅ | `Obeid2016Tueg` | #43/#72 |
| TUAB / TUAR / TUEV / TUSZ | ✅ | `Lopez2017Tuab`, `Hamid2020Tuar`, `Harati2015Tuev`, `Shah2018Tusz` | #43 |
| HBN | ✅ | `Shirazi2024Hbn` (OpenNeuro ds005516) | brain-age + SAE; retires `download_hbn_s3_direct.py` + eegdash R7/R11 workaround |
| NSD (fMRI) | ✅ | `Allen2022Massive` | MindEye/NSD |
| Sleep-EDF / eegmat / depression / dementia | ✅ | `Kemp2000`, `Zyma2019`, `Mumtaz2018`, `Miltiadous2023` | various |
| MOABB-class EEG | ✅ | `Schalk2004Bci2000Moabb`, Dreyer2023, Hubner, Cho2017, Cattan2019, cVEP | the corpus we sweep |
| **CHBP** | ❌ | — | **#42 stays a hand-rolled Synapse pull** |
| **NSRR sleep** | ❌ | — | **#47 hand-roll (NSRR not a backend)** |
| **Cam-CAN / TDBRAIN / LEMON / OMEGA** | ❌ | — | #44/#57/#41/gated — hand-roll |

Each covered study exposes a `query` field (subject/manifest selection — a production
version of the hand-rolled TUEG manifest→select) + an exca `infra`. **Gated sources
(TUH DUA, etc.) still need credentials in-env** — neuralfetch wraps, it doesn't bypass auth.

**First real adoption target:** TUEG (#72) — `Obeid2016Tueg` with a brain-age `query`,
gated on supplying TUH DUA credentials in the container. Drop-in for the hand-rolled
manifest→select→BIDS pipeline.

## neuralbench — eval harness (recommended next)

`tasks/eeg/psychopathology/config.yaml` **is** the BrainCapture HBN bifactor eval
(canonical R5 held-out split) — it replaces our hand-rolled subject-stratified probe for
the downstream/SAE/brain-age work. CLI: `neuralbench eeg <task> -m reve`. Adopting it
removes hand-rolled-eval risk; our SAE is the novel `downstream_model_wrapper` to upstream.

## What we keep hand-rolling (on purpose)

The **FMScope identity-trap audit** (LEACE erasure, pooled-vs-per-trial) is not a
NeuralBench task — that layer (`scripts/moabb_tcm_pertrial.py`,
`scripts/moabb_identity_leaderboard.py`) stays ours, ideally upstreamed as an
`erasure_probe` wrapper. And CHBP/NSRR/Cam-CAN/TDBRAIN staging stays hand-rolled.

See memory: `reference_neuralfetch_coverage`, `reference_neuralbench`.
