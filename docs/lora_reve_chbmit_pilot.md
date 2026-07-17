# LoRA fine-tuning vs frozen probe — identity-free pilot on CHB-MIT

*Motivated by OpenEEGBench (github.com/braindecode/OpenEEGBench) — a 12-dataset
EEG-FM parameter-efficient-fine-tuning (PEFT) benchmark with 8 strategies
(LoRA/DoRA/OFT/IA3/full-FT/frozen probes) but no identity-trap axis. Question:
does LoRA fine-tuning REVE trap harder or softer on subject identity than our
existing frozen-embedding + linear-probe pipeline (`docs/chbmit_epilepsy_
results.md`: REVE normal AUC 0.014, identity-free AUC 0.093, Δ=−0.079 — REVE
already near-floor on CHB-MIT's bipolar montage via the anchor-electrode
approximation)?*

## Setup

- LoRA-adapted the `to_qkv`/`to_out` attention projections in REVE blocks
  {0,1,2,3,10} — the exact blocks flagged as under-trained (α<2, HT-SR
  analysis, `reference_reve_weightwatcher_alphas.md`) — rank 8, plus a
  linear classification head on the mean-pooled layer-6 feature (same
  layer/pooling as the frozen `embed_reve` path).
- Bounded, disclosed pilot scope (not the full corpus): all seizure-containing
  recordings + ≤1 clean recording per patient, applied to **train, dev, AND
  eval** (114/22/27 recordings) — a first attempt loaded the full 23-patient
  corpus for post-finetune embedding and stalled >7h with no progress; this
  bounded scope is what actually completed. Eval N here is therefore smaller
  than `docs/chbmit_epilepsy_results.md`'s full-corpus eval — the intended
  comparison is the *shape* of the identity-free Δ under otherwise-identical
  bounded data, not absolute-AUC parity with that doc.
- 1 epoch, 59,984 windows (991 seizure), Adam lr=1e-3, class-weighted BCE.
- After fine-tuning: freeze everything, extract pooled embeddings, run the
  *same* downstream pipeline as frozen REVE (StandardScaler + optional LEACE
  eraser + LogisticRegression + SzCORE `evaluate()`) — isolating the
  fine-tuning variable while keeping the identity-free comparison
  apples-to-apples.

## Results

| Config | eval AUC[.1-10 FP/d] | sens@1/d | sens@10/d | F1 |
|---|---|---|---|---|
| LoRA-REVE normal | 0.000 | 0.0% | 0.0% | 0.165 @ 256.9 FP/day |
| LoRA-REVE identity-free | 0.004 | 0.0% | 2.9% | 0.165 @ 256.9 FP/day |

Δ(eval AUC) = **−0.004**. Training loss after 1 epoch: 7.30 (BCE, still high).

## Finding — honest, not spun

**This pilot does not answer the motivating question.** Both configs collapse
to the scorer's floor (AUC ≈0, F1 driven by a very high false-alarm rate,
256.9 FP/day) — essentially the same near-degenerate outcome REVE already
showed frozen on CHB-MIT (`docs/chbmit_epilepsy_results.md`, AUC 0.014/0.093),
now slightly worse under LoRA. With no real classifier signal in either
config, the −0.004 Δ is noise around a floor, not evidence that LoRA traps
identity more softly than a frozen probe.

**Plausible causes, not diagnosed further here:** (1) 1 epoch on a bounded
~60k-window subset is very likely under-trained (loss was still 7.3, far
from converged); (2) CHB-MIT's bipolar montage + the anchor-electrode
approximation already degrades REVE's frozen performance to near-floor, so
LoRA is fine-tuning on top of an already-poor input representation; (3)
class-weighted BCE with severe imbalance (991/59984 ≈ 1.7% positive) can be
unstable at lr=1e-3 in one epoch. Any of these could be the dominant factor;
distinguishing them was out of scope for a single pilot.

**What would be needed to actually test the hypothesis:** more epochs (or a
learning-rate schedule) until the LoRA-tuned probe reaches a real (non-floor)
operating point comparable to frozen REVE's 0.014, so there is an actual
signal for identity-erasure to act on. Not pursued further in this pilot —
reported as an honest null result, not chased into a longer training run
without direction.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/lora_reve_chbmit_pilot.py --both --epochs 1 --max-nonseizure-per-patient 1
```
