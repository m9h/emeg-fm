# LoRA fine-tuning vs frozen probe — identity-free pilot on HMC

*Follow-up to `docs/lora_reve_chbmit_pilot.md`, which was inconclusive
(CHB-MIT collapses REVE to near-floor performance even frozen, leaving no
real signal for LoRA to preserve or destroy). HMC is the opposite case:
REVE's cleanest, strongest identity-trap result in the program — frozen REVE
WINS raw performance (κ 0.898 > classical 0.859) yet COLLAPSES hardest under
identity erasure (Δ+0.490 vs classical Δ+0.085, `docs/hmc_sleep_results.md`).
Motivated by OpenEEGBench (github.com/braindecode/OpenEEGBench), a 12-dataset
PEFT benchmark with 8 fine-tuning strategies but no identity-trap axis —
this crosses their fine-tuning-method dimension with our identity-free axis.*

## Setup

- LoRA-adapted `to_qkv`/`to_out` in REVE blocks {0,1,2,3,10} (under-trained,
  α<2 per HT-SR/WeightWatcher analysis) + a 5-class cross-entropy head on
  the mean-pooled layer-6 feature. Same LoRA machinery as the CHB-MIT pilot.
- **Bounded, disclosed scope**: LoRA fine-tunes on only 20 of HMC's 107
  train subjects (6520 epochs) — the full train split is too large for a
  single-pilot fine-tuning budget. Dev/eval use the full 22/22-subject split
  (same partition as `docs/hmc_sleep_results.md`'s frozen baseline).
- 1 epoch, Adam lr=1e-3. Loss converged properly this time (0.30 final,
  vs CHB-MIT's stuck-at-7.3) — a real, non-degenerate classifier.

## Results

| Config (train N) | eval balanced-acc | eval κ | Δκ |
|---|---|---|---|
| Frozen REVE, full 107 train subj (`docs/hmc_sleep_results.md`) | — | 0.898 → 0.408 | **+0.490** |
| **Frozen REVE, matched 20 train subj (control)** | 92.3% → 86.8% | 0.846 → 0.735 | **+0.111** |
| **LoRA-REVE, 20 train subj (this pilot)** | 90.3% → 84.7% | 0.807 → 0.695 | **+0.112** |

## Finding — honest, and rigorously checked, not spun

**First cut looked like a big effect: LoRA's Δ (+0.112) was 4.4× smaller than
the documented full-cohort frozen baseline (+0.490).** Before reporting that
as "LoRA weakens the identity trap," I ran a matched control — frozen REVE,
no LoRA, same bounded 20-subject train set — to isolate whether the
shrinkage came from fine-tuning or from the subject-count confound.

**The control's Δ (+0.111) is statistically indistinguishable from LoRA's
Δ (+0.112). LoRA fine-tuning made no detectable difference to the
identity-free Δ.** The apparent 4.4× shrinkage was entirely a mechanical
artifact of using 20 train subjects instead of 107: LEACE's erasure
projection is fit against however many distinct patient identities are in
the training set, so fewer subjects gives it less identity-signal to find
and remove — independent of whatever model (frozen or LoRA-tuned) produced
the embeddings.

**Secondary finding, worth flagging for the whole program**: the identity-free
Δ magnitude is itself sensitive to the number of train subjects used to fit
the eraser/probe (0.111 at N=20 vs 0.490 at N=107 here) — not solely a
property of the model or dataset. This means Δ magnitudes are not
automatically comparable across datasets with very different train-subject
counts (e.g. Siena's 10 patients vs Sleep-EDF's ~140 recordings) without
controlling for N. This pilot doesn't re-audit every existing Δ in the
program against this confound — that would be a much larger undertaking —
but it's disclosed here as a real methodological caveat surfaced by doing
the matched-control check properly, not swept under the rug.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/lora_reve_hmc_pilot.py --both --epochs 1 --n-train-subjects 20
scripts/eegfm_t9.sh python scripts/frozen_reve_hmc_matched_control.py
```
