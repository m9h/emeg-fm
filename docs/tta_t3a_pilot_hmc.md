# T3A test-time adaptation pilot on HMC — results

*New research axis opened after reviewing NeuroAdapt-Bench
(github.com/leegabriel/NeuroAdapt-Bench, arXiv 2604.16926), which studies
test-time adaptation (TTA) for EEG-FMs under distribution shift and finds T3A
(optimization-free, prototype-based) is their most reliable method. Original
question for our identity-trap program: T3A's online support-set literally
incorporates the held-out test subject's own features — does any T3A gain
shrink under LEACE identity-erasure (evidence it was partly identity
exploitation, not genuine adaptation)? Completed 2026-07-10 as a first pass
on this axis, using HMC (our cleanest existing identity-trap dataset: REVE
wins raw perf, κ 0.898, but collapses hardest under identity erasure,
Δ+0.490 — `docs/hmc_sleep_results.md`).*

## Setup

- Ported T3A from `NeuroAdapt-Bench/methods/t3a/t3a.py` to pure numpy
  (`scripts/tta_t3a.py`) — fully optimization-free, applicable to our REVE
  LogisticRegression probe (classical's GBM has no linear weight vectors to
  seed prototypes from, so this pilot is REVE-only).
- Applied T3A sequentially, per eval recording, in the recording's own
  temporal (chronological) epoch order — the natural way to run online TTA
  on a continuous clinical monitoring stream.
- Compared: baseline (frozen probe, no TTA) vs T3A-adapted, under both the
  normal and identity-free (LEACE-erased) probe fits.

## Results

| Config | baseline bacc/κ | T3A-adapted bacc/κ | T3A gain (Δκ) |
|---|---|---|---|
| normal | 94.9% / 0.898 | 63.1% / 0.262 | **−0.636** |
| identity-free | 70.4% / 0.408 | 56.5% / 0.129 | **−0.279** |

## Finding — honest, not spun

**T3A does not provide a net benefit here at all — it substantially HURTS
performance in both configs.** This makes the original identity-leakage
question moot for this pilot (there's no net gain to attribute to identity
exploitation vs genuine adaptation).

**Why, plausibly**: sleep staging is maximally temporally autocorrelated —
stages persist for long contiguous runs (minutes of uninterrupted N2, etc.),
unlike the i.i.d.-shuffled test batches TTA methods are typically designed
and benchmarked against (including, most likely, NeuroAdapt-Bench's own
evaluation protocol). T3A's online support-set accumulates pseudo-labeled
features causally, one epoch at a time, with no ground truth. Under a
strongly autocorrelated stream, a confident-but-wrong early prediction can
seed the support set with mislabeled prototypes that then reinforce
themselves for a long subsequent run of the same true stage — a known
failure mode for online pseudo-label-based TTA under correlated (non-i.i.d.)
streams, here landing on the worst possible signal type (clinical
polysomnography) for that pathology.

**Practical implication**: naive per-epoch sequential T3A is unsafe for
continuous, strongly-autocorrelated clinical monitoring signals (sleep
staging, and plausibly continuous seizure monitoring) without some
safeguard — e.g. periodic support-set reset, batching non-sequential epoch
subsets, or a confidence floor before admitting a support. None of these are
implemented or tested here; this is reported as a real, reproducible
methodological caveat surfaced by a single honest pilot, not as a general
indictment of T3A (which NeuroAdapt-Bench reports works well in its own,
presumably non-sequential, evaluation setting) or a completed investigation
of the identity-leakage question.

## What would resolve the identity-leakage question (not done here)

To actually test the original hypothesis, T3A would need to provide a real
net gain first. Two candidate directions if this axis is pursued further:
- Try T3A on a **non-autocorrelated** decode task from our program (e.g. a
  single-trial BCI/ERP dataset where epochs are naturally i.i.d.-shuffled
  trials, not a contiguous stream) — closer to T3A's intended regime.
- Add a safeguard against the autocorrelation pathology (confidence
  threshold before admitting a support, or periodic reset) and re-test on
  HMC to see if a genuine gain emerges that can THEN be compared
  normal-vs-identity-free.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/tta_pilot_hmc.py
```
