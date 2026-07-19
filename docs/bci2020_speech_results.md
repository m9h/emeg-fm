# BCI Competition 2020 Track 3 (imagined speech) — results

*New task domain: imagined speech (5-class), and one of OpenEEGBench's 12.
15 subjects, 64-channel extended 10-20 montage.*

## Setup

- 11 train / 2 dev / 2 eval subjects, subject-disjoint (small cohort —
  only 15 subjects total).
- Pulled via braindecode's own `BaseConcatDataset.pull_from_hub`
  (`braindecode/bcic2020-3`, open HF mirror).

## Results (eval split)

| Model | Config | balanced-acc | κ |
|---|---|---|---|
| classical (log-band-power) | normal | 18.5% | −0.019 |
| classical (log-band-power) | identity-free | 19.5% | −0.006 |
| REVE | normal | 20.4% | 0.005 |
| REVE | identity-free | 19.9% | −0.002 |

Identity-free Δ(eval κ): classical **−0.012**, REVE **+0.006** — both noise.

## Finding — honest, not spun

**Neither model does better than chance** (5-class chance = 20%; every
balanced-acc here is within a point or two of 20%, every κ within noise of
0). This is the hardest task in the program so far — imagined speech from
scalp EEG is a genuinely difficult decode, and with only 15 subjects (2 in
eval) there isn't enough data for either a frozen probe or REVE's embedding
to find real signal. The identity-free Δ is uninterpretable here for the
same reason as Helsinki/CHB-MIT/Siena: with no real classifier signal in
any config, there's nothing for identity erasure to meaningfully act on.
Reported as a clean null result, not chased into hyperparameter tuning.

## Reproduce

```bash
scripts/eegfm_t9.sh python scripts/atlas_speech_bci2020.py --model logbandpower --both
scripts/eegfm_t9.sh python scripts/atlas_speech_bci2020.py --model reve --both
```
