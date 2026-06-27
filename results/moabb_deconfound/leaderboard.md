# MOABB identity-free leaderboard — LeftRightImagery

Two deconfounded views of cross-subject decoding. **Pipeline** columns: raw vs identity-free ROC-AUC for the best real MOABB pipeline, with the subject subspace erased via LEACE in the pipeline's own feature space (`Δ` = identity inflation). **FM** columns (optional): the identity-trap profile of a frozen EEG-FM embedding of the same cohort — task balanced-accuracy before/after subject erasure, the fraction of representation variance that is subject identity, and a coarse verdict.

| Dataset | N | Pipeline | Raw AUC | Id-free AUC | Δ (inflation) |
|---|---:|---|---:|---:|---:|
| BNCI2014-004 | 9 | CSP+LDA | 0.646 | 0.540 | 0.106 |
| Weibo2014 | 10 | TS+LR | 0.674 | 0.667 | 0.007 |
| Dreyer2023 | 87 | TS+LR | 0.698 | 0.693 | 0.005 |
| Shin2017A | 29 | TS+LR | 0.746 | 0.742 | 0.004 |
| Lee2019-MI | 54 | TS+LR | 0.718 | 0.716 | 0.002 |
| Zhou2016 | 4 | TS+LR | 0.829 | 0.828 | 0.001 |
| Dreyer2023A | 60 | TS+LR | 0.692 | 0.690 | 0.001 |
| Cho2017 | 52 | TS+LR | 0.653 | 0.652 | 0.000 |
| Dreyer2023B | 21 | TS+LR | 0.667 | 0.667 | 0.000 |
| BNCI2014-001 | 9 | TS+LR | 0.695 | 0.695 | -0.000 |
| Schirrmeister2017 | 14 | TS+LR | 0.708 | 0.709 | -0.000 |
| Dreyer2023C | 6 | TS+LR | 0.722 | 0.722 | -0.000 |

### Skipped / failed

- `Liu2024` — FAILED: BadZipFile: File is not a zip file
