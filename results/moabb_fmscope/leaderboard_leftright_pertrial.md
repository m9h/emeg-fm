# MOABB identity-free leaderboard — LeftRightImagery

Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647) on the LeftRightImagery task contrast. **Raw BA** / **Identity-free BA** are the cross-subject task balanced-accuracy from the **per-trial** erasure decode (each trial its own recording, StratifiedGroupKFold grouped by subject; `n ≫ p`) before / after the subject subspace is erased (LEACE); **Δ = Identity-free − Raw**. NB: a recording-pooled decode crushes within-class variance and fabricates a large Δ — the artifact that made every cell read TRAP — so these numbers are per-trial. `subj_frac` = fraction of representation variance explained by subject identity (reported, but NOT part of the verdict); `c̄` = cross-subject direction-consistency of the task axis (≈0 ⇒ no axis that generalizes across people). **Verdict:** `no-transfer` = raw BA below the interpretability gate (no above-chance cross-subject task signal — nothing to trap); `TRAP` = interpretable raw signal that erasure lifts (> 0.02); `task-carried` = interpretable signal erasure does not lift (genuine, identity-robust task skill).

| Dataset | N subj | Raw BA | Identity-free BA | Δ (lift) | subj_frac | label_frac | c̄ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Schirrmeister2017 | 14 | 0.546 | 0.551 | 0.005 | 0.952 | 0.003 | 0.003 | no-transfer |
| BNCI2014-001 | 9 | 0.537 | 0.539 | 0.002 | 0.893 | 0.014 | 0.028 | no-transfer |
| Dreyer2023B | 21 | 0.577 | 0.577 | 0.000 | 0.910 | 0.005 | 0.002 | task-carried |
| BNCI2014-004 | 9 | 0.672 | 0.672 | 0.000 | 0.916 | 0.012 | 0.055 | task-carried |
| Dreyer2023C | 6 | 0.532 | 0.530 | -0.002 | 0.851 | 0.027 | 0.008 | no-transfer |
| Cho2017 | 52 | 0.621 | 0.618 | -0.003 | 0.910 | 0.005 | 0.031 | task-carried |
| Shin2017A | 29 | 0.541 | 0.537 | -0.004 | 0.894 | 0.004 | 0.002 | no-transfer |
| Lee2019-MI | 54 | 0.587 | 0.582 | -0.005 | 0.905 | 0.003 | 0.013 | task-carried |
| Weibo2014 | 10 | 0.530 | 0.517 | -0.013 | 0.909 | 0.005 | -0.040 | no-transfer |
| Dreyer2023 | 87 | 0.592 | 0.578 | -0.015 | 0.941 | 0.001 | 0.008 | task-carried |
| Dreyer2023A | 60 | 0.589 | 0.574 | -0.016 | 0.942 | 0.002 | 0.019 | task-carried |
| Stieger2021 | 62 | 0.730 | 0.710 | -0.020 | 0.976 | 0.006 | 0.237 | task-carried |

### Skipped / failed

- `GrosseWentrup2009` — FAILED: ValueError: none of the input electrode_names are in REVE's position vocabulary; got ['1', '2', '3', '4', '5', '6', '7', '8']...
- `Liu2024` — FAILED: BadZipFile: File is not a zip file
- `PhysionetMotorImagery` — FAILED: ValueError: operands could not be broadcast together with shapes (602,) (601,) 
- `Zhou2016` — FAILED: ValueError: Cannot have number of splits n_splits=5 greater than the number of groups: 4.
