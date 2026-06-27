# MOABB identity-free leaderboard — LeftRightImagery

Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647). **Identity-free BA** = task balanced-accuracy after the subject subspace is erased (LEACE); **Δ = Identity-free − Raw** is the task signal *recovered* by erasure. A large positive Δ means subject-identity variance was masking the left/right axis in MOABB's cross-subject evaluation — the identity trap. `subj_frac` = fraction of representation variance explained by subject identity; `c̄` = cross-subject direction-consistency of the task axis (≈0 ⇒ the left/right axis does not generalize across people).

| Dataset | N subj | Raw BA | Identity-free BA | Δ (lift) | subj_frac | label_frac | c̄ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Kumar2024 | 18 | 0.648 | 0.972 | 0.324 | 0.992 | 0.001 | -0.008 | TRAP |
| BNCI2014-001 | 9 | 0.667 | 0.963 | 0.296 | 0.893 | 0.014 | 0.028 | TRAP |
| Schirrmeister2017 | 14 | 0.690 | 0.964 | 0.274 | 0.952 | 0.003 | 0.003 | TRAP |
| Dreyer2023 | 87 | 0.709 | 0.966 | 0.257 | 0.941 | 0.001 | 0.008 | TRAP |
| Zhou2016 | 4 | 0.750 | 1.000 | 0.250 | 0.822 | 0.055 | 0.078 | TRAP |
| Dreyer2023B | 21 | 0.714 | 0.952 | 0.238 | 0.910 | 0.005 | 0.002 | TRAP |
| Dreyer2023A | 60 | 0.714 | 0.944 | 0.231 | 0.942 | 0.002 | 0.019 | TRAP |
| Weibo2014 | 10 | 0.500 | 0.700 | 0.200 | 0.909 | 0.005 | -0.040 | TRAP |
| Dreyer2023C | 6 | 0.639 | 0.833 | 0.194 | 0.851 | 0.027 | 0.008 | TRAP |
| Cho2017 | 52 | 0.715 | 0.907 | 0.192 | 0.910 | 0.005 | 0.031 | TRAP |
| Lee2019-MI | 54 | 0.698 | 0.858 | 0.160 | 0.855 | 0.004 | 0.009 | TRAP |
| Kaya2018 | 7 | 0.833 | 0.976 | 0.143 | 0.954 | 0.011 | 0.128 | TRAP |
| Shin2017A | 29 | 0.563 | 0.690 | 0.126 | 0.894 | 0.004 | 0.002 | TRAP |
| BNCI2014-004 | 9 | 0.889 | 1.000 | 0.111 | 0.916 | 0.012 | 0.055 | TRAP |
| HefmiIch2025 | 37 | 0.572 | 0.613 | 0.041 | 0.928 | 0.002 | -0.008 | TRAP |
| Stieger2021 | 62 | 0.981 | 1.000 | 0.019 | 0.976 | 0.006 | 0.237 | TRAP |
| Liu2024 | 50 | 0.573 | 0.580 | 0.007 | 0.893 | 0.002 | -0.005 | TRAP |

### Skipped / failed

- `GrosseWentrup2009` — FAILED: ValueError: none of the input electrode_names are in REVE's position vocabulary; got ['1', '2', '3', '4', '5', '6', '7', '8']...
- `PhysionetMotorImagery` — FAILED: ValueError: operands could not be broadcast together with shapes (602,) (601,) 
