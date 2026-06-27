# MOABB identity-free leaderboard — LeftRightImagery

Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647). **Identity-free BA** = task balanced-accuracy after the subject subspace is erased (LEACE); **Δ = Identity-free − Raw** is the task signal *recovered* by erasure. A large positive Δ means subject-identity variance was masking the left/right axis in MOABB's cross-subject evaluation — the identity trap. `subj_frac` = fraction of representation variance explained by subject identity; `c̄` = cross-subject direction-consistency of the task axis (≈0 ⇒ the left/right axis does not generalize across people).

| Dataset | N subj | Raw BA | Identity-free BA | Δ (lift) | subj_frac | label_frac | c̄ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BNCI2014-001 | 9 | 0.667 | 0.963 | 0.296 | 0.893 | 0.014 | 0.028 | TRAP |
| Schirrmeister2017 | 14 | 0.690 | 0.964 | 0.274 | 0.952 | 0.003 | 0.003 | TRAP |
| Dreyer2023 | 87 | 0.709 | 0.966 | 0.257 | 0.941 | 0.001 | 0.008 | TRAP |
| Zhou2016 | 4 | 0.750 | 1.000 | 0.250 | 0.822 | 0.055 | 0.078 | TRAP |
| Dreyer2023B | 21 | 0.714 | 0.952 | 0.238 | 0.910 | 0.005 | 0.002 | TRAP |
| Dreyer2023A | 60 | 0.714 | 0.944 | 0.231 | 0.942 | 0.002 | 0.019 | TRAP |
| Weibo2014 | 10 | 0.500 | 0.700 | 0.200 | 0.909 | 0.005 | -0.040 | TRAP |
| Dreyer2023C | 6 | 0.639 | 0.833 | 0.194 | 0.851 | 0.027 | 0.008 | TRAP |
| Cho2017 | 52 | 0.715 | 0.907 | 0.192 | 0.910 | 0.005 | 0.031 | TRAP |
| Lee2019-MI | 54 | 0.753 | 0.889 | 0.136 | 0.905 | 0.003 | 0.013 | TRAP |
| Shin2017A | 29 | 0.563 | 0.690 | 0.126 | 0.894 | 0.004 | 0.002 | TRAP |
| BNCI2014-004 | 9 | 0.889 | 1.000 | 0.111 | 0.916 | 0.012 | 0.055 | TRAP |
| Stieger2021 | 62 | 0.981 | 1.000 | 0.019 | 0.976 | 0.006 | 0.237 | TRAP |

## Depth & 1/f diagnostics

`L first/last/max` = subject-grouped label balanced-accuracy at the shallowest / deepest / best REVE block; `argmax` = depth-fraction of the best block. `sign`: `+early` = task signal peaks shallow then the head trades it away (max−last ≥0.04, argmax ≤0.35); `−deep` = final block barely separates the task (≤0.45). `state_drop` / `subj_drop` = label / subject BA lost when the 1/f aperiodic slope is ablated (FOOOF); `role`: `state signal` (1/f helps the task, drop >0.03) vs `subject confound` (1/f is an identity fingerprint, drop >0.05).

| Dataset | L first | L last | L max | argmax | sign | state_drop | subj_drop | 1/f role |
|---|---:|---:|---:|---:|---|---:|---:|---|
| BNCI2014-001 | 0.504 | 0.543 | 0.560 | 0.41 | — | 0.048 | 0.177 | subject confound;state signal |
| Schirrmeister2017 | 0.529 | 0.543 | 0.565 | 0.73 | — | 0.019 | 0.393 | subject confound |
| Dreyer2023 | 0.531 | 0.568 | 0.601 | 0.32 | — | 0.089 | 0.501 | subject confound;state signal |
| Zhou2016 | 0.521 | 0.559 | 0.612 | 0.41 | — | 0.065 | 0.286 | subject confound;state signal |
| Dreyer2023B | 0.522 | 0.547 | 0.583 | 0.36 | — | 0.032 | 0.409 | subject confound;state signal |
| Dreyer2023A | 0.535 | 0.556 | 0.596 | 0.36 | — | 0.075 | 0.509 | subject confound;state signal |
| Weibo2014 | 0.484 | 0.514 | 0.563 | 0.73 | — | 0.010 | 0.389 | subject confound |
| Dreyer2023C | 0.474 | 0.529 | 0.529 | 1.00 | — | 0.000 | 0.272 | subject confound |
| Cho2017 | 0.548 | 0.572 | 0.628 | 0.27 | +early | 0.113 | 0.417 | subject confound;state signal |
| Lee2019-MI | 0.522 | 0.551 | 0.584 | 0.36 | — | 0.072 | 0.348 | subject confound;state signal |
| Shin2017A | 0.514 | 0.529 | 0.570 | 0.50 | — | 0.051 | 0.594 | subject confound;state signal |
| BNCI2014-004 | 0.580 | 0.602 | 0.634 | 0.68 | — | 0.089 | 0.115 | subject confound;state signal |
| Stieger2021 | 0.596 | 0.613 | 0.715 | 0.32 | +early | 0.189 | 0.272 | subject confound;state signal |

### Skipped / failed

- `GrosseWentrup2009` — FAILED: ValueError: none of the input electrode_names are in REVE's position vocabulary; got ['1', '2', '3', '4', '5', '6', '7', '8']...
- `PhysionetMotorImagery` — FAILED: ValueError: operands could not be broadcast together with shapes (602,) (601,) 
- `Liu2024` — FAILED: BadZipFile: File is not a zip file
