# MOABB identity-free leaderboard — SSVEP

Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647) on the SSVEP task contrast. **Identity-free BA** = task balanced-accuracy after the subject subspace is erased (LEACE); **Δ = Identity-free − Raw** is the task signal *recovered* by erasure. A large positive Δ means subject-identity variance was masking the task axis in MOABB's cross-subject evaluation — the identity trap. `subj_frac` = fraction of representation variance explained by subject identity; `c̄` = cross-subject direction-consistency of the task axis (≈0 ⇒ the task axis does not generalize across people).

| Dataset | N subj | Raw BA | Identity-free BA | Δ (lift) | subj_frac | label_frac | c̄ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MAMEM2 | 10 | 0.517 | 0.550 | 0.033 | 0.874 | 0.015 | -0.012 | TRAP |
| Kalunga2016 | 12 | 0.681 | 0.708 | 0.028 | 0.669 | 0.030 | -0.003 | TRAP |
| MAMEM1 | 10 | 0.450 | 0.467 | 0.017 | 0.858 | 0.016 | -0.070 | TRAP |
| MAMEM3 | 10 | 0.467 | 0.433 | -0.033 | 0.901 | 0.011 | 0.034 | identity-reliant |

## Depth & 1/f diagnostics

`L first/last/max` = subject-grouped label balanced-accuracy at the shallowest / deepest / best REVE block; `argmax` = depth-fraction of the best block. `sign`: `+early` = task signal peaks shallow then the head trades it away (max−last ≥0.04, argmax ≤0.35); `−deep` = final block barely separates the task (≤0.45). `state_drop` / `subj_drop` = label / subject BA lost when the 1/f aperiodic slope is ablated (FOOOF); `role`: `state signal` (1/f helps the task, drop >0.03) vs `subject confound` (1/f is an identity fingerprint, drop >0.05).

| Dataset | L first | L last | L max | argmax | sign | state_drop | subj_drop | 1/f role |
|---|---:|---:|---:|---:|---|---:|---:|---|
| MAMEM2 | 0.442 | 0.508 | 0.557 | 0.68 | — | 0.046 | 0.530 | subject confound;state signal |
| Kalunga2016 | 0.535 | 0.526 | 0.547 | 0.41 | — | 0.017 | 0.171 | subject confound |
| MAMEM1 | 0.496 | 0.506 | 0.530 | 0.55 | — | -0.021 | 0.604 | subject confound |
| MAMEM3 | 0.465 | 0.527 | 0.542 | 0.32 | — | -0.006 | 0.542 | subject confound |

### Skipped / failed

- `Nakanishi2015` — FAILED: TypeError: reshape() got an unexpected keyword argument 'newshape'
