# MOABB identity-free leaderboard — P300

Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647) on the P300 task contrast. **Identity-free BA** = task balanced-accuracy after the subject subspace is erased (LEACE); **Δ = Identity-free − Raw** is the task signal *recovered* by erasure. A large positive Δ means subject-identity variance was masking the task axis in MOABB's cross-subject evaluation — the identity trap. `subj_frac` = fraction of representation variance explained by subject identity; `c̄` = cross-subject direction-consistency of the task axis (≈0 ⇒ the task axis does not generalize across people).

| Dataset | N subj | Raw BA | Identity-free BA | Δ (lift) | subj_frac | label_frac | c̄ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ErpCore2021-N400 | 40 | 0.758 | 0.971 | 0.212 | 0.835 | 0.035 | 0.187 | TRAP |
| ErpCore2021-N2pc | 40 | 0.717 | 0.904 | 0.188 | 0.943 | 0.004 | 0.055 | TRAP |
| ErpCore2021-N170 | 40 | 0.779 | 0.900 | 0.121 | 0.906 | 0.010 | 0.084 | TRAP |
| ErpCore2021-LRP | 40 | 0.892 | 1.000 | 0.108 | 0.938 | 0.009 | 0.121 | TRAP |
| ErpCore2021-P3 | 40 | 0.767 | 0.871 | 0.104 | 0.818 | 0.057 | 0.284 | TRAP |
| ErpCore2021-MMN | 40 | 0.804 | 0.887 | 0.083 | 0.927 | 0.006 | 0.065 | TRAP |
| ErpCore2021-ERN | 40 | 0.987 | 0.988 | 0.000 | 0.682 | 0.100 | 0.299 | TRAP |
| BNCI2014-008 | 8 | 0.938 | 0.938 | 0.000 | 0.761 | 0.080 | 0.263 | TRAP |
| BNCI2014-009 | 10 | 1.000 | 1.000 | 0.000 | 0.551 | 0.218 | 0.513 | TRAP |

## Depth & 1/f diagnostics

`L first/last/max` = subject-grouped label balanced-accuracy at the shallowest / deepest / best REVE block; `argmax` = depth-fraction of the best block. `sign`: `+early` = task signal peaks shallow then the head trades it away (max−last ≥0.04, argmax ≤0.35); `−deep` = final block barely separates the task (≤0.45). `state_drop` / `subj_drop` = label / subject BA lost when the 1/f aperiodic slope is ablated (FOOOF); `role`: `state signal` (1/f helps the task, drop >0.03) vs `subject confound` (1/f is an identity fingerprint, drop >0.05).

| Dataset | L first | L last | L max | argmax | sign | state_drop | subj_drop | 1/f role |
|---|---:|---:|---:|---:|---|---:|---:|---|
| ErpCore2021-N400 | 0.588 | 0.593 | 0.619 | 0.18 | — | 0.046 | 0.404 | subject confound;state signal |
| ErpCore2021-N2pc | 0.527 | 0.522 | 0.554 | 0.23 | — | 0.031 | 0.399 | subject confound;state signal |
| ErpCore2021-N170 | 0.567 | 0.588 | 0.640 | 0.18 | +early | 0.080 | 0.427 | subject confound;state signal |
| ErpCore2021-LRP | 0.558 | 0.548 | 0.636 | 0.32 | +early | 0.110 | 0.412 | subject confound;state signal |
| ErpCore2021-P3 | 0.601 | 0.628 | 0.647 | 0.36 | — | 0.029 | 0.387 | subject confound |
| ErpCore2021-MMN | 0.519 | 0.522 | 0.550 | 0.09 | — | 0.021 | 0.323 | subject confound |
| ErpCore2021-ERN | 0.643 | 0.693 | 0.778 | 0.18 | +early | 0.124 | 0.220 | subject confound;state signal |
| BNCI2014-008 | 0.533 | 0.547 | 0.589 | 0.32 | +early | 0.061 | 0.173 | subject confound;state signal |
| BNCI2014-009 | 0.626 | 0.648 | 0.735 | 0.14 | +early | 0.157 | 0.067 | subject confound;state signal |

### Skipped / failed

- `Sosulski2019` — FAILED: BadZipFile: File is not a zip file
