# MOABB identity-free leaderboard — LeftRightImagery

Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647). **Identity-free BA** = task balanced-accuracy after the subject subspace is erased (LEACE); **Δ = Identity-free − Raw** is the task signal *recovered* by erasure. A large positive Δ means subject-identity variance was masking the left/right axis in MOABB's cross-subject evaluation — the identity trap. `subj_frac` = fraction of representation variance explained by subject identity; `c̄` = cross-subject direction-consistency of the task axis (≈0 ⇒ the left/right axis does not generalize across people).

| Dataset | N subj | Raw BA | Identity-free BA | Δ (lift) | subj_frac | label_frac | c̄ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BNCI2014-001 | 9 | 0.667 | 1.000 | 0.333 | 0.893 | 0.014 | 0.028 | TRAP |
| Zhou2016 | 4 | 0.750 | 1.000 | 0.250 | 0.822 | 0.055 | 0.078 | TRAP |
| BNCI2014-004 | 9 | 0.944 | 1.000 | 0.056 | 0.916 | 0.012 | 0.055 | TRAP |
