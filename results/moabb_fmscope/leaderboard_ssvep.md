# MOABB identity-free leaderboard — SSVEP

Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647) on the SSVEP task contrast. **Raw BA** / **Identity-free BA** are the cross-subject task balanced-accuracy from the **per-trial** erasure decode (each trial its own recording, StratifiedGroupKFold grouped by subject; `n ≫ p`) before / after the subject subspace is erased (LEACE); **Δ = Identity-free − Raw**. NB: a recording-pooled decode crushes within-class variance and fabricates a large Δ — the artifact that made every cell read TRAP — so these numbers are per-trial. `subj_frac` = fraction of representation variance explained by subject identity (reported, but NOT part of the verdict); `c̄` = cross-subject direction-consistency of the task axis (≈0 ⇒ no axis that generalizes across people). **Verdict:** `no-transfer` = raw BA below the interpretability gate (no above-chance cross-subject task signal — nothing to trap); `TRAP` = interpretable raw signal that erasure lifts (> 0.02); `task-carried` = interpretable signal erasure does not lift (genuine, identity-robust task skill).

| Dataset | N subj | Raw BA | Identity-free BA | Δ (lift) | subj_frac | label_frac | c̄ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Lee2019-SSVEP | 54 | 0.764 | 1.000 | 0.236 | 0.857 | 0.012 | 0.053 | TRAP |
| Nakanishi2015 | 9 | 0.926 | 1.000 | 0.074 | 0.724 | 0.115 | 0.305 | TRAP |
| MAMEM2 | 10 | 0.517 | 0.550 | 0.033 | 0.874 | 0.015 | -0.012 | no-transfer |
| Kalunga2016 | 12 | 0.681 | 0.708 | 0.028 | 0.669 | 0.030 | -0.003 | TRAP |
| MAMEM1 | 10 | 0.450 | 0.467 | 0.017 | 0.858 | 0.016 | -0.070 | no-transfer |
| MAMEM3 | 10 | 0.467 | 0.433 | -0.033 | 0.901 | 0.011 | 0.034 | no-transfer |

### Skipped / failed

- `Wang2016` — FAILED: ValueError: DigMontage is only a subset of info. There are 2 channel positions not present in the DigMontage. The channels missing from the montage are:

['CB1', 'CB2'].

Consider using inst.r
