# MOABB identity-free leaderboard — P300

Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647) on the P300 task contrast. **Raw BA** / **Identity-free BA** are the cross-subject task balanced-accuracy from the **per-trial** erasure decode (each trial its own recording, StratifiedGroupKFold grouped by subject; `n ≫ p`) before / after the subject subspace is erased (LEACE); **Δ = Identity-free − Raw**. NB: a recording-pooled decode crushes within-class variance and fabricates a large Δ — the artifact that made every cell read TRAP — so these numbers are per-trial. `subj_frac` = fraction of representation variance explained by subject identity (reported, but NOT part of the verdict); `c̄` = cross-subject direction-consistency of the task axis (≈0 ⇒ no axis that generalizes across people). **Verdict:** `no-transfer` = raw BA below the interpretability gate (no above-chance cross-subject task signal — nothing to trap); `TRAP` = interpretable raw signal that erasure lifts (> 0.02); `task-carried` = interpretable signal erasure does not lift (genuine, identity-robust task skill).

| Dataset | N subj | Raw BA | Identity-free BA | Δ (lift) | subj_frac | label_frac | c̄ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ErpCore2021-N400 | 40 | 0.688 | 0.861 | 0.174 | 0.826 | 0.041 | 0.206 | TRAP |
| ErpCore2021-N2pc | 40 | 0.771 | 0.938 | 0.167 | 0.937 | 0.008 | 0.093 | TRAP |
| ErpCore2021-N170 | 40 | 0.750 | 0.875 | 0.125 | 0.901 | 0.008 | 0.048 | TRAP |
| ErpCore2021-MMN | 40 | 0.708 | 0.826 | 0.118 | 0.937 | 0.006 | 0.047 | TRAP |
| Huebner2018 | 12 | 0.903 | 1.000 | 0.097 | 0.891 | 0.029 | 0.206 | TRAP |
| Kojima2024A | 11 | 0.864 | 0.955 | 0.091 | 0.749 | 0.105 | 0.407 | TRAP |
| BrainInvaders2014b | 38 | 0.750 | 0.840 | 0.090 | 0.888 | 0.020 | 0.147 | TRAP |
| Huebner2017 | 13 | 0.923 | 1.000 | 0.077 | 0.918 | 0.019 | 0.237 | TRAP |
| BrainInvaders2012 | 25 | 0.778 | 0.847 | 0.069 | 0.930 | 0.013 | 0.117 | TRAP |
| BrainInvaders2013a | 24 | 0.833 | 0.903 | 0.069 | 0.920 | 0.019 | 0.217 | TRAP |
| ErpCore2021-LRP | 40 | 0.931 | 1.000 | 0.069 | 0.938 | 0.009 | 0.104 | TRAP |
| BrainInvaders2015a | 43 | 0.910 | 0.958 | 0.049 | 0.938 | 0.028 | 0.456 | TRAP |
| ErpCore2021-P3 | 40 | 0.757 | 0.799 | 0.042 | 0.809 | 0.053 | 0.237 | TRAP |
| BrainInvaders2014a | 64 | 0.750 | 0.778 | 0.028 | 0.955 | 0.007 | 0.161 | TRAP |
| BNCI2015-003 | 10 | 0.783 | 0.800 | 0.017 | 0.835 | 0.020 | -0.005 | task-carried |
| Cattan2019-VR | 21 | 0.738 | 0.754 | 0.016 | 0.905 | 0.015 | 0.096 | task-carried |
| Lee2019-ERP | 54 | 0.944 | 0.958 | 0.014 | 0.776 | 0.099 | 0.516 | task-carried |
| BNCI2014-008 | 8 | 0.938 | 0.938 | 0.000 | 0.761 | 0.080 | 0.263 | task-carried |
| BNCI2014-009 | 10 | 1.000 | 1.000 | 0.000 | 0.551 | 0.218 | 0.513 | task-carried |
| DemonsP300 | 60 | 0.639 | 0.632 | -0.007 | 0.863 | 0.009 | 0.037 | task-carried |
| BrainInvaders2015b | 44 | 0.826 | 0.812 | -0.014 | 0.877 | 0.024 | 0.301 | task-carried |
| ErpCore2021-ERN | 40 | 0.965 | 0.931 | -0.035 | 0.680 | 0.097 | 0.289 | task-carried |

### Skipped / failed

- `EPFLP300` — FAILED: ConnectionError: HTTPConnectionPool(host='documents.epfl.ch', port=80): Max retries exceeded with url: /groups/m/mm/mmspg/www/BCI/p300/subject1.zip (Caused by NewConnectionError("HTTPConnectio
- `Kojima2024B` — FAILED: ValueError: No objects to concatenate
- `RomaniBF2025ERP` — FAILED: ValueError: No objects to concatenate
- `Sosulski2019` — FAILED: BadZipFile: File is not a zip file
