# MOABB identity-free leaderboard — P300

Frozen REVE (block 6) features audited with FMScope (arXiv 2606.06647) on the P300 task contrast. **Raw BA** / **Identity-free BA** are the cross-subject task balanced-accuracy from the **per-trial** erasure decode (each trial its own recording, StratifiedGroupKFold grouped by subject; `n ≫ p`) before / after the subject subspace is erased (LEACE); **Δ = Identity-free − Raw**. NB: a recording-pooled decode crushes within-class variance and fabricates a large Δ — the artifact that made every cell read TRAP — so these numbers are per-trial. `subj_frac` = fraction of representation variance explained by subject identity (reported, but NOT part of the verdict); `c̄` = cross-subject direction-consistency of the task axis (≈0 ⇒ no axis that generalizes across people). **Verdict:** `no-transfer` = raw BA below the interpretability gate (no above-chance cross-subject task signal — nothing to trap); `TRAP` = interpretable raw signal that erasure lifts (> 0.02); `task-carried` = interpretable signal erasure does not lift (genuine, identity-robust task skill).

| Dataset | N subj | Raw BA | Identity-free BA | Δ (lift) | subj_frac | label_frac | c̄ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DemonsP300 | 60 | 0.537 | 0.552 | 0.015 | 0.863 | 0.009 | 0.037 | no-transfer |
| RomaniBF2025ERP | 22 | 0.673 | 0.679 | 0.007 | 0.892 | 0.016 | 0.129 | task-carried |
| BrainInvaders2015a | 43 | 0.657 | 0.663 | 0.006 | 0.938 | 0.028 | 0.456 | task-carried |
| BrainInvaders2012 | 25 | 0.628 | 0.633 | 0.005 | 0.930 | 0.013 | 0.117 | task-carried |
| Kojima2024A | 11 | 0.610 | 0.615 | 0.005 | 0.749 | 0.105 | 0.407 | task-carried |
| BrainInvaders2015b | 44 | 0.610 | 0.614 | 0.004 | 0.877 | 0.024 | 0.301 | task-carried |
| ErpCore2021-LRP | 40 | 0.653 | 0.657 | 0.004 | 0.938 | 0.009 | 0.104 | task-carried |
| BNCI2014-009 | 10 | 0.741 | 0.744 | 0.003 | 0.551 | 0.218 | 0.513 | task-carried |
| ErpCore2021-N170 | 40 | 0.628 | 0.631 | 0.003 | 0.901 | 0.008 | 0.048 | task-carried |
| Huebner2018 | 12 | 0.635 | 0.638 | 0.003 | 0.891 | 0.029 | 0.206 | task-carried |
| BrainInvaders2013a | 24 | 0.639 | 0.640 | 0.002 | 0.920 | 0.019 | 0.217 | task-carried |
| Huebner2017 | 13 | 0.648 | 0.649 | 0.001 | 0.918 | 0.019 | 0.237 | task-carried |
| ErpCore2021-N2pc | 40 | 0.576 | 0.576 | -0.000 | 0.937 | 0.008 | 0.093 | task-carried |
| BrainInvaders2014a | 64 | 0.575 | 0.574 | -0.000 | 0.955 | 0.007 | 0.161 | task-carried |
| ErpCore2021-MMN | 40 | 0.552 | 0.552 | -0.000 | 0.937 | 0.006 | 0.047 | task-carried |
| BNCI2014-008 | 8 | 0.640 | 0.640 | -0.000 | 0.761 | 0.080 | 0.263 | task-carried |
| ErpCore2021-N400 | 40 | 0.587 | 0.586 | -0.000 | 0.826 | 0.041 | 0.206 | task-carried |
| Cattan2019-VR | 21 | 0.585 | 0.584 | -0.001 | 0.905 | 0.015 | 0.096 | task-carried |
| Lee2019-ERP | 54 | 0.675 | 0.675 | -0.001 | 0.776 | 0.099 | 0.516 | task-carried |
| BrainInvaders2014b | 38 | 0.607 | 0.606 | -0.001 | 0.888 | 0.020 | 0.147 | task-carried |
| ErpCore2021-P3 | 40 | 0.614 | 0.611 | -0.004 | 0.809 | 0.053 | 0.237 | task-carried |
| BNCI2015-003 | 10 | 0.597 | 0.588 | -0.009 | 0.835 | 0.020 | -0.005 | task-carried |
| ErpCore2021-ERN | 40 | 0.750 | 0.732 | -0.018 | 0.680 | 0.097 | 0.289 | task-carried |

### Skipped / failed

- `EPFLP300` — FAILED: ConnectionError: HTTPConnectionPool(host='documents.epfl.ch', port=80): Max retries exceeded with url: /groups/m/mm/mmspg/www/BCI/p300/subject1.zip (Caused by NewConnectionError("HTTPConnectio
- `Kojima2024B` — FAILED: ValueError: No objects to concatenate
- `Sosulski2019` — FAILED: BadZipFile: File is not a zip file
