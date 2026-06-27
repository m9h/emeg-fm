# BNCI2014_001 — per-trial vs pooled subject-axis erasure

REVE block-6 (brain-bzh/reve-base), 9 subjects, 2592 trials, CV=stratified-kfold. Same features both ways; only the recording grouping (and thus prediction pooling) differs.

| grouping | raw | erased | lift | subj BA pre→post (chance) | interp | degenerate |
|---|---:|---:|---:|:--|:--|:--|
| pooled_subject_class | 0.667 | 0.963 | +0.296±0.069 | 0.61→0.06 (0.111) | True | False |
| per_trial | 0.536 | 0.540 | +0.004±0.005 | 0.61→0.06 (0.111) | False | False |

**Verdict:** ARTIFACT — pooled lift does not survive per-trial
