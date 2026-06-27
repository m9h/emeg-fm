# ERP CORE decoding — Luck-lab parity comparison

Linear-SVM pseudo-ERP decoding under the ERPLAB12 `pop_decoding_regularization` protocol (3-fold crossblock, Gamma grid, EqualizeTrials). `scalp_within`/`reve_within` are within-participant grand-average accuracy (Luck's native regime) on raw scalp voltage vs frozen REVE block-6 embeddings. `reve_*_cross` are cross-subject (leave-subject-block-out) decode before/after LEACE subject-axis erasure; `lift` = identity-free − raw.

| Component | N | Scalp (within) | REVE (within) | REVE raw (cross) | REVE id-free (cross) | Lift |
|---|---:|---:|---:|---:|---:|---:|
| N170 | 40 | 0.666 | 0.816 | 0.792 | 1.000 | 0.208 |
| MMN | 40 | 0.661 | 0.638 | 0.863 | 0.998 | 0.135 |
| N2pc | 40 | 0.620 | 0.664 | 0.702 | 0.977 | 0.275 |
| P3 | 40 | 0.709 | 0.755 | 0.932 | 0.998 | 0.067 |
| N400 | 40 | 0.741 | 0.779 | 0.962 | 0.997 | 0.035 |
| ERN | 40 | 0.862 | 0.897 | 0.970 | 1.000 | 0.030 |
| LRP | 40 | 0.839 | 0.770 | 0.873 | 1.000 | 0.127 |
