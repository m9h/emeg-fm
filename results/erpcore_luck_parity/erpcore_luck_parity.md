# ERP CORE decoding — Luck-lab parity comparison

**Within-participant** (Luck's native regime): linear-SVM pseudo-ERP decoding under the ERPLAB12 `pop_decoding_regularization` protocol (3-fold crossblock, Gamma grid, EqualizeTrials), best over the grid. `scalp_within` is raw scalp voltage (the ERPLAB DECODE baseline); `reve_within` is frozen REVE block-6 embeddings under the same protocol.

**Cross-subject** (`reve_cross_*`): single-trial decode of the component label, one trial = one recording, StratifiedGroupKFold grouped by subject (train/test never share a subject), balanced accuracy — `n ≫ p`, so the score reflects generalization, not separability. `raw` → before, `free` → after LEACE subject-axis erasure; `lift = free − raw` (± across-seed SD) is the identity-trap lift. It is meaningful only when **interpretable** (raw ≥ 0.55) and not **degenerate** (subject subspace < 0.95·dim); `subj BA` shows the linear identity axis (pre → post erasure, vs chance) that was removed.

| Component | N | Scalp (within) | REVE (within) | REVE cross raw | REVE cross id-free | Lift | subj BA (pre→post / chance) | valid |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| N170 | 40 | 0.666 | 0.816 | 0.652 | 0.637 | -0.015 ±0.003 | 0.787→0.029 / 0.025 | ✓ |
| MMN | 40 | 0.661 | 0.638 | 0.562 | 0.561 | -0.001 ±0.001 | 0.584→0.043 / 0.025 | ✓ |
| N2pc | 40 | 0.620 | 0.664 | 0.567 | 0.566 | -0.002 ±0.002 | 0.770→0.019 / 0.025 | ✓ |
| P3 | 40 | 0.709 | 0.755 | 0.643 | 0.628 | -0.015 ±0.003 | 0.774→0.016 / 0.025 | ✓ |
| N400 | 40 | 0.741 | 0.779 | 0.608 | 0.601 | -0.007 ±0.002 | 0.777→0.002 / 0.025 | ✓ |
| ERN | 40 | 0.862 | 0.897 | 0.803 | 0.773 | -0.029 ±0.003 | 0.707→0.027 / 0.025 | ✓ |
| LRP | 40 | 0.839 | 0.770 | 0.649 | 0.643 | -0.006 ±0.002 | 0.665→0.025 / 0.025 | ✓ |
