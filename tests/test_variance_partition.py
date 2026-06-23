"""Red-green TDD for emeg_fm.variance_partition — the tier-1 EEG↔anatomy variance partition.

Commonality analysis (subject-level CV ridge R²) of how much of EEG's age signal is **redundant with
anatomy** (reproducible from structural VBM/DWI features — consistent with volume conduction) vs
**unique to EEG** (not linearly recoverable from anatomy — candidate neural signal). NB this is a
correlational redundancy split, NOT causal proof of conduction; the forward model (tier 3) is the
causal test. Pure numpy so it runs under any numpy interpreter.

    <numpy-python> tests/test_variance_partition.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from variance_partition import variance_partition   # noqa: E402  (red until the module exists)


def _normcols(M):
    return M / (M.std(0, keepdims=True) + 1e-8)


def _synth(n, d_eeg, d_anat, conduction, neural, seed):
    """Plant a known redundancy split. Anatomy sees a *shared-latent noisy* age (caps R²_A≈0.4 — per-dim
    independent noise would otherwise average out to a near-perfect predictor). EEG = a conduction part
    f(A) (redundant with anatomy) + a neural part = a *cleaner* age channel anatomy lacks (gives EEG
    unique age info) + noise."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    age_a = y + 1.2 * rng.standard_normal(n)                                # anatomy's noisy age (shared) → caps R²_A
    A = age_a[:, None] * rng.standard_normal((1, d_anat)) + 0.3 * rng.standard_normal((n, d_anat))
    E_cond = _normcols(A @ rng.standard_normal((d_anat, d_eeg)))            # in span(A) → redundant
    age_e = y + 0.1 * rng.standard_normal(n)                                # EEG's cleaner age (different shared noise)
    E_neural = _normcols(age_e[:, None] * rng.standard_normal((1, d_eeg)) + 0.05 * rng.standard_normal((n, d_eeg)))
    E = conduction * E_cond + neural * E_neural + 0.3 * rng.standard_normal((n, d_eeg))
    return E, A, y


def test_pure_redundant_eeg_age_is_anatomy_explainable():
    E, A, y = _synth(600, 40, 20, conduction=1.0, neural=0.0, seed=1)
    r = variance_partition(E, A, y)
    assert r["r2_eeg"] > 0.2 and r["r2_anat"] > 0.2          # both carry age
    assert r["redundant_fraction"] > 0.7                     # EEG age-signal mostly reproducible from anatomy
    assert r["eeg_unique_fraction"] < 0.3


def test_eeg_unique_when_eeg_has_a_cleaner_age_channel():
    E, A, y = _synth(600, 40, 20, conduction=0.3, neural=1.0, seed=2)
    r = variance_partition(E, A, y)
    assert r["r2_eeg"] > 0.2
    assert r["eeg_unique_fraction"] > 0.4                    # EEG age-info anatomy cannot reproduce
    assert r["eeg_unique_fraction"] > r["redundant_fraction"]


def test_fractions_and_keys_well_formed():
    E, A, y = _synth(400, 30, 15, conduction=0.7, neural=0.7, seed=3)
    r = variance_partition(E, A, y)
    for k in ("r2_eeg", "r2_anat", "r2_joint", "redundant", "eeg_unique", "anat_unique",
              "redundant_fraction", "eeg_unique_fraction"):
        assert k in r
    assert -0.1 <= r["redundant_fraction"] <= 1.1


if __name__ == "__main__":
    for _fn in (test_pure_redundant_eeg_age_is_anatomy_explainable,
                test_eeg_unique_when_eeg_has_a_cleaner_age_channel,
                test_fractions_and_keys_well_formed):
        _fn(); print(f"PASS  {_fn.__name__}")
    print("all variance-partition tests passed")
