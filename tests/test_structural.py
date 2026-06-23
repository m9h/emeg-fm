"""Red-green TDD for the pure-numpy cores of emeg_fm.structural (block_pool, assemble)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from structural import block_pool, assemble   # noqa: E402  (red until the module exists)


def test_block_pool_shape_and_constant():
    v = np.full((20, 17, 23), 3.0)
    f = block_pool(v, grid=(4, 4, 4))
    assert f.shape == (64,)
    assert np.allclose(f, 3.0)                        # constant volume → constant pooled cells


def test_block_pool_monotone_gradient():
    x = np.arange(16)[:, None, None] * np.ones((16, 8, 8))   # increasing along axis 0
    f = block_pool(x, grid=(4, 1, 1))                         # 4 slabs along axis 0
    assert np.all(np.diff(f) > 0)                             # pooled slab means increase


def test_assemble_aligns_and_subsets():
    ids = ["sub-A", "sub-B", "sub-C", "sub-D"]
    X = np.arange(4 * 5).reshape(4, 5).astype(float)
    ages = np.array([7.0, 8.0, 9.0, 10.0])
    anat = {"sub-A": np.ones(3), "sub-C": 2 * np.ones(3)}     # only A, C have anatomy
    E, A, y, kept = assemble(ids, X, ages, anat)
    assert kept == ["sub-A", "sub-C"]
    assert np.array_equal(E, X[[0, 2]]) and np.array_equal(y, ages[[0, 2]])
    assert A.shape == (2, 3) and np.allclose(A[1], 2.0)


if __name__ == "__main__":
    for _fn in (test_block_pool_shape_and_constant, test_block_pool_monotone_gradient,
                test_assemble_aligns_and_subsets):
        _fn(); print(f"PASS  {_fn.__name__}")
    print("all structural tests passed")
