"""TDD for emeg_fm.leadfield.block_pool_field — the tier-3 leadfield-summary core (pure numpy)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from leadfield import block_pool_field   # noqa: E402


def test_pools_into_normalized_grid_shape_and_mean():
    rng = np.random.default_rng(0)
    centroids = rng.uniform(-30, 50, size=(2000, 3))      # arbitrary bbox / offset
    mag = rng.uniform(0, 1, size=(4, 2000))               # 4 electrodes
    out = block_pool_field(mag, centroids, grid=(2, 2, 2))
    assert out.shape == (4, 8)                            # n_elec × prod(grid)
    assert np.isfinite(out).all() and (out >= 0).all()


def test_localizes_planted_hot_cell():
    # all centroids in one octant get a high value for electrode 0 -> that cell dominates
    rng = np.random.default_rng(1)
    centroids = rng.uniform(0, 1, size=(4000, 3))
    mag = np.full((1, 4000), 0.1)
    hot = (centroids[:, 0] > 0.5) & (centroids[:, 1] > 0.5) & (centroids[:, 2] > 0.5)  # top octant
    mag[0, hot] = 10.0
    out = block_pool_field(mag, centroids, grid=(2, 2, 2))[0]
    hot_cell = (1 * 2 + 1) * 2 + 1                        # row-major index of (1,1,1)
    assert out[hot_cell] == out.max() and out[hot_cell] > 5.0
    assert out[hot_cell] > 50 * np.median(np.delete(out, hot_cell))


def test_empty_cells_are_zero_not_nan():
    centroids = np.zeros((10, 3))                         # all in one corner -> 7 of 8 cells empty
    out = block_pool_field(np.ones((1, 10)), centroids, grid=(2, 2, 2))
    assert np.isfinite(out).all()
    assert (out == 0).sum() == 7 and out.max() == 1.0


if __name__ == "__main__":
    for _fn in (test_pools_into_normalized_grid_shape_and_mean,
                test_localizes_planted_hot_cell,
                test_empty_cells_are_zero_not_nan):
        _fn(); print(f"PASS  {_fn.__name__}")
    print("all leadfield tests passed")
