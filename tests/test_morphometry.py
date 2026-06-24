"""TDD for emeg_fm.morphometry — parsing FastSurfer .stats and assembling the feature matrix."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from morphometry import parse_volume_stats, assemble_morphometry, normalize_by_total   # noqa: E402

_STATS = """\
# Title Segmentation Statistics
# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1   2  300000  300000.0  Left-Cerebral-White-Matter  100 5 80 120 40
  2   3  250000  250123.5  Left-Cerebral-Cortex        90  6 70 110 40
  3  41  295000  295000.0  Right-Cerebral-White-Matter 101 5 81 121 40
"""


def test_parse_volume_stats():
    with tempfile.NamedTemporaryFile("w", suffix=".stats", delete=False) as f:
        f.write(_STATS); p = f.name
    d = parse_volume_stats(p)
    os.unlink(p)
    assert d["Left-Cerebral-Cortex"] == 250123.5
    assert d["Right-Cerebral-White-Matter"] == 295000.0
    assert len(d) == 3 and "ColHeaders" not in d        # comment lines skipped


def test_assemble_aligns_and_fills_missing():
    sbs = {
        "sub-A": {"GM": 200.0, "WM": 300.0, "Thalamus": 8.0},
        "sub-B": {"GM": 210.0, "WM": 290.0},               # missing Thalamus
    }
    X, regions, ids = assemble_morphometry(sbs)
    assert ids == ["sub-A", "sub-B"]
    assert regions == ["GM", "Thalamus", "WM"]             # sorted union
    ti = regions.index("Thalamus")
    assert X[1, ti] == 0.0                                 # missing filled with 0
    assert X[0, regions.index("GM")] == 200.0


def test_pinned_region_order():
    sbs = {"s": {"B": 2.0, "A": 1.0}}
    X, regions, _ = assemble_morphometry(sbs, regions=["A", "B", "C"])
    assert regions == ["A", "B", "C"]
    assert list(X[0]) == [1.0, 2.0, 0.0]


def test_normalize_by_total():
    X = np.array([[1.0, 3.0], [2.0, 2.0]])
    Xn = normalize_by_total(X)
    assert np.allclose(Xn.sum(1), 1.0)
    assert np.allclose(Xn[0], [0.25, 0.75])


if __name__ == "__main__":
    for _fn in (test_parse_volume_stats, test_assemble_aligns_and_fills_missing,
                test_pinned_region_order, test_normalize_by_total):
        _fn(); print(f"PASS  {_fn.__name__}")
    print("all morphometry tests passed")
