"""M3CV enrollment loader: info-csv grouping/capping + 65->64 channel drop.
scripts/eegfm_t9.sh python -m pytest tests/test_m3cv_identity.py -q
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import atlas_m3cv_identity as mi  # noqa: E402


def _write_info(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EpochID", "subject", "session", "condition", "usage"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_load_m3cv_enrollment_groups_by_subject_condition_and_caps_cell(tmp_path, monkeypatch):
    root = tmp_path
    (root / "Enrollment").mkdir()
    rows = []
    for subj in ("sub001", "sub002"):
        for cond in ("1", "2"):
            for i in range(4):
                eid = f"epoch_{subj}_{cond}_{i}"
                rows.append({"EpochID": eid, "subject": subj, "session": "1",
                             "condition": cond, "usage": "1"})
                d = np.zeros((65, 1000), dtype=np.float32)
                d[:] = int(cond)  # tag data with condition value for a sanity check
                import scipy.io as sio
                sio.savemat(root / "Enrollment" / f"{eid}.mat", {"epoch_data": d})
    _write_info(root / "Enrollment_Info.csv", rows)

    monkeypatch.setattr(mi, "ROOT", str(root))
    X, y, subj = mi.load_m3cv_enrollment(max_per_cell=2)

    # 2 subjects x 2 conditions x capped-to-2 = 8 epochs
    assert X.shape == (8, 64, 1000)
    assert len(y) == 8 and len(subj) == 8
    assert set(subj.tolist()) == {"sub001", "sub002"}
    # y is 0-indexed condition-1; conditions "1"/"2" -> y in {0,1}
    assert set(y.tolist()) == {0, 1}
    # 64 channels kept, 65th (constant marker in this synthetic fixture) dropped
    assert X.shape[1] == 64


def test_load_m3cv_enrollment_respects_max_subjects(tmp_path, monkeypatch):
    root = tmp_path
    (root / "Enrollment").mkdir()
    rows = []
    for subj in ("sub001", "sub002", "sub003"):
        eid = f"epoch_{subj}"
        rows.append({"EpochID": eid, "subject": subj, "session": "1",
                     "condition": "1", "usage": "1"})
        import scipy.io as sio
        sio.savemat(root / "Enrollment" / f"{eid}.mat",
                    {"epoch_data": np.zeros((65, 1000), dtype=np.float32)})
    _write_info(root / "Enrollment_Info.csv", rows)

    monkeypatch.setattr(mi, "ROOT", str(root))
    X, y, subj = mi.load_m3cv_enrollment(max_subjects=2)
    assert len(set(subj.tolist())) == 2
