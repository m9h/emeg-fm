"""TDD for the natview EEG->BOLD prediction pipeline primitives (emeg_fm.natview)."""
import numpy as np
import pytest

from emeg_fm import natview as nv


def test_roi_timecourse_means_over_mask():
    bold = np.zeros((2, 2, 2, 3))
    bold[0, 0, 0, :] = [1, 2, 3]
    bold[1, 1, 1, :] = [3, 4, 5]
    mask = np.zeros((2, 2, 2), bool)
    mask[0, 0, 0] = mask[1, 1, 1] = True
    tc = nv.roi_timecourse(bold, mask)
    assert tc.shape == (3,)
    assert np.allclose(tc, [2, 3, 4])  # per-timepoint mean of the two voxels


def test_brain_mask_thresholds_mean_intensity():
    bold = np.zeros((3, 3, 3, 4))
    bold[1, 1, 1, :] = 10.0
    m = nv.brain_mask(bold, frac=0.15)
    assert m[1, 1, 1] and m.sum() == 1


def test_regress_confounds_removes_linear_trend_keeps_signal():
    T = 200
    t = np.linspace(0, 1, T)
    conf = t[:, None]                       # a linear nuisance (e.g. drift)
    y = 3 * t + np.sin(2 * np.pi * 6 * t)   # trend + oscillation
    r = nv.regress_confounds(y, conf)
    assert abs(np.corrcoef(r, t)[0, 1]) < 0.1   # trend removed
    assert r.std() > 0.4                        # oscillation preserved


def test_scan_onset_sample_finds_first_matching_trigger():
    onsets = np.array([0.5, 1.0, 2.0, 3.0])
    values = np.array(["boundary", "R128", "R128", "QRS"])
    s = nv.scan_onset_sample(onsets, values, trigger="R128", sfreq=250.0)
    assert s == int(round(1.0 * 250))


def test_scan_onset_sample_raises_when_absent():
    with pytest.raises(ValueError):
        nv.scan_onset_sample(np.array([0.0]), np.array(["x"]), trigger="R128", sfreq=250.0)


def test_hrf_peaks_near_5s():
    h = nv.hrf(tr=1.0, length=32.0)
    assert 4 <= int(np.argmax(h)) <= 6          # canonical double-gamma peaks ~5 s
    assert h[-1] == pytest.approx(0, abs=0.05)  # returns to baseline


def test_bin_to_tr_averages_windows():
    sf, tr, n_tr = 10.0, 1.0, 3
    x = np.concatenate([np.full(10, 1.0), np.full(10, 2.0), np.full(10, 3.0)])
    b = nv.bin_to_tr(x, sf, tr, n_tr)
    assert np.allclose(b, [1, 2, 3])
