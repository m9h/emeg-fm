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


def test_occipital_mask_selects_posterior_by_world_y():
    mask = np.ones((2, 10, 2), bool)
    affine = np.eye(4)                       # world-Y == voxel-Y (anterior = high Y)
    occ = nv.occipital_mask(mask, affine, frac=0.3)
    ys = np.where(occ.any(axis=(0, 2)))[0]
    assert ys.min() == 0 and ys.max() <= 3   # posterior (low world-Y) kept
    assert not occ[:, 9, :].any()            # anterior dropped


def test_occipital_mask_is_orientation_robust():
    mask = np.ones((2, 10, 2), bool)
    affine = np.eye(4)
    affine[1, 1] = -1                        # voxel-Y up -> world-Y DOWN (flipped)
    occ = nv.occipital_mask(mask, affine, frac=0.3)
    ys = np.where(occ.any(axis=(0, 2)))[0]
    assert ys.max() == 9                     # posterior now = HIGH voxel-Y


def test_bin_by_triggers_averages_between_volume_triggers():
    # 3 volume triggers at samples 0/10/20; signal is constant within each window
    x = np.concatenate([np.full(10, 1.0), np.full(10, 2.0), np.full(10, 3.0)])
    trig = np.array([0, 10, 20])
    b = nv.bin_by_triggers(x, trig)
    assert len(b) == 3                         # one value per volume trigger
    assert np.allclose(b, [1, 2, 3])           # last window runs to the end


def test_bin_by_triggers_excludes_pre_scan_samples():
    # the natview case: EEG starts BEFORE the scan -> pre-first-trigger samples must drop
    x = np.concatenate([np.full(5, 99.0), np.full(10, 1.0), np.full(10, 2.0)])
    trig = np.array([5, 15])                   # first fMRI volume onset at sample 5
    b = nv.bin_by_triggers(x, trig)
    assert len(b) == 2
    assert np.allclose(b, [1, 2])              # the 99.0 pre-scan region is excluded


def test_select_occipital_channels():
    chs = ["Fp1", "Cz", "O1", "Oz", "O2", "PO7", "PO8", "T7"]
    idx = nv.select_occipital_channels(chs)
    assert sorted(chs[i] for i in idx) == ["O1", "O2", "Oz", "PO7", "PO8"]
