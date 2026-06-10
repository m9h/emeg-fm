"""Tests for emeg_fm.decoder.StreamingReveDecoder.

REVE is bypassed via an injected ``feature_fn`` (flatten the z-scored epoch),
so the calibration→retrieval path runs without torch or the foundation model.
Each stimulus code gets a distinct temporal template; with low-noise repeats the
ridge head should map features → the correct CLIP-gallery row and retrieval
should recover the shown image.
"""
import numpy as np
import pytest

from emeg_fm.decoder import StreamingReveDecoder
from emeg_fm.streaming import Trial


N_CODES, C, T, D = 6, 4, 40, 8
SFREQ = 40.0   # == decoder sfreq_out → preprocess_for_reve won't resample


def _flatten_features(proc, ch_names):
    return proc.reshape(proc.shape[0], -1)


def _make_world(seed=0):
    rng = np.random.default_rng(seed)
    gallery = rng.standard_normal((N_CODES, D))
    codes = np.arange(N_CODES)                       # gallery_ids
    # distinct fixed waveform template per code (survives per-channel z-score)
    templates = rng.standard_normal((N_CODES, C, T)) * 3.0
    return gallery, codes, templates, rng


def _trial(code, epoch, t=0.0):
    return Trial(code=int(code), epoch=epoch.astype(np.float32),
                 ch_names=[f"c{i}" for i in range(C)], sfreq=SFREQ, onset_ts=t)


def _calib_trials(templates, rng, repeats=10, noise=0.05):
    trials = []
    for code in range(N_CODES):
        for _ in range(repeats):
            ep = templates[code] + rng.standard_normal((C, T)) * noise
            trials.append(_trial(code, ep))
    rng.shuffle(trials)
    return trials


def _decoder(gallery, codes):
    return StreamingReveDecoder(gallery, codes, sfreq_out=SFREQ,
                                ridge_alpha=1.0, feature_fn=_flatten_features)


def test_fit_from_trials_report():
    gallery, codes, templates, rng = _make_world()
    dec = _decoder(gallery, codes)
    report = dec.fit_from_trials(_calib_trials(templates, rng), average=False)
    assert report["n_images"] == N_CODES
    assert report["d_feat"] == C * T
    assert dec.is_fitted


def test_retrieve_recovers_shown_image():
    gallery, codes, templates, rng = _make_world()
    dec = _decoder(gallery, codes)
    dec.fit_from_trials(_calib_trials(templates, rng), average=False)
    # clean (noise-free) probe for each code → should top-1 retrieve it
    hits = 0
    for code in range(N_CODES):
        top = dec.retrieve(_trial(code, templates[code]), k=1)
        hits += int(top[0][0] == code)
    assert hits >= N_CODES - 1            # allow at most one miss


def test_evaluate_beats_chance_on_heldout():
    gallery, codes, templates, rng = _make_world()
    dec = _decoder(gallery, codes)
    dec.fit_from_trials(_calib_trials(templates, rng, repeats=12), average=False)
    held = [_trial(c, templates[c] + rng.standard_normal((C, T)) * 0.05)
            for c in range(N_CODES) for _ in range(3)]
    res = dec.evaluate(held, ks=(1, 5))
    assert res["top1"] > res["chance_top1"]
    assert res["top1"] >= 0.7


def test_average_repeats_path():
    gallery, codes, templates, rng = _make_world()
    dec = _decoder(gallery, codes)
    report = dec.fit_from_trials(_calib_trials(templates, rng, repeats=10),
                                 average=True)
    assert report["averaged"] is True
    assert report["n_images"] == N_CODES
    assert report["mean_trials_per_image"] == pytest.approx(10.0)


def test_retrieve_before_fit_raises():
    gallery, codes, templates, _ = _make_world()
    dec = _decoder(gallery, codes)
    with pytest.raises(RuntimeError, match="not calibrated"):
        dec.retrieve(_trial(0, templates[0]))


def test_codes_outside_gallery_filtered():
    gallery, codes, templates, rng = _make_world()
    dec = _decoder(gallery, codes)
    trials = _calib_trials(templates, rng, repeats=4)
    # inject trials with an unknown code 999 — should be ignored, not crash
    trials += [_trial(999, templates[0]) for _ in range(3)]
    report = dec.fit_from_trials(trials, average=True)
    assert report["n_images"] == N_CODES    # 999 dropped
