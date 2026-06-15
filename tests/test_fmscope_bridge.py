"""Tests for the FMScope-bridge depth-probe / 1-f-role diagnostics.

These cover the pure-numpy / sklearn scoring helpers and the REVE-native
``reve_layer_probe`` / ``fooof_role`` orchestration with a *fake* extractor —
no torch, no REVE weights, no FOOOF (the real ``fooof_ablate`` is monkeypatched
to an identity so we exercise the orchestration without the SIF-only dep). The
driver-side summary + rubric helpers (``_summarize_layer_probe``,
``_layer_sign``, ``_oneoverf_role``) are loaded straight from the leaderboard
script and checked against hand-built dicts.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest

# fmscope is vendored, not pip-installed; put it on the path for the
# fooof_ablation import inside fooof_role.
_FMSCOPE = str(pathlib.Path(__file__).resolve().parents[1] / "fmscope")
if _FMSCOPE not in sys.path:
    sys.path.insert(0, _FMSCOPE)

from emeg_fm.fmscope_bridge import (  # noqa: E402
    _label_probe_ba,
    _subject_probe_ba,
    fooof_role,
    reve_layer_probe,
)


# Load the leaderboard script as a module (stdlib-only at import time).
def _load_driver():
    path = (pathlib.Path(__file__).resolve().parents[1]
            / "scripts" / "moabb_identity_leaderboard.py")
    spec = importlib.util.spec_from_file_location("moabb_lb", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mlb = _load_driver()


# --------------------------------------------------------------------------- #
# Probe scoring helpers
# --------------------------------------------------------------------------- #
def test_label_probe_ba_separates_linear_signal():
    rng = np.random.default_rng(0)
    n = 60
    labels = np.array([0, 1] * (n // 2))
    sids = np.repeat(np.arange(6), n // 6)
    feats = rng.normal(0, 0.01, (n, 4)).astype(np.float32)
    feats[:, 0] += labels * 5.0
    ba = _label_probe_ba(feats, labels, sids, n_folds=3, seed=0)
    assert ba > 0.9


def test_subject_probe_ba_reidentifies_across_recordings():
    rng = np.random.default_rng(1)
    sids, rec, feats = [], [], []
    rid = 0
    for s in range(4):
        for _r in range(2):                    # two recordings per subject
            for _w in range(5):
                sids.append(s)
                rec.append(rid)
                feats.append([s * 5.0, 0.0, 0.0, 0.0])
            rid += 1
    feats = np.asarray(feats, np.float32)
    feats[:, 0] += rng.normal(0, 0.05, 40)     # noise only on the signal dim
    ba = _subject_probe_ba(feats, np.asarray(sids), np.asarray(rec),
                           n_folds=3, seed=0)
    assert ba > 0.8


def test_subject_probe_ba_nan_when_single_recording_per_subject():
    sids = np.repeat(np.arange(4), 3)
    rec = np.repeat(np.arange(4), 3)            # rec == sid → 1 recording each
    feats = np.random.default_rng(0).normal(size=(12, 4)).astype(np.float32)
    assert math.isnan(
        _subject_probe_ba(feats, sids, rec, n_folds=3, seed=0))


# --------------------------------------------------------------------------- #
# reve_layer_probe with a fake all-layers extractor
# --------------------------------------------------------------------------- #
class _FakeCohort:
    def __init__(self, recordings, sfreq=200.0):
        self._recs = recordings
        self.sfreq = sfreq

    def iter_recordings(self):
        yield from self._recs


class _FakeLayerExtractor:
    """Shallow block encodes the label (channel 0), deep block the subject
    (channel 1) — so the probe should localize each axis to a distinct depth.

    Deterministic (no RNG state): a pure function of the input, so the FOOOF
    identity-ablation test gets bit-identical orig/ablated features. Only the
    signal dims carry values; the rest stay 0 (StandardScaler maps constant
    columns to zeros, so they add no competing noise to the probes)."""

    def __init__(self, n_blocks=3, d=4):
        self.n_blocks = n_blocks
        self.d = d

    def all_layer_feats(self, x):
        x = np.asarray(x, np.float32)
        b = x.shape[0]
        label_sig = x[:, 0, :].mean(axis=1)
        subj_sig = x[:, 1, :].mean(axis=1)
        f = np.zeros((self.n_blocks, b, self.d), np.float32)
        f[0, :, 0] = label_sig * 3.0           # shallow → label
        f[-1, :, 1] = subj_sig * 3.0           # deep → subject
        return f

    def __call__(self, x):                      # final-block pooled (B, D)
        return self.all_layer_feats(x)[-1]


def _toy_cohort(seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for s in range(4):
        for lab in (0, 1):                      # each subject: one rec per label
            w = np.zeros((4, 2, 4), np.float32)
            w[:, 0, :] = lab + rng.normal(0, 0.01, (4, 4))
            w[:, 1, :] = s + rng.normal(0, 0.01, (4, 4))
            recs.append((s, lab, w))
    return _FakeCohort(recs)


def test_reve_layer_probe_shapes_and_depth_localization():
    ext = _FakeLayerExtractor(n_blocks=3, d=4)
    cohort = _toy_cohort()
    out = reve_layer_probe(ext, cohort, batch_size=4, n_folds=3, seed=0)

    assert out["n_layers"] == 3
    pd = out["per_depth"]
    assert len(pd) == 3
    for d, entry in enumerate(pd):
        assert entry["depth"] == d
        assert abs(entry["depth_fraction"] - (d + 1) / 3) < 1e-9
        assert math.isfinite(entry["label_ba_mean"])

    # Label info lives shallow; subject info lives deep.
    assert pd[0]["label_ba_mean"] > 0.8
    assert pd[-1]["subject_ba_mean"] > 0.7
    assert pd[-1]["subject_ba_mean"] > pd[0]["subject_ba_mean"]


def test_reve_layer_probe_empty_cohort_raises():
    with pytest.raises(ValueError):
        reve_layer_probe(_FakeLayerExtractor(), _FakeCohort([]), batch_size=4)


# --------------------------------------------------------------------------- #
# fooof_role (identity ablation → zero drop)
# --------------------------------------------------------------------------- #
def test_fooof_role_identity_ablation_zero_drop(monkeypatch):
    import fmscope.preprocess.fooof_ablation as fa

    monkeypatch.setattr(
        fa, "fooof_ablate",
        lambda epochs, sfreq, **kw: np.asarray(epochs, dtype=np.float32))

    ext = _FakeLayerExtractor(n_blocks=3, d=4)
    cohort = _toy_cohort()
    res = fooof_role(ext, cohort, sfreq=200.0, batch_size=4, n_folds=3, seed=0)

    # Ablation is identity → original and ablated features are bit-identical,
    # so both BA drops are exactly zero.
    assert res["state_drop_mean"] == 0.0
    assert res["subject_drop_mean"] == 0.0
    assert res["mode"] == "aperiodic_removed"
    assert math.isfinite(res["orig_label_ba"])


# --------------------------------------------------------------------------- #
# Driver summary + rubric helpers
# --------------------------------------------------------------------------- #
def _probe(label_bas, subj_bas):
    n = len(label_bas)
    return {"per_depth": [
        {"depth": i, "depth_fraction": (i + 1) / n,
         "label_ba_mean": label_bas[i], "subject_ba_mean": subj_bas[i]}
        for i in range(n)]}


def test_summarize_layer_probe_keys_and_argmax():
    s = mlb._summarize_layer_probe(_probe([0.9, 0.7, 0.6], [0.5, 0.6, 0.8]))
    assert s["label_ba_first"] == 0.9
    assert s["label_ba_last"] == 0.6
    assert s["label_ba_max"] == 0.9
    assert abs(s["argmax_depth"] - 1 / 3) < 1e-9


def test_summarize_layer_probe_none_on_empty_or_all_nan():
    assert mlb._summarize_layer_probe({"per_depth": []}) is None
    nan = float("nan")
    assert mlb._summarize_layer_probe(_probe([nan, nan], [nan, nan])) is None


def test_layer_sign_early_and_deep():
    # peaks shallow, big drop to last, last still > 0.45 → +early only.
    early = {"label_ba_first": 0.9, "label_ba_last": 0.6,
             "label_ba_max": 0.9, "argmax_depth": 1 / 3}
    assert mlb._layer_sign(early) == "+early"

    # final block collapses ≤0.45 → -deep present.
    deep = {"label_ba_first": 0.5, "label_ba_last": 0.42,
            "label_ba_max": 0.55, "argmax_depth": 0.66}
    assert "-deep" in mlb._layer_sign(deep)

    assert mlb._layer_sign(None) == ""


def test_oneoverf_role_thresholds():
    both = mlb._oneoverf_role({"state_drop_mean": 0.05, "subject_drop_mean": 0.06})
    assert "subject confound" in both and "state signal" in both
    assert mlb._oneoverf_role({"state_drop_mean": 0.01,
                               "subject_drop_mean": 0.01}) == ""
    assert mlb._oneoverf_role(None) == ""
