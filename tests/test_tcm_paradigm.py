"""Tests for the multi-paradigm parameterization of the Wang TCM per-trial audit.

The TCM sweep started life hardwired to ``LeftRightImagery`` (motor imagery). To
get the full cross-paradigm identity-trap picture it must also drive the P300
(ERP) and SSVEP MOABB paradigms — under the SAME broadband contract the FM audit
uses (0.5–99.5 Hz, 200 Hz resample), with the per-Nyquist fmax overrides and the
binary ``n_classes=2`` restriction SSVEP needs for the binary LEACE erasure.

These tests stub ``moabb.paradigms`` (moabb is container-only here) so the
paradigm-wiring logic is exercised without the heavy dependency: the registry
selects the right class, threads the right band/resample, and applies the
Nyquist fmax overrides.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts",
                       "moabb_tcm_pertrial.py")


def _load():
    spec = importlib.util.spec_from_file_location("_tcmpt", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def stub_moabb(monkeypatch):
    """Install a fake ``moabb.paradigms`` recording constructor kwargs."""
    calls = {}

    def _mk(name):
        class _P:
            def __init__(self, **kw):
                self.kw = kw
                self.__class__.__name__ = name
                calls[name] = kw
        _P.__name__ = name
        return _P

    mod = types.ModuleType("moabb.paradigms")
    mod.LeftRightImagery = _mk("LeftRightImagery")
    mod.P300 = _mk("P300")
    mod.SSVEP = _mk("SSVEP")
    pkg = types.ModuleType("moabb")
    pkg.paradigms = mod
    monkeypatch.setitem(sys.modules, "moabb", pkg)
    monkeypatch.setitem(sys.modules, "moabb.paradigms", mod)
    return calls


def test_registry_has_three_paradigms():
    m = _load()
    assert set(m.PARADIGMS) == {"leftright", "erp", "ssvep"}
    assert m.PARADIGMS["leftright"]["display"] == "LeftRightImagery"
    assert m.PARADIGMS["erp"]["display"] == "P300"
    assert m.PARADIGMS["ssvep"]["display"] == "SSVEP"


def test_build_leftright(stub_moabb):
    m = _load()
    p = m._build_paradigm("SomeMI", "leftright")
    assert p.__class__.__name__ == "LeftRightImagery"
    assert p.kw == {"fmin": 0.5, "fmax": 99.5, "resample": 200.0}


def test_build_erp_is_p300_broadband(stub_moabb):
    m = _load()
    p = m._build_paradigm("BNCI2014-009", "erp")
    assert p.__class__.__name__ == "P300"
    assert p.kw == {"fmin": 0.5, "fmax": 99.5, "resample": 200.0}


def test_build_ssvep_is_binary(stub_moabb):
    m = _load()
    p = m._build_paradigm("SSVEPExo", "ssvep")
    assert p.__class__.__name__ == "SSVEP"
    # SSVEP is natively multi-class; the binary LEACE erasure needs n_classes=2.
    assert p.kw["n_classes"] == 2
    assert p.kw["fmin"] == 0.5 and p.kw["resample"] == 200.0


def test_nyquist_fmax_override(stub_moabb):
    m = _load()
    # 128 Hz cohorts (Nyquist 64) must cap fmax below Nyquist or MNE rejects it.
    p = m._build_paradigm("MAMEM3", "ssvep")
    assert p.kw["fmax"] == 60.0
    p2 = m._build_paradigm("PhysionetMotorImagery", "leftright")
    assert p2.kw["fmax"] == 60.0


def test_parse_nyquist_from_mne_error():
    m = _load()
    msg = "h_freq ([99.5]) must be less than the Nyquist frequency 64.0"
    assert m._nyquist_from_error(msg) == 64.0
    # ValueError instance, not just a string.
    assert m._nyquist_from_error(ValueError(msg)) == 64.0
    # Unrelated error -> None (so the caller re-raises rather than masking it).
    assert m._nyquist_from_error("some other failure") is None


def test_clamp_fmax_below_nyquist():
    m = _load()
    # Clears a 64 Hz Nyquist with margin, matching the static 60 Hz override.
    assert m._clamp_fmax_below(64.0) == 60.0
    # Never goes non-positive on a tiny Nyquist.
    assert m._clamp_fmax_below(2.0) >= 1.0
    # Strictly below the Nyquist it was handed.
    assert m._clamp_fmax_below(128.0) < 128.0


def test_batched_tcm_loadings_match_loop():
    # Best-practice adopted from smni-cmi moabb_shootout.tcm_feat: project all
    # trials in two tensordots instead of a Python loop. Must be numerically
    # identical to the per-trial tcm_loading it replaces.
    m = _load()
    rng = np.random.RandomState(0)
    N, C, T, Ks, Kt = 7, 5, 11, 3, 4
    X = rng.randn(N, C, T)
    B = rng.randn(C, Ks)         # spatial basis (C, Ks)
    Cb = rng.randn(Kt, T)        # temporal basis (Kt, T)
    loop = np.stack([m.tcm_loading(X[i], B, Cb) for i in range(N)])
    batch = m._tcm_loadings_batch(X, B, Cb)
    assert batch.shape == (N, Ks * Kt)
    np.testing.assert_allclose(batch, loop, rtol=1e-10, atol=1e-12)
