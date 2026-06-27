"""Merge logic for the cross-model WeightWatcher summary table.

Regression cover for the stale-`reve`-row bug: a model that fails to load (e.g.
a transient gated-HF 401 or a user-site triton shadowing the SIF) returns a row
with ``status != "ok"`` and no metric fields. The merged summary must never let
such a row overwrite a previously good row, or it silently wipes real numbers
from ``all_models_summary.json``.
"""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_eegfm_weightwatcher.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("ww_eegfm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_mod()


def _by_model(rows):
    return {r["model"]: r for r in rows}


def test_failed_row_does_not_overwrite_ok_row():
    existing = [{"model": "reve", "status": "ok", "alpha_mean": 3.61, "n_layers": 89}]
    new = [{"model": "reve", "status": "load_failed", "error": "triton AttrsDescriptor"}]
    out = _by_model(mod._merge_summaries(existing, new, order=["reve"]))
    assert out["reve"]["status"] == "ok"
    assert out["reve"]["alpha_mean"] == 3.61


def test_ok_row_repairs_stale_failed_row():
    existing = [{"model": "reve", "status": "load_failed", "error": "x"}]
    new = [{"model": "reve", "status": "ok", "alpha_mean": 3.61}]
    out = _by_model(mod._merge_summaries(existing, new, order=["reve"]))
    assert out["reve"]["status"] == "ok"
    assert out["reve"]["alpha_mean"] == 3.61


def test_fresh_ok_overwrites_previous_ok():
    existing = [{"model": "luna", "status": "ok", "alpha_mean": 3.9}]
    new = [{"model": "luna", "status": "ok", "alpha_mean": 4.0}]
    out = _by_model(mod._merge_summaries(existing, new, order=["luna"]))
    assert out["luna"]["alpha_mean"] == 4.0


def test_first_time_failure_is_recorded():
    out = _by_model(mod._merge_summaries([], [{"model": "neurorvq", "status": "load_failed"}],
                                         order=["neurorvq"]))
    assert out["neurorvq"]["status"] == "load_failed"


def test_known_models_ordered_first_then_extras():
    existing = [{"model": "zztop", "status": "ok", "alpha_mean": 1.0},
                {"model": "reve", "status": "ok", "alpha_mean": 3.61}]
    out = mod._merge_summaries(existing, [], order=["reve", "lumamba"])
    assert out[0]["model"] == "reve"
    assert {r["model"] for r in out} == {"reve", "zztop"}
