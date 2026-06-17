#!/usr/bin/env python
"""Recompute the ``verdict`` column of an existing leaderboard CSV and re-render
its MD — no FM re-extraction.

Use after a sweep wrote rows under stale verdict logic (e.g. the pre-fix
variance-only TRAP rule that ignored interpretability), or to re-label a legacy
pooled CSV. The numeric columns (raw / identity-free BA, Δ, fractions) are
untouched; only the derived ``verdict`` string and the rendered table change.
Pure-Python (CPU) — reuses the driver's own ``_verdict`` / ``_read_rows`` /
``_write_csv`` / ``_render_md`` so the taxonomy stays single-sourced.
"""
from __future__ import annotations

import argparse
import importlib.util
import os


def _load_driver():
    p = os.path.join(os.path.dirname(__file__), "moabb_identity_leaderboard.py")
    spec = importlib.util.spec_from_file_location("_mil", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="leaderboard_*.csv to re-derive in place")
    ap.add_argument("--paradigm-name", default="LeftRightImagery",
                    help="title for the re-rendered MD")
    args = ap.parse_args()

    mil = _load_driver()
    rows = mil._read_rows(args.csv)
    n_changed = 0
    for r in rows:
        if r.get("status") != "ok":
            continue
        old = r.get("verdict")
        r["verdict"] = mil._verdict(
            _f(r.get("raw_label_ba")),
            _f(r.get("identity_free_label_ba")),
            r.get("erasure_interpretable"),
        )
        if r["verdict"] != old:
            n_changed += 1
    mil._write_csv(args.csv, rows)
    md_path = os.path.splitext(args.csv)[0] + ".md"
    mil._render_md(args.csv, md_path, args.paradigm_name)
    print(f"[rederive] {len(rows)} rows, {n_changed} verdict(s) changed "
          f"-> {args.csv} + {md_path}")


if __name__ == "__main__":
    main()
