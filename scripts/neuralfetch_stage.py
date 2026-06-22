#!/usr/bin/env python
"""Stage EEG/MEG/fMRI datasets via Meta's neuralfetch (neuroai stack).

neuralfetch registers ~132 curated studies that download to neuralset-ready
tensors + BIDS, with a per-study ``query`` selector. This wraps it for our infra:
list the catalog, show which dataset-staging roadmap items it covers vs which
still need a hand-rolled pull, and stage one study to /mnt/t9 (never /data NFS).

neuralfetch is imported lazily so this module loads (and ``roadmap_coverage`` is
testable) without it; the staging path needs it installed (see the sbatch, which
installs it to a --target dir inside the Docker NGC container -- never pip --user).

Usage:
  python neuralfetch_stage.py --coverage              # roadmap vs live catalog
  python neuralfetch_stage.py --list                  # full catalog
  python neuralfetch_stage.py --study Obeid2016Tueg \
        --query '{"target":"age"}' --out /mnt/t9/tueg  # download + run one study
"""
from __future__ import annotations

import argparse
import json
import os
import re

# Our dataset-staging roadmap -> (regex over catalog names, the task it bears on).
# Drives --coverage so we never assume a gated/absent set is handled by neuralfetch.
ROADMAP = {
    "TUEG":      (r"Obeid2016Tueg|(?<![A-Za-z])Tueg", "#43/#72 TUEG brain-age"),
    "TUAB":      (r"Lopez2017Tuab|Tuab", "#43 TUH abnormal"),
    "TUH-other": (r"Tuar|Tuev|Tusz", "#43 TUH artifact/event/seizure"),
    "HBN":       (r"Shirazi2024Hbn|Hbn", "brain-age + SAE (ds005516)"),
    "NSD":       (r"Allen2022Massive", "fMRI MindEye/NSD"),
    "Sleep-EDF": (r"Kemp2000", "#47-adjacent (PhysioNet sleep)"),
    "eegmat":    (r"Zyma2019", "FMScope mental-arithmetic repro"),
    "depression":(r"Mumtaz2018", "TDBRAIN MDD-vs-ADHD adjacent (#60)"),
    "CHBP":      (r"chbp|cuban", "#42 (Synapse) -- hand-roll if absent"),
    "NSRR":      (r"nsrr|shhs|mesa|mros|wsc", "#47 (sleepdata.org) -- hand-roll if absent"),
    "Cam-CAN":   (r"camcan|cam.?can", "#44 (gated) -- hand-roll if absent"),
    "TDBRAIN":   (r"tdbrain|brain.?dt", "#57/#60 -- hand-roll if absent"),
    "LEMON":     (r"lemon", "#41 (already staged)"),
}


def roadmap_coverage(catalog_names):
    """Classify each roadmap dataset as covered/not by the given catalog names.
    Pure: ``catalog_names`` is a list of study names (no neuralfetch import)."""
    out = {}
    for label, (pat, task) in ROADMAP.items():
        rx = re.compile(pat, re.I)
        matches = [n for n in catalog_names if rx.search(n)]
        out[label] = {"covered": bool(matches), "matches": matches, "task": task}
    return out


def _catalog():
    """Live neuralfetch catalog name->class (registers on neuralfetch import)."""
    import neuralfetch  # noqa: F401  (import registers the 132 curated studies)
    import neuralset as ns
    cat = ns.Study.catalog()
    return cat() if callable(cat) else cat


def _print_coverage(names):
    cov = roadmap_coverage(names)
    print(f"[catalog] {len(names)} studies registered\n")
    print(f"{'roadmap':12s} {'covered':8s} match / task")
    for label, rec in cov.items():
        mark = "YES" if rec["covered"] else "no"
        m = ",".join(rec["matches"][:3]) if rec["matches"] else "-"
        print(f"{label:12s} {mark:8s} {m}  [{rec['task']}]")


def _stage(study, query, out):
    cat = _catalog()
    if study not in cat:
        raise SystemExit(f"[stage] '{study}' not in catalog ({len(cat)} studies). "
                         f"Run --list. It may need a hand-rolled pull.")
    os.makedirs(out, exist_ok=True)
    kwargs = {"path": out}
    if query:
        kwargs["query"] = json.loads(query)
    print(f"[stage] {study} -> {out}  query={kwargs.get('query')}", flush=True)
    st = cat[study](**kwargs)
    st.download()                       # gated sources still need creds in-env
    events = st.run()                   # neuralset events / tensors
    n = getattr(events, "shape", ["?"])[0] if hasattr(events, "shape") else "?"
    print(f"[stage] done: {study} -> {out} (events rows={n})", flush=True)
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage", action="store_true",
                    help="print roadmap datasets vs the live neuralfetch catalog")
    ap.add_argument("--list", action="store_true", help="print the full catalog")
    ap.add_argument("--study", help="catalog study name to stage")
    ap.add_argument("--query", default=None, help="JSON dict passed to the study (e.g. subject/target filter)")
    ap.add_argument("--out", default="/mnt/t9/neurofetch", help="output dir (NOT /data NFS)")
    args = ap.parse_args()

    if args.coverage:
        _print_coverage(sorted(_catalog().keys()))
    elif args.list:
        for n in sorted(_catalog().keys()):
            print(n)
    elif args.study:
        _stage(args.study, args.query, args.out)
    else:
        ap.error("one of --coverage / --list / --study is required")


if __name__ == "__main__":
    main()
