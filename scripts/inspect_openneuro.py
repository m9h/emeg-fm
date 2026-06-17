#!/usr/bin/env python
"""Stage + inspect OpenNeuro BIDS structure for the FMScope from-raw repro.

ds002893 (auditory-visual oddball, target vs standard) and ds004148 (eyes-closed
vs mental arithmetic) have dataset-specific task names + event codings that must
be read before the contrast loaders can be written. This downloads a minimal
slice (dataset_description, participants, sub-01's EEG sidecars + events) to
/mnt/t9 and prints: available tasks, events.tsv columns + unique trial types,
channel names, and sampling rate — enough to write correct loaders.
"""
from __future__ import annotations

import glob
import json
import os
import sys

TARGET = os.environ.get("ONEURO_DIR", "/mnt/t9/openneuro")
DATASETS = ["ds002893", "ds004148"]


def _download(ds):
    import csv
    import openneuro
    dest = os.path.join(TARGET, ds)
    os.makedirs(dest, exist_ok=True)

    # openneuro-py `include` entries are bare path prefixes (NOT ** globs).
    def _get(inc):
        try:
            openneuro.download(dataset=ds, target_dir=dest, include=[inc])
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {inc}: {type(e).__name__})", flush=True)
            return False

    for inc in ("dataset_description.json", "participants.tsv"):
        _get(inc)
    # Discover the real first-subject label (zero-padding varies by dataset).
    subj = None
    pt = os.path.join(dest, "participants.tsv")
    if os.path.exists(pt):
        rows = list(csv.DictReader(open(pt), delimiter="\t"))
        if rows and rows[0].get("participant_id"):
            subj = rows[0]["participant_id"]
    for cand in [c for c in (subj, "sub-01", "sub-001") if c]:
        if _get(cand):
            print(f"  downloaded {cand}", flush=True)
            break
    return dest


def _inspect(dest, ds):
    print(f"\n===== {ds} @ {dest} =====", flush=True)
    dd = os.path.join(dest, "dataset_description.json")
    if os.path.exists(dd):
        print("Name:", json.load(open(dd)).get("Name"))
    # tasks
    tasks = sorted({os.path.basename(p).split("task-")[1].split("_")[0]
                    for p in glob.glob(f"{dest}/**/*task-*", recursive=True)
                    if "task-" in os.path.basename(p)})
    print("tasks:", tasks)
    # events
    for ev in sorted(glob.glob(f"{dest}/sub-01/**/*_events.tsv", recursive=True)):
        import csv
        with open(ev) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        cols = list(rows[0].keys()) if rows else []
        print(f"\n  {os.path.relpath(ev, dest)}  ({len(rows)} events)")
        print("   cols:", cols)
        for key in ("trial_type", "value", "stim_type", "stimulus"):
            if key in cols:
                vals = {}
                for r in rows:
                    vals[r[key]] = vals.get(r[key], 0) + 1
                print(f"   {key} counts:", dict(sorted(vals.items(),
                      key=lambda kv: -kv[1])[:12]))
    # channels + sfreq from one eeg json
    for ej in sorted(glob.glob(f"{dest}/sub-01/**/*_eeg.json", recursive=True))[:1]:
        j = json.load(open(ej))
        print("\n  eeg.json:", os.path.relpath(ej, dest))
        print("   sfreq:", j.get("SamplingFrequency"),
              "| EEGref:", j.get("EEGReference"),
              "| n_chan:", j.get("EEGChannelCount"),
              "| PowerLineFreq:", j.get("PowerLineFrequency"))
    for ch in sorted(glob.glob(f"{dest}/sub-01/**/*_channels.tsv", recursive=True))[:1]:
        import csv
        with open(ch) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        names = [r.get("name") for r in rows]
        print("   channels:", names)
    # raw file formats present
    fmts = sorted({os.path.splitext(p)[1] for p in
                   glob.glob(f"{dest}/sub-01/**/*", recursive=True)
                   if os.path.splitext(p)[1] in (".edf", ".bdf", ".vhdr", ".set", ".fif")})
    print("  raw formats:", fmts)


def main():
    os.makedirs(TARGET, exist_ok=True)
    for ds in DATASETS:
        try:
            dest = _download(ds)
            _inspect(dest, ds)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {ds}: {type(exc).__name__}: {exc}", flush=True)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
