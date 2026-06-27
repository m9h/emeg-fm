#!/usr/bin/env python
"""Live (or replayed) per-subject EEG→image decoding session.

Ties the three pieces together:

    emeg_fm.streaming.LSLAcquisition   — EEG + Markers LSL → time-locked epochs
    emeg_fm.decoder.StreamingReveDecoder — frozen REVE + per-subject ridge
    emeg_fm.stimuli.ImageStimulusSet     — image gallery + code↔image mapping

Two phases in one run:

    1. CALIBRATE — collect labelled epochs for ``--calib-trials`` trials (or
       until the presentation block goes idle) and fit the ridge head. This is
       the "fine-tune to the new subject" step; it is closed-form and finishes
       in well under a second once the data is in. The 10-minute budget is the
       *presentation* of the calibration block, not the compute.
    2. ONLINE — for every subsequent epoch, predict the CLIP embedding and
       print the top-k retrieved images, tracking running accuracy.

Headless validation
--------------------
``--replay`` feeds a pre-epoched Alljoined subject through the identical Trial
interface, so the whole calibration→online path can be exercised on the GPU box
(inside the PyTorch SIF) with no headset before a real session::

    python scripts/run_streaming_decode.py --replay \
        --eeg-npy   .../sub-01/preprocessed_eeg_test_flat.npy \
        --stim-parquet .../sub-01/experiment_metadata_categories.parquet \
        --stimuli-dir /tmp/alljoined_stimuli \
        --max-images 64 --calib-frac 0.8 --out session.json

Live (after building a gallery with expy_image_experiment.py --build-gallery)::

    python scripts/run_streaming_decode.py --gallery session_gallery.npz \
        --calib-trials 240 --montage Fp1 Fp2 F3 F4 ... --out session.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _build_replay_gallery(args):
    """For replay: embed the Alljoined stimuli actually shown to this subject,
    keyed by the unique-image code the FileReplaySource assigns."""
    import os
    import pandas as pd
    from emeg_fm.alljoined import load_subject_npy, average_by_image
    from emeg_fm.stimuli import clip_image_embeddings

    rec = load_subject_npy(args.eeg_npy)
    n_epochs = rec["eeg"].shape[0]
    stim = pd.read_parquet(args.stim_parquet)
    stim = stim[stim["partition"] == args.partition]
    if "dropped" in stim.columns:
        stim = stim[~stim["dropped"].astype(bool)]
    stim = stim.reset_index(drop=True)
    if len(stim) != n_epochs:
        raise ValueError(f"{n_epochs} epochs vs {len(stim)} kept rows")
    basenames = [os.path.basename(p) for p in stim["image_path"].tolist()]
    # average_by_image sorts unique ids; FileReplaySource codes == that order.
    _avg, uniq, _counts = average_by_image(rec["eeg"], basenames)

    idx = {}
    for root, _d, files in os.walk(args.stimuli_dir):
        for f in files:
            idx[f] = os.path.join(root, f)
    codes, paths = [], []
    for code, bn in enumerate(uniq):
        if bn in idx:
            codes.append(int(code))
            paths.append(idx[bn])
    print(f"[replay] embedding {len(paths)} stimulus images with CLIP", flush=True)
    gallery = clip_image_embeddings(paths, args.clip_model, args.device)
    return gallery, np.array(codes, dtype=int)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # source
    ap.add_argument("--replay", action="store_true",
                    help="replay a pre-epoched Alljoined subject (no headset)")
    ap.add_argument("--eeg-npy"); ap.add_argument("--stim-parquet")
    ap.add_argument("--stimuli-dir"); ap.add_argument("--partition", default="stim_test")
    ap.add_argument("--gallery", help="cached gallery npz (live mode)")
    # epoching / montage
    ap.add_argument("--tmin", type=float, default=-0.2)
    ap.add_argument("--tmax", type=float, default=1.0)
    ap.add_argument("--montage", nargs="*", default=None,
                    help="explicit channel labels to subset/reorder to")
    ap.add_argument("--montage-preset", default=None,
                    help="named montage preset (e.g. 'epocflex32'); "
                         "overrides --montage")
    ap.add_argument("--skip-montage-check", action="store_true",
                    help="skip the headless standard_1020 label pre-flight")
    # Layer-3 device bridge (live only; replay data is already filtered)
    ap.add_argument("--bridge", action="store_true",
                    help="enable the DeviceBridge front-end (live consumer headset)")
    ap.add_argument("--mains", type=float, default=60.0,
                    help="mains notch freq (60 US / 50 EU); 0 disables")
    ap.add_argument("--highpass", type=float, default=0.5)
    ap.add_argument("--lowpass", type=float, default=99.5)
    ap.add_argument("--reref", default=None,
                    help="'average' for common-average reference, else None")
    # model
    ap.add_argument("--model", default="brain-bzh/reve-base")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--ridge-alpha", type=float, default=1000.0)
    ap.add_argument("--device", default=None)
    # calibration split
    ap.add_argument("--calib-trials", type=int, default=None,
                    help="live: number of trials to calibrate on")
    ap.add_argument("--calib-frac", type=float, default=0.8,
                    help="replay: fraction of trials used to calibrate")
    ap.add_argument("--idle-timeout", type=float, default=20.0)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--no-average", action="store_true",
                    help="don't trial-average repeats before fitting")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Resolve a named preset and pre-flight every label against MNE
    # standard_1020 before we touch a GPU or the gated REVE checkpoint. A
    # typo'd or legacy label (T3 for T7) fails here in milliseconds instead of
    # mid-session. Replay data carries its own montage, so only check when an
    # explicit montage/preset was given.
    if args.montage_preset:
        from emeg_fm.montage import get_preset
        args.montage = get_preset(args.montage_preset)
    if args.montage and not args.skip_montage_check:
        from emeg_fm.montage import validate_montage
        args.montage = validate_montage(args.montage)   # raises on unknown
        print(f"[montage] {len(args.montage)} labels resolve to standard_1020",
              flush=True)

    from emeg_fm.decoder import StreamingReveDecoder

    # --- gallery + source ---------------------------------------------------
    if args.replay:
        for req in ("eeg_npy", "stim_parquet", "stimuli_dir"):
            if getattr(args, req) is None:
                ap.error(f"--replay requires --{req.replace('_','-')}")
        from emeg_fm.streaming import FileReplaySource
        gallery, gallery_ids = _build_replay_gallery(args)
        src = FileReplaySource(args.eeg_npy, args.stim_parquet,
                               partition=args.partition, montage=args.montage,
                               max_images=args.max_images)
        src.tmin, src.tmax = args.tmin, args.tmax    # bookkeeping only for replay
    else:
        if not args.gallery:
            ap.error("live mode requires --gallery")
        from emeg_fm.stimuli import ImageStimulusSet
        from emeg_fm.streaming import LSLAcquisition
        ss = ImageStimulusSet.load_gallery(args.gallery)
        gallery, gallery_ids = ss.gallery, ss.gallery_ids
        src = LSLAcquisition(tmin=args.tmin, tmax=args.tmax, montage=args.montage)

    # Layer-3 bridge: live consumer-headset cleanup. Built lazily once the
    # source sfreq is known; replay keeps bridge=None (data already filtered).
    bridge = None
    if args.bridge:
        if args.replay:
            ap.error("--bridge is for live mode; replay data is already filtered")
        from emeg_fm.device import DeviceBridge
        sfreq = src.connect().sfreq            # resolves the EEG LSL stream
        bridge = DeviceBridge(
            sfreq, highpass=args.highpass, lowpass=args.lowpass,
            notch=(args.mains or None), reref=args.reref,
        )
        print(f"[bridge] {sfreq}Hz hp={args.highpass} lp={bridge.lowpass} "
              f"notch={bridge.notch} reref={args.reref}", flush=True)

    decoder = StreamingReveDecoder(
        gallery, gallery_ids, model_id=args.model, layer=args.layer,
        ridge_alpha=args.ridge_alpha, device=args.device, bridge=bridge,
    )

    # --- collect epochs -----------------------------------------------------
    if args.replay:
        gallery_codes = {int(c) for c in gallery_ids}
        all_trials = [t for t in src.stream_epochs(max_trials=None)
                      if t.code in gallery_codes]
        n_cal = max(2, int(round(len(all_trials) * args.calib_frac)))
        calib_trials, online_trials = all_trials[:n_cal], all_trials[n_cal:]
        print(f"[session] replay: {len(calib_trials)} calib / "
              f"{len(online_trials)} online trials", flush=True)
        report = decoder.fit_from_trials(calib_trials, average=not args.no_average)
        print(f"[calibrate] {report}", flush=True)
        results = decoder.evaluate(online_trials) if online_trials else {}
        if results:
            print(f"[online] top1={results['top1']:.3f} top5={results['top5']:.3f} "
                  f"top10={results['top10']:.3f} median_rank={results['median_rank']:.1f} "
                  f"chance_top1={results['chance_top1']:.4f}", flush=True)
        payload = {"mode": "replay", "calibration": report,
                   "n_online": len(online_trials),
                   "online": {k: v for k, v in results.items() if k != "ranks"}}
    else:
        with src:
            print(f"[session] collecting {args.calib_trials} calibration trials…",
                  flush=True)
            calib_trials = list(src.stream_epochs(max_trials=args.calib_trials,
                                                  idle_timeout=args.idle_timeout))
            report = decoder.fit_from_trials(calib_trials,
                                             average=not args.no_average)
            print(f"[calibrate] {report} — entering online decode", flush=True)

            hits1 = hits5 = total = 0
            for trial in src.stream_epochs(idle_timeout=args.idle_timeout):
                top = decoder.retrieve(trial, k=10)
                ids = [i for i, _ in top]
                total += 1
                hits1 += int(trial.code == ids[0])
                hits5 += int(trial.code in ids[:5])
                print(f"[online] shown={trial.code} top5={ids[:5]} "
                      f"acc@1={hits1/total:.2f} acc@5={hits5/total:.2f} "
                      f"(n={total})", flush=True)
            payload = {"mode": "live", "calibration": report,
                       "n_online": total,
                       "online": {"top1": hits1 / max(total, 1),
                                  "top5": hits5 / max(total, 1)}}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
