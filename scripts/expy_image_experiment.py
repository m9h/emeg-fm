#!/usr/bin/env python
"""Image-RSVP stimulus presentation for NeuroTechX EEG-ExPy.

Presents a closed set of images to a subject and pushes an LSL **Markers**
outlet at each stimulus onset (marker value = the image's integer code from
:class:`emeg_fm.stimuli.ImageStimulusSet`). This is the EEG-ExPy contract: the
device backend (EmotivPRO-LSL / CyKit for a 32-ch Emotiv wet cap, or EEG-ExPy's
own muse/openbci/brainflow push) supplies the **EEG** LSL stream; this script
supplies the **Markers** stream. The companion consumer is
``scripts/run_streaming_decode.py`` (via ``emeg_fm.streaming.LSLAcquisition``).

Run it standalone, or register the ``present`` body inside an EEG-ExPy
``BaseExperiment`` — the only coupling to EEG-ExPy is the LSL Markers outlet,
which is the same convention ``eegnb`` uses.

PsychoPy + pylsl are imported lazily so ``--build-gallery`` (the offline prep
step) runs on a box without a display or LSL.

Usage
-----
    # 0. one-time: discover images + cache the CLIP gallery (GPU box / SIF)
    python scripts/expy_image_experiment.py --build-gallery \
        --image-dir /path/to/images --gallery-out session_gallery.npz \
        --max-images 64

    # 1. on the presentation laptop (display + headset streaming to LSL):
    python scripts/expy_image_experiment.py \
        --gallery session_gallery.npz --n-repeats 5 \
        --stim-s 1.0 --isi-s 0.5
"""
from __future__ import annotations

import argparse
import sys

from emeg_fm.stimuli import ImageStimulusSet


def build_gallery(args):
    ss = ImageStimulusSet.from_dir(args.image_dir, max_images=args.max_images,
                                   seed=args.seed)
    print(f"[gallery] {len(ss.paths)} images; embedding with {args.clip_model}",
          flush=True)
    ss.compute_gallery(args.clip_model, device=args.device)
    ss.save_gallery(args.gallery_out)
    print(f"[gallery] wrote {args.gallery_out} "
          f"({ss.gallery.shape[0]}×{ss.gallery.shape[1]})", flush=True)


def present(args):
    import pylsl
    from psychopy import visual, core, event

    ss = ImageStimulusSet.load_gallery(args.gallery)
    schedule = ss.build_schedule(n_repeats=args.n_repeats, seed=args.seed)
    print(f"[present] {len(ss.paths)} images × {args.n_repeats} = "
          f"{len(schedule)} trials; ~{len(schedule)*(args.stim_s+args.isi_s)/60:.1f} min",
          flush=True)

    info = pylsl.StreamInfo("Markers", "Markers", 1, 0, "int32",
                            "emeg-fm-rsvp")
    outlet = pylsl.StreamOutlet(info)

    win = visual.Window(fullscr=args.fullscreen, color=(0, 0, 0), units="norm")
    fix = visual.TextStim(win, text="+", height=0.15, color=(1, 1, 1))
    msg = visual.TextStim(win, height=0.07, color=(1, 1, 1),
                          text="Fixate the cross. Press any key to begin.")
    msg.draw(); win.flip(); event.waitKeys()

    clock = core.Clock()
    try:
        for n, (code, path) in enumerate(schedule, 1):
            if event.getKeys(keyList=["escape"]):
                print("[present] aborted by escape", flush=True)
                break
            fix.draw(); win.flip(); core.wait(args.isi_s)
            img = visual.ImageStim(win, image=path)
            img.draw()
            win.callOnFlip(outlet.push_sample, [int(code)])   # marker at onset
            win.flip()
            core.wait(args.stim_s)
            if n % 25 == 0:
                print(f"[present] {n}/{len(schedule)}  t={clock.getTime():.0f}s",
                      flush=True)
    finally:
        win.close()
    print("[present] done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-gallery", action="store_true",
                    help="offline prep: discover images + cache CLIP gallery")
    ap.add_argument("--image-dir")
    ap.add_argument("--gallery-out", default="session_gallery.npz")
    ap.add_argument("--gallery", help="cached gallery npz for presentation")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=64)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--stim-s", type=float, default=1.0)
    ap.add_argument("--isi-s", type=float, default=0.5)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.build_gallery:
        if not args.image_dir:
            ap.error("--build-gallery requires --image-dir")
        build_gallery(args)
    else:
        if not args.gallery:
            ap.error("presentation requires --gallery (run --build-gallery first)")
        present(args)


if __name__ == "__main__":
    sys.exit(main())
