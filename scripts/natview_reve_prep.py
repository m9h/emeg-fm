"""Phase 1 (host / CPU) of the REVE-vs-band-power EEG->BOLD comparison on natview.

Decouples the MNE/FSL work (host, .venv-models) from the REVE/torch GPU work (NGC
container). Per pair, caches everything the comparison needs so mcflirt is never re-run:
  y          occipital BOLD (mcflirt-cleaned, standardized), (n_tr,)
  bp         occipital-channel 5-band power, trigger-binned (NOT yet HRF-conv), (n_tr, 5)
  reve_win   per-TR EEG windows resampled to 200 Hz, (n_trig, n_ch, 420) for REVE
  ch_names   the montage channel names (for REVE electrode lookup)
Trigger-aligned via R128. Output: /mnt/t9/natview_reve/prep_{sub}_{ses}.npz
Usage: natview_reve_prep.py [task]
"""
import glob
import os
import sys
import tempfile

import mne
import numpy as np

from emeg_fm import natview as nv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natview_eeg_to_bold import (  # noqa: E402
    NV, bandpower_per_tr, clean_bold, mcflirt, volume_trigger_samples,
)

mne.set_log_level("ERROR")
TASK = sys.argv[1] if len(sys.argv) > 1 else "inscapes"
OUT = "/mnt/t9/natview_reve"
SF_REVE = 200.0        # REVE input contract
W = 420                # 2.1 s TR at 200 Hz (>= 1 patch of 200)


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = []
    for e in sorted(glob.glob(f"{NV}/preproc_data/sub-*/ses-*/eeg/*task-{TASK}*_eeg.set")):
        p = e.split("/")
        sub = next(x for x in p if x.startswith("sub-"))
        ses = next(x for x in p if x.startswith("ses-"))
        b = sorted(glob.glob(f"{NV}/raw_data/{sub}/{ses}/func/{sub}_{ses}_task-{TASK}*_bold.nii.gz"))
        if b:
            pairs.append((sub, ses, e, b[0]))
    print(f"[prep] {len(pairs)} {TASK} pairs -> {OUT}", flush=True)

    for i, (sub, ses, eeg_path, bold_path) in enumerate(pairs):
        out = f"{OUT}/prep_{sub}_{ses}.npz"
        if os.path.exists(out):
            print(f"  [{i + 1}/{len(pairs)}] {sub} {ses}  cached", flush=True)
            continue
        try:
            with tempfile.TemporaryDirectory(dir="/mnt/t9") as wd:
                d, par, affine = mcflirt(bold_path, wd)
            y = clean_bold(d, nv.occipital_mask(nv.brain_mask(d), affine), par)
            n_tr = d.shape[3]
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
            occ_ch = nv.select_occipital_channels(raw.ch_names)
            if not occ_ch:
                print(f"  [{i + 1}/{len(pairs)}] {sub} {ses}  SKIP (no occ ch)", flush=True)
                continue
            trig0 = volume_trigger_samples(eeg_path, raw.info["sfreq"])
            bp = bandpower_per_tr(raw, trig0, n_tr, occ_ch)          # occipital band-power
            # REVE per-TR windows: resample whole recording to 200 Hz, window from each trigger
            raw200 = raw.copy().resample(SF_REVE)
            X = raw200.get_data()
            trig2 = volume_trigger_samples(eeg_path, SF_REVE)
            wins = []
            for t in trig2:
                seg = X[:, t:t + W]
                if seg.shape[1] < W:
                    seg = np.pad(seg, ((0, 0), (0, W - seg.shape[1])))
                wins.append(seg.astype(np.float32))
            reve_win = np.stack(wins)                                # (n_trig, n_ch, 420)
            np.savez_compressed(out, y=y.astype(np.float32), bp=bp.astype(np.float32),
                                reve_win=reve_win, ch_names=np.array(raw.ch_names),
                                n_tr=n_tr)
            print(f"  [{i + 1}/{len(pairs)}] {sub} {ses}  y[{len(y)}] bp{bp.shape} "
                  f"reve_win{reve_win.shape}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"  [{i + 1}/{len(pairs)}] {sub} {ses}  ERR {type(ex).__name__}: {ex}",
                  flush=True)
    print("[prep] done", flush=True)


if __name__ == "__main__":
    main()
