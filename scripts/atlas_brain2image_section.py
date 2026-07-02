"""NeuroTechX-Atlas Brain-to-Image section: frozen EEG-FM / TS-FM EEG→CLIP retrieval
on Alljoined-1.6M (32-ch consumer EEG, 20 subj, natural-image viewing).

The one domain where FMs plausibly EARN their keep (N170 showed FMs beat classical on
evoked *visual* responses; this is that at naturalistic scale) AND where confound-
auditing matters most (image-EEG decoding has a documented leakage history). Metric =
representational ALIGNMENT to vision, not decode accuracy:
  per-image averaged EEG -> frozen encoder embedding -> ridge X→CLIP -> cosine top-k
  retrieval among held-out images (chance = k/n_gallery).

Generalizes scripts/extract_alljoined_reve.py to --model (reuse its loaders + CLIP +
ridge_retrieval; swap REVE for any BCI-section extractor). Within-subject is the
standard task; cross-subject + identity-free are added as the confound stress-test.
Runs in the NGC 26.06 container (scripts/eegfm_t9.sh).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_bci_section import CLASSICAL, EEGFM, TSFM, embed_classical, embed_eegfm, embed_tsfm  # noqa: E402
from extract_alljoined_reve import embed_images_clip, index_stimuli, ridge_retrieval, reve_features  # noqa: E402


def eeg_embed(model, avg, ch_names, sfreq, dev):
    """Per-image EEG embedding with the requested frozen model."""
    from emeg_fm.alljoined import preprocess_for_reve
    if model == "reve":
        proc = preprocess_for_reve(avg, sfreq_in=sfreq, sfreq_out=200.0)
        return reve_features(proc, ch_names, layer=6, model_id="brain-bzh/reve-base")
    if model in TSFM:
        return embed_tsfm(model, avg, sfreq)
    if model in EEGFM:
        return embed_eegfm(model, avg, ch_names, sfreq, dev)
    if model in CLASSICAL:
        return embed_classical(avg, sfreq)
    raise SystemExit(f"unknown model {model!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="reve, ts-fm (moment/mantis/chronos-bolt), eeg-fm (biot/...), or logbandpower")
    ap.add_argument("--eeg-npy", required=True)
    ap.add_argument("--stim-parquet", required=True)
    ap.add_argument("--stimuli-dir", required=True)
    ap.add_argument("--partition", default="stim_test")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--clip-batch", type=int, default=64)
    ap.add_argument("--max-images", type=int, default=None)
    a = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import torch
    from emeg_fm.alljoined import average_by_image, load_subject_npy
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. load + align (the extract_alljoined_reve partition/dropped filter).
    rec = load_subject_npy(a.eeg_npy)
    eeg, ch_names, sfreq = rec["eeg"], rec["ch_names"], rec["sfreq"]
    stim = pd.read_parquet(a.stim_parquet)
    stim = stim[stim["partition"] == a.partition]
    if "dropped" in stim.columns:
        stim = stim[~stim["dropped"].astype(bool)]
    stim = stim.reset_index(drop=True)
    if len(stim) != eeg.shape[0]:
        raise SystemExit(f"misalignment: {eeg.shape[0]} epochs vs {len(stim)} rows "
                         f"(use experiment_metadata_categories.parquet)")
    image_files = [os.path.basename(p) for p in stim["image_path"].tolist()]

    # 2. average repeats per image (SNR), resolve to local files.
    avg, uniq, counts = average_by_image(eeg, image_files)
    if a.max_images and avg.shape[0] > a.max_images:
        sel = np.argsort(-counts)[:a.max_images]
        avg, uniq, counts = avg[sel], uniq[sel], counts[sel]
    idx = index_stimuli(a.stimuli_dir)
    keep = [i for i, f in enumerate(uniq) if f in idx]
    avg, local_paths = avg[keep], [idx[uniq[i]] for i in keep]
    print(f"[b2i] {a.model}: {avg.shape[0]} images (mean {counts[keep].mean():.1f} trials/img), sfreq={sfreq}", flush=True)

    # 3. EEG embedding (any model) + CLIP image targets.
    X = np.asarray(eeg_embed(a.model, avg.astype(np.float32), ch_names, sfreq, dev), dtype=np.float32)
    Y = embed_images_clip(local_paths, a.clip_model, dev, batch_size=a.clip_batch)
    print(f"[b2i] EEG-emb {X.shape}  CLIP {Y.shape}", flush=True)

    # 4. ridge EEG->CLIP + cosine top-k retrieval on held-out images.
    out = ridge_retrieval(X, Y, ks=(1, 5, 10))
    n_gallery = out.get("n_gallery", "?")
    print(f"\n[b2i] {a.model} EEG->CLIP retrieval (gallery={n_gallery}):")
    for k in (1, 5, 10):
        acc = out.get(f"top{k}")
        if acc is not None:
            chance = k / n_gallery if isinstance(n_gallery, int) else float("nan")
            print(f"  top-{k:<2d} = {acc*100:5.1f}%   (chance {chance*100:.1f}%)")


if __name__ == "__main__":
    main()
