#!/usr/bin/env python
"""Single-subject Alljoined EEG→image retrieval smoke with frozen REVE.

Pipeline
--------
    preprocessed_eeg/sub-XX/*_flat.npy  (pickled dict: (n,32,250)@250Hz + ch_names)
      → join stim_order.parquet (image_path per trial, filter partition)
      → average trials per image (SNR)         [emeg_fm.alljoined.average_by_image]
      → resample 250→200 Hz, z-score, clamp ±15 [...preprocess_for_reve]
      → REVE forward (frozen) → mean-pool tokens → (n_img, d_model)
    stimuli → CLIP image embeddings → (n_img, d_clip)
      → ridge: REVE feat → CLIP embed  → cosine top-k retrieval on held-out images
      → raw-EEG (PCA) baseline for comparison

The point: does a frozen EEG foundation model decode seen images above chance
on a cheap 32-ch consumer rig, and does it beat naive raw-EEG features?

Runs inside the PyTorch NGC SIF (REVE + CLIP are torch). No jax needed — the
``emeg_fm`` package __init__ is import-light and ``emeg_fm.adapters`` imports
jax only inside ``torch_to_jax`` (never on the REVE path), so the submodules
import cleanly in a jax-less SIF without any namespace/jax shims.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Stimulus / image handling
# ---------------------------------------------------------------------------

def index_stimuli(stimuli_dir: str) -> dict:
    """Map image basename → local path for every image under ``stimuli_dir``.

    Alljoined's parquet stores server-absolute paths
    (``/srv/.../images/16641.jpg``); we resolve by basename against the
    locally unzipped stimuli tree.
    """
    idx = {}
    for root, _dirs, files in os.walk(stimuli_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                idx[f] = os.path.join(root, f)
    return idx


def embed_images_clip(local_paths, model_id, device, batch_size=64):
    """CLIP image embeddings for a list of local image paths → (n, d_clip)."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)

    feats = []
    for i in range(0, len(local_paths), batch_size):
        chunk = local_paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        inp = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            # Compute the projected image embedding explicitly rather than via
            # get_image_features: in this transformers version the latter
            # returns a BaseModelOutputWithPooling, not the tensor. vision_model
            # → visual_projection is the canonical CLIP image-embed path and is
            # stable across versions (gives the 512-d projected space).
            vis = model.vision_model(pixel_values=inp["pixel_values"])
            emb = model.visual_projection(vis.pooler_output)
        feats.append(emb.float().cpu().numpy())
        print(f"  [clip] {min(i + batch_size, len(local_paths))}/{len(local_paths)}",
              flush=True)
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# REVE feature extraction
# ---------------------------------------------------------------------------

def reve_features(eeg, ch_names, layer, model_id, batch_size=32):
    """Frozen REVE forward over (n_img, C, T) → mean-pooled (n_img, d_model)."""
    from emeg_fm.eeg_fm import REVEAdapter, REVE_BASE_ID  # noqa: F401

    adapter = REVEAdapter(layer=layer)
    loaded = adapter.load_model(model_id)
    print(f"  [reve] loaded {model_id} layer={layer} d_model={adapter.output_dim}",
          flush=True)

    pooled = []
    n = eeg.shape[0]
    t0 = time.time()
    for i in range(0, n, batch_size):
        chunk = eeg[i:i + batch_size]
        feats = adapter.extract_features(
            loaded,
            {"eeg": chunk, "electrode_names": ch_names, "ch_names": ch_names},
        )
        if feats.ndim == 3:                 # (B, P, D) → mean over tokens
            feats = feats.mean(axis=1)
        pooled.append(np.asarray(feats, dtype=np.float32))
        print(f"  [reve] {min(i + batch_size, n)}/{n}  "
              f"({(min(i + batch_size, n)) / (time.time() - t0 + 1e-9):.1f} img/s)",
              flush=True)
    return np.concatenate(pooled, axis=0)


# ---------------------------------------------------------------------------
# Decoding eval
# ---------------------------------------------------------------------------

def ridge_retrieval(X, Y, seed=0, test_frac=0.2, ks=(1, 5, 10)):
    """Standardize X, ridge-regress X→Y, cosine top-k retrieval on held-out
    images (gallery = the held-out true CLIP embeddings)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    from emeg_fm.alljoined import topk_retrieval

    idx = np.arange(X.shape[0])
    tr, te = train_test_split(idx, test_size=test_frac, random_state=seed)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

    reg = Ridge(alpha=1000.0).fit(Xtr, Y[tr])
    pred = reg.predict(Xte)
    out = topk_retrieval(pred, Y[te], ks=ks)
    out["n_test"] = int(len(te))
    out["n_train"] = int(len(tr))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eeg-npy", required=True,
                    help="preprocessed_eeg/sub-XX/preprocessed_eeg_test_flat.npy")
    ap.add_argument("--stim-parquet", required=True,
                    help="preprocessed_eeg/sub-XX/experiment_metadata_categories.parquet "
                         "(carries the per-trial `dropped` flag; stim_order.parquet does not)")
    ap.add_argument("--stimuli-dir", required=True,
                    help="directory of unzipped stimulus images")
    ap.add_argument("--partition", default="stim_test",
                    help="stim_order partition matching the npy (stim_test/stim_train)")
    ap.add_argument("--model", default="brain-bzh/reve-base")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--sfreq-out", type=float, default=200.0)
    ap.add_argument("--reve-batch", type=int, default=32)
    ap.add_argument("--clip-batch", type=int, default=64)
    ap.add_argument("--max-images", type=int, default=None,
                    help="cap unique images for a fast smoke")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from emeg_fm.alljoined import (
        load_subject_npy, average_by_image, preprocess_for_reve,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}", flush=True)

    # 1. Load epochs + align image_path per trial via the partition filter.
    rec = load_subject_npy(args.eeg_npy)
    eeg, ch_names, sfreq = rec["eeg"], rec["ch_names"], rec["sfreq"]
    print(f"[load] eeg={eeg.shape} sfreq={sfreq} n_ch={len(ch_names)}", flush=True)
    print(f"[load] ch_names={ch_names}", flush=True)

    stim = pd.read_parquet(args.stim_parquet)
    stim = stim[stim["partition"] == args.partition]
    # The flat npy contains only kept epochs; the metadata parquet marks the
    # ones rejected during preprocessing. Drop them so the remaining rows align
    # 1:1, in order, with the epochs (e.g. sub-01 stim_test: 16224 → 16217).
    if "dropped" in stim.columns:
        stim = stim[~stim["dropped"].astype(bool)]
    stim = stim.reset_index(drop=True)
    if len(stim) != eeg.shape[0]:
        raise ValueError(
            f"trial/stim misalignment: {eeg.shape[0]} epochs vs {len(stim)} "
            f"kept '{args.partition}' rows. Pass experiment_metadata_categories"
            f".parquet (it has the `dropped` flag); stim_order.parquet lacks it "
            f"and will be off by the number of rejected epochs."
        )
    image_files = [os.path.basename(p) for p in stim["image_path"].tolist()]

    # 2. Average repeated trials per image (SNR).
    avg, uniq_files, counts = average_by_image(eeg, image_files)
    print(f"[avg] {avg.shape[0]} unique images "
          f"(mean {counts.mean():.1f} trials/img)", flush=True)
    if args.max_images is not None and avg.shape[0] > args.max_images:
        sel = np.argsort(-counts)[:args.max_images]   # keep best-sampled images
        avg, uniq_files, counts = avg[sel], uniq_files[sel], counts[sel]
        print(f"[avg] capped to {avg.shape[0]} images", flush=True)

    # 3. Resolve images to local files (drop any missing).
    idx = index_stimuli(args.stimuli_dir)
    keep, local_paths = [], []
    for i, f in enumerate(uniq_files):
        if f in idx:
            keep.append(i)
            local_paths.append(idx[f])
    keep = np.asarray(keep)
    if len(keep) < avg.shape[0]:
        print(f"[stim] {avg.shape[0] - len(keep)} images missing locally, "
              f"keeping {len(keep)}", flush=True)
    avg, uniq_files, counts = avg[keep], uniq_files[keep], counts[keep]

    # 4. REVE-ready preprocessing.
    proc = preprocess_for_reve(avg, sfreq_in=sfreq, sfreq_out=args.sfreq_out)
    print(f"[prep] {proc.shape} (resampled to {args.sfreq_out} Hz)", flush=True)

    # 5. REVE features + CLIP targets.
    Xreve = reve_features(proc, ch_names, args.layer, args.model,
                          batch_size=args.reve_batch)
    Yclip = embed_images_clip(local_paths, args.clip_model, device,
                              batch_size=args.clip_batch)
    print(f"[feat] REVE={Xreve.shape}  CLIP={Yclip.shape}", flush=True)

    # 6. Decoding: REVE vs. a raw-EEG (PCA) baseline.
    from sklearn.decomposition import PCA
    Xraw = proc.reshape(proc.shape[0], -1)
    Xraw = PCA(n_components=min(512, Xraw.shape[0] - 1, Xraw.shape[1]),
               random_state=0).fit_transform(Xraw)

    reve_res = ridge_retrieval(Xreve, Yclip)
    raw_res = ridge_retrieval(Xraw, Yclip)

    print("\n=== RESULTS (held-out image retrieval) ===", flush=True)
    print(f"  gallery size (n_test) = {reve_res['n_test']}  "
          f"chance top1 = {reve_res['chance_top1']:.4f}", flush=True)
    for name, r in [("REVE", reve_res), ("raw-EEG(PCA)", raw_res)]:
        print(f"  {name:13s} top1={r['top1']:.3f} top5={r['top5']:.3f} "
              f"top10={r['top10']:.3f} median_rank={r['median_rank']:.1f}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model, "layer": args.layer, "clip_model": args.clip_model,
        "subject_npy": args.eeg_npy, "partition": args.partition,
        "n_unique_images": int(avg.shape[0]),
        "mean_trials_per_image": float(counts.mean()),
        "ch_names": ch_names, "sfreq_in": sfreq, "sfreq_out": args.sfreq_out,
        "reve": {k: v for k, v in reve_res.items() if k != "ranks"},
        "raw_pca": {k: v for k, v in raw_res.items() if k != "ranks"},
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
