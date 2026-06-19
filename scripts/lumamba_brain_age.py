#!/usr/bin/env python
"""Frozen LuMamba (PulpBio Mamba/SSM EEG-FM) brain-age on the SAME HBN CV splits
as the classical / NEOBA / REVE baselines.

Per subject: resting epochs -> resample 256 Hz -> per-channel IQR normalize ->
crop to a multiple of patch_size (40) -> LuMamba.encode() (LUNA channel-unification
+ bidirectional Mamba) -> mean-pool over patches then epochs -> one 384-d
embedding. Ridge under KFold(10, shuffle, random_state=42) -> directly comparable
to REVE (1.67/0.607) and a 3rd member for the NEOBA(+)REVE fusion.

EGI channels use 3D GSN-HydroCel-128 coordinates (LUNA is topology-agnostic; no
vocabulary remap). Runs in the Docker NGC 26.05 container via run_lumamba.sh
(BioFoundation arch + built mamba-ssm on /mnt/t9).
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

CFG = dict(patch_size=40, num_queries=6, embed_dim=64, num_heads=2, mlp_ratio=4.,
           exp=2, num_blocks=2, bidirectional=True, bidirectional_strategy="add",
           num_classes=0)
CKPT = "LuMamba_LeJEPA_reconstruction_300slices.safetensors"
PATCH = 40
TARGET_SFREQ = 256.0


def _load_ages(participants):
    import pandas as pd
    df = pd.read_csv(participants, sep=None, engine="python")
    idc = "participant_id" if "participant_id" in df.columns else df.columns[0]
    ids = df[idc].astype(str).str.replace("sub-", "", regex=False)
    return dict(zip(ids, df["age"].astype(float)))


def _channel_coords(ch_names):
    """3D coords from GSN-HydroCel-128 for the channels MNE knows; returns the
    kept indices + (Ckeep, 3) array (LuMamba normalizes coords internally)."""
    import mne
    pos = mne.channels.make_standard_montage(
        "GSN-HydroCel-128").get_positions()["ch_pos"]
    keep, coords = [], []
    for i, nm in enumerate(ch_names):
        if nm in pos:
            keep.append(i)
            coords.append(pos[nm])
    return keep, np.asarray(coords, dtype=np.float32)


def _iqr_norm(data):
    """Per-channel IQR normalization over all of a subject's samples (LuMamba
    input contract). data (n_ep, C, T)."""
    flat = data.transpose(1, 0, 2).reshape(data.shape[1], -1)  # (C, n_ep*T)
    med = np.median(flat, axis=1)
    q75, q25 = np.percentile(flat, [75, 25], axis=1)
    iqr = np.where((q75 - q25) < 1e-8, 1.0, q75 - q25)
    return (data - med[None, :, None]) / iqr[None, :, None]


def _subject_embedding(model, data, sfreq_in, ch_names, batch_size, device):
    import scipy.signal as ss
    import torch
    if abs(sfreq_in - TARGET_SFREQ) > 1e-3:                      # resample -> 256
        n = int(round(data.shape[-1] * TARGET_SFREQ / sfreq_in))
        data = ss.resample(data, n, axis=-1)
    Tc = (data.shape[-1] // PATCH) * PATCH                       # multiple of 40
    data = data[..., :Tc]
    keep, coords = _channel_coords(ch_names)
    data = _iqr_norm(data[:, keep, :]).astype(np.float32)
    cl = torch.tensor(coords, device=device).unsqueeze(0)        # (1, C, 3)
    embs = []
    for i in range(0, data.shape[0], batch_size):
        xb = torch.tensor(data[i:i + batch_size], device=device)
        with torch.no_grad():
            z = model.encode(xb, cl.repeat(xb.shape[0], 1, 1))  # (B, S, 384)
        embs.append(z.mean(dim=1).float().cpu().numpy())
    return np.concatenate(embs, 0).mean(0)                       # (384,)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs-glob",
                   default="/data/derivatives/brain_age/HBN_EEG/sub-*/eeg/"
                           "*proc-autoreject_epo.fif")
    p.add_argument("--participants", default="/data/datasets/hbn-eeg/participants.tsv")
    p.add_argument("--subject-regex", default=r"sub-([A-Za-z0-9]+)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--out", default="/mnt/t9/lumamba_hbn_emb.npz")
    args = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import mne
    mne.set_log_level("error")
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from models.LuMamba import LuMamba

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    if out.exists():
        z = np.load(out)
        X, ages = z["X"], z["ages"]
        print(f"loaded cached embeddings {X.shape} from {out}")
    else:
        model = LuMamba(**CFG).eval()
        model.load_state_dict(load_file(hf_hub_download("PulpBio/LuMamba", CKPT)),
                              strict=False)
        model = model.to(device)
        print(f"LuMamba loaded ({sum(q.numel() for q in model.parameters())/1e6:.1f}M)"
              f", d=384, device={device}", flush=True)
        age = _load_ages(args.participants)
        pat = re.compile(args.subject_regex)
        files = sorted(Path("/").glob(args.epochs_glob.lstrip("/")))
        embs, ages, t0 = [], [], time.time()
        for f in files:
            m = pat.search(f.name)
            if not m:
                continue
            sid = m.group(1)
            if sid not in age or not np.isfinite(age[sid]):
                continue
            ep = mne.read_epochs(f, preload=True, verbose="error")
            emb = _subject_embedding(model, ep.get_data(copy=False),
                                     ep.info["sfreq"], ep.ch_names,
                                     args.batch_size, device)
            embs.append(emb)
            ages.append(age[sid])
            del ep
            if len(ages) % 25 == 0:
                print(f"  {len(ages)} subj ({time.time()-t0:.0f}s)", flush=True)
            if args.max_subjects and len(ages) >= args.max_subjects:
                break
        X = np.vstack(embs).astype(np.float32)
        ages = np.asarray(ages)
        np.savez(out, X=X, ages=ages)
        print(f"\ncached embeddings -> {out}", flush=True)

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    print(f"\nLuMamba brain-age: {X.shape[0]} subjects, d={X.shape[1]}")
    print(f"dummy-mean MAE = {np.mean(np.abs(ages - ages.mean())):.2f} yr")
    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 25)))
    sc = cross_validate(reg, X, ages, cv=cv,
                        scoring=("neg_mean_absolute_error", "r2"))
    mae = -sc["test_neg_mean_absolute_error"]
    r2 = sc["test_r2"]
    print(f"LuMamba+RidgeCV  MAE = {mae.mean():.2f} +/- {mae.std():.2f} yr   "
          f"R^2 = {r2.mean():.3f} +/- {r2.std():.3f}")


if __name__ == "__main__":
    main()
