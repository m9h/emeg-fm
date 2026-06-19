#!/usr/bin/env python
"""Smoke test for LuMamba as a frozen feature extractor (de-risk before HBN).

Validates the whole chain in-container: import the cloned BioFoundation arch +
mamba-ssm CUDA kernels, instantiate the tiny config, load the HF safetensors
(checking state-dict key match), build 3D channel coords from the GSN-HydroCel-128
montage, and run LuMamba.encode() on a dummy (epochs, channels, time) batch ->
the (B, S, 384) encoder representation we mean-pool into a subject embedding.
"""
from __future__ import annotations

import numpy as np
import torch

# Tiny config (config/model/LuMamba_tiny.yaml): patch 40, 6 queries, embed 64.
CFG = dict(patch_size=40, num_queries=6, embed_dim=64, num_heads=2, mlp_ratio=4.,
           exp=2, num_blocks=2, bidirectional=True, bidirectional_strategy="add",
           num_classes=0)
CKPT = "LuMamba_LeJEPA_reconstruction_300slices.safetensors"


def _strip_prefix(sd):
    """Drop a common 'model.'/'module.'/'net.' prefix if the checkpoint has one."""
    for p in ("model.", "module.", "net.", "_orig_mod."):
        if all(k.startswith(p) for k in sd):
            return {k[len(p):]: v for k, v in sd.items()}
    return sd


def main():
    from models.LuMamba import LuMamba          # BioFoundation on PYTHONPATH
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import mne

    model = LuMamba(**CFG).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LuMamba tiny instantiated: {n_params/1e6:.2f}M params, "
          f"d_model={model.d_model} (=Q*E={CFG['num_queries']*CFG['embed_dim']})")

    path = hf_hub_download("PulpBio/LuMamba", CKPT)
    sd = _strip_prefix(load_file(path))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected "
          f"(of {len(sd)} ckpt tensors)")
    if missing:
        print("  e.g. missing:", missing[:5])
    if unexpected:
        print("  e.g. unexpected:", unexpected[:5])

    model = model.cuda()
    # Dummy: 2 epochs, 128 EGI channels, 10 s @ 256 Hz = 2560 (= 64 patches of 40).
    B, C, T = 2, 128, 2560
    x = torch.randn(B, C, T, device="cuda")
    pos = mne.channels.make_standard_montage("GSN-HydroCel-128").get_positions()["ch_pos"]
    coords = np.stack([pos[f"E{i}"] for i in range(1, C + 1)]).astype(np.float32)  # (C,3)
    cl = torch.tensor(coords, device="cuda").unsqueeze(0).repeat(B, 1, 1)           # (B,C,3)

    with torch.no_grad():
        z = model.encode(x, cl)            # (B, S, d_model=384)
    emb = z.mean(dim=1)                     # (B, 384) per-epoch embedding
    print(f"ENCODE_OK z={tuple(z.shape)}  pooled_emb={tuple(emb.shape)}  "
          f"finite={bool(torch.isfinite(emb).all())}")


if __name__ == "__main__":
    main()
