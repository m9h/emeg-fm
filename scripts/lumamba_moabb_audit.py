#!/usr/bin/env python
"""FMScope identity-trap audit of LuMamba on a MOABB cohort (default BNCI2014_001).

Does a Mamba/SSM EEG-FM encode subject identity / show the identity trap the way
the transformer FMs (REVE) do? Builds a RAW MOABB cohort (no REVE normalization),
extracts frozen LuMamba.encode() embeddings, and runs subject_axis_erasure both
the paper-method (pooled subject*class) and per-trial (n>>p) ways — exactly the
machinery behind the REVE BNCI2014_001 result (pooled 0.667->0.963, per-trial
0.536). Reports subject-probe BA (identity encoding) + label raw/erased (trap).

Runs in Docker NGC via run_lumamba.sh (BioFoundation arch + built mamba-ssm).
"""
from __future__ import annotations

import argparse

import numpy as np

CFG = dict(patch_size=40, num_queries=6, embed_dim=64, num_heads=2, mlp_ratio=4.,
           exp=2, num_blocks=2, bidirectional=True, bidirectional_strategy="add",
           num_classes=0)
CKPT = "LuMamba_LeJEPA_reconstruction_300slices.safetensors"
PATCH = 40
TARGET_SFREQ = 256.0


def _iqr_norm(data):
    flat = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    med = np.median(flat, axis=1)
    q75, q25 = np.percentile(flat, [75, 25], axis=1)
    iqr = np.where((q75 - q25) < 1e-8, 1.0, q75 - q25)
    return (data - med[None, :, None]) / iqr[None, :, None]


class LuMambaExtractor:
    """Callable (B,C,T) raw torch tensor @ sfreq_in -> (B, 384) np embeddings."""

    def __init__(self, ch_names, sfreq_in=200.0, device="cuda"):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from models.LuMamba import LuMamba
        from emeg_fm.montage import channel_coords3d
        self.device, self.sfreq_in = device, float(sfreq_in)
        self.model = LuMamba(**CFG).eval()
        self.model.load_state_dict(load_file(hf_hub_download("PulpBio/LuMamba", CKPT)),
                                   strict=False)
        self.model = self.model.to(device)
        self.keep, coords = channel_coords3d(list(ch_names))
        if not self.keep:
            raise SystemExit("no channels resolved to 3D coords for LuMamba")
        self.cl = torch.tensor(coords, device=device).unsqueeze(0)  # (1, Ckeep, 3)

    def __call__(self, x):
        import scipy.signal as ss
        import torch
        data = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        if abs(self.sfreq_in - TARGET_SFREQ) > 1e-3:
            n = int(round(data.shape[-1] * TARGET_SFREQ / self.sfreq_in))
            data = ss.resample(data, n, axis=-1)
        Tc = (data.shape[-1] // PATCH) * PATCH
        data = _iqr_norm(data[:, self.keep, :Tc]).astype(np.float32)
        xb = torch.tensor(data, device=self.device)
        with torch.no_grad():
            z = self.model.encode(xb, self.cl.repeat(xb.shape[0], 1, 1))
        return z.mean(dim=1).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from emeg_fm.moabb_cohort import build_moabb_cohort
    from fmscope.verdict.audit import _extract_features
    from fmscope.diagnostics.erasure import subject_axis_erasure
    import moabb.datasets as mds
    from moabb.paradigms import LeftRightImagery

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = getattr(mds, args.dataset)()
    paradigm = LeftRightImagery(fmin=0.5, fmax=99.5, resample=200.0)
    coh = build_moabb_cohort(dataset=dataset, paradigm=paradigm, normalize=False)
    print(f"[cohort] {args.dataset}: {coh.ch_names and len(coh.ch_names)} ch, device={device}",
          flush=True)
    ext = LuMambaExtractor(ch_names=coh.ch_names, sfreq_in=200.0, device=device)

    feats, sids, labels, _ = _extract_features(ext, coh, batch_size=args.batch_size,
                                               device=device)
    n = len(labels)
    print(f"[feats] {feats.shape} from {np.unique(sids).size} subjects", flush=True)

    pooled = subject_axis_erasure(feats, sids, labels, cv="stratified-kfold")
    per_trial = subject_axis_erasure(
        feats, sids, labels, window_recording=np.arange(n),
        rec_labels=labels, rec_pids=sids, cv="stratified-kfold")

    print(f"\n=== LuMamba identity-trap audit: {args.dataset} ===")
    print(f"REVE reference        : pooled 0.667->0.963 (Δ+0.296) | per-trial 0.536 (Δ+0.004)")
    for tag, er in (("pooled (subj×class)", pooled), ("per-trial (n>>p)", per_trial)):
        print(f"LuMamba {tag:<20}: raw={er.label_ba_raw:.3f} erased={er.label_ba_erased:.3f} "
              f"Δ={er.label_ba_delta:+.3f} | subj_BA {er.subj_ba_linear_pre:.2f}->"
              f"{er.subj_ba_linear_post:.2f} (chance {er.chance:.3f}) "
              f"interp={er.interpretable} degenerate={er.degenerate}")


if __name__ == "__main__":
    main()
