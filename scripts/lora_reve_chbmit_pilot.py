"""Pilot: does LoRA fine-tuning of REVE change the identity-free Delta,
compared to a frozen linear probe, on CHB-MIT seizure detection?

Motivated by OpenEEGBench (github.com/braindecode/OpenEEGBench) -- a 12-dataset
EEG-FM PEFT (parameter-efficient fine-tuning) benchmark covering 8 strategies
(LoRA/DoRA/OFT/IA3/full-FT/frozen probes) with NO identity-trap axis at all.
Our program's differentiator is the identity-free (LEACE subject-erasure) Delta;
OpenEEGBench's is a richer fine-tuning-method axis. This pilot crosses the two:
does LoRA fine-tuning trap harder or softer on subject identity than our
existing frozen-REVE-embedding + linear-probe pipeline (docs/chbmit_epilepsy_
results.md: normal AUC 0.014, identity-free AUC 0.093, Delta=-0.079 -- REVE
already collapses near-floor here on CHB-MIT's bipolar montage via the
anchor-electrode approximation)?

Design: LoRA-adapt REVE's attention Q/K/V ("to_qkv") and output ("to_out")
projections in blocks {0,1,2,3,10} -- the exact blocks flagged as
under-trained (alpha<2, HT-SR/WeightWatcher analysis, docs/reference_reve_
weightwatcher_alphas.md) -- plus a linear classification head on the
mean-pooled layer-6 feature (same layer/pooling as embed_reve). Fine-tune
LoRA params + head on a BOUNDED train subset (all seizure-containing
recordings + <=1 non-seizure recording per train patient, to keep runtime
tractable for a pilot -- this is explicitly disclosed, not a full-corpus
retrain). After fine-tuning, freeze everything and extract the LoRA-adapted
model's pooled embeddings for train/dev/eval, then run the SAME downstream
pipeline as the frozen-REVE case (StandardScaler + optional LEACE eraser +
LogisticRegression + SzCORE evaluate()) -- isolating the fine-tuning
variable while keeping the identity-free comparison apples-to-apples.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_epilepsy_chbmit as cb  # noqa: E402
import atlas_sleep_section as sl  # noqa: E402  (reuses nothing here, see _fit_probe below)

LORA_BLOCKS = [0, 1, 2, 3, 10]
LORA_RANK = 8
LORA_ALPHA = 16.0


def _make_lora_linear_cls():
    """nn.Module subclass built lazily so importing this module doesn't
    require torch (tests exercise _lora_delta directly, no GPU needed)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRALinear(nn.Module):
        """Wraps an existing nn.Linear with a frozen base + trainable
        low-rank delta: y = base(x) + scaling * (x @ A^T) @ B^T. Must be an
        nn.Module (not a plain wrapper) to be assignable as a submodule in
        place of the original Linear -- nn.Module.__setattr__ rejects a
        non-Module value for an attribute name that already holds one."""

        def __init__(self, base_linear, rank=LORA_RANK, alpha=LORA_ALPHA):
            super().__init__()
            self.base = base_linear
            for p in self.base.parameters():
                p.requires_grad_(False)
            d_in, d_out = base_linear.in_features, base_linear.out_features
            self.A = nn.Parameter(torch.randn(rank, d_in) * (1.0 / rank ** 0.5))
            self.B = nn.Parameter(torch.zeros(d_out, rank))
            self.scaling = alpha / rank

        def forward(self, x):
            base_out = self.base(x)
            delta = F.linear(F.linear(x, self.A), self.B) * self.scaling
            return base_out + delta

    return LoRALinear


def _lora_delta(x, A, B, scaling):
    """Pure-numpy LoRA delta, mirrors LoRALinear.__call__'s math exactly --
    used by tests to verify the low-rank update without touching torch."""
    return (x @ A.T @ B.T) * scaling


def iter_bounded_recordings(subjects, max_nonseizure_per_patient=1):
    """Like cb.iter_recordings but bounded per patient: ALL seizure-containing
    recordings + at most `max_nonseizure_per_patient` non-seizure recordings,
    for the given (already patient-id-merged) subject set. Keeps a real,
    class-imbalanced-but-tractable pilot dataset instead of the full corpus."""
    by_patient = {}
    for edf, events, subj in cb.iter_recordings():
        pid = cb._patient_id(subj)
        if pid not in subjects:
            continue
        by_patient.setdefault(pid, {"seiz": [], "clean": []})
        (by_patient[pid]["seiz"] if events else by_patient[pid]["clean"]).append((edf, events, subj))
    out = []
    for pid, d in by_patient.items():
        out.extend(d["seiz"])
        out.extend(d["clean"][:max_nonseizure_per_patient])
    return out


def _patch_lora(model, blocks):
    """Replace to_qkv/to_out in the given block indices with LoRA-wrapped
    versions. Returns the flat list of trainable LoRA parameters."""
    LoRALinear = _make_lora_linear_cls()
    backbone = model.transformer
    trainable = []
    for k in blocks:
        attn = backbone.layers[k][0]
        for name in ("to_qkv", "to_out"):
            base = getattr(attn, name)
            wrapped = LoRALinear(base).to(base.weight.device)
            setattr(attn, name, wrapped)
            trainable.extend([wrapped.A, wrapped.B])
    return trainable


def _pooled_forward(model, pos_bank, eeg, electrode_names, device, layer=6):
    """Differentiable forward -- same input construction as REVEAdapter.
    _forward_out_layers (channel filtering, patch padding, bf16 autocast) but
    WITHOUT torch.no_grad(), so LoRA params get gradients."""
    import torch
    known = getattr(pos_bank, "mapping", None)
    if known is not None:
        keep = [i for i, nm in enumerate(electrode_names) if nm in known]
        eeg = eeg[:, keep, :]
        electrode_names = [electrode_names[i] for i in keep]
    patch_size = int(getattr(getattr(model, "config", None), "patch_size", 200))
    if eeg.shape[-1] < patch_size:
        eeg = torch.nn.functional.pad(eeg, (0, patch_size - eeg.shape[-1]))
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )
    with autocast_ctx:
        with torch.no_grad():
            positions = pos_bank(electrode_names).expand(eeg.size(0), -1, -1)
        out_layers = model(eeg, positions, return_output=True)
        feat = out_layers[layer + 1].float().mean(1)  # mean-pool over time
    return feat


def _prep_batch(X, device):
    import torch
    mu = X.mean(-1, keepdims=True)
    Xn = np.clip((X - mu) / (X.std(-1, keepdims=True) + 1e-8), -15, 15).astype(np.float32)
    return torch.from_numpy(Xn).to(device)


def finetune_and_embed(train_recs, splits_X, ch_names, device, epochs=1, lr=1e-3, batch=8):
    """train_recs: list of (X, y) window arrays for the bounded train subset.
    splits_X: dict name -> list of raw window arrays to embed post-finetune
    (train/dev/eval, full — not just the bounded subset — so downstream
    metrics stay comparable to the frozen-REVE full-split evaluation).
    Returns dict name -> (n, d) pooled-embedding array."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel
    from emeg_fm.eeg_fm import REVE_BASE_ID, REVE_POS_ID

    pos_bank = AutoModel.from_pretrained(REVE_POS_ID, trust_remote_code=True).to(device).eval()
    model = AutoModel.from_pretrained(REVE_BASE_ID, trust_remote_code=True).to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params = _patch_lora(model, LORA_BLOCKS)
    head = nn.Linear(512, 1).to(device)
    opt = torch.optim.Adam(lora_params + list(head.parameters()), lr=lr)

    Xtr = np.concatenate([X for X, _ in train_recs])
    ytr = np.concatenate([y for _, y in train_recs]).astype(np.float32)
    n_pos, n_neg = ytr.sum(), len(ytr) - ytr.sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = len(Xtr)
    print(f"[lora-pilot] fine-tuning on {n} windows ({int(n_pos)} seizure) "
          f"for {epochs} epoch(s), {len(lora_params)} LoRA tensors", flush=True)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb = _prep_batch(Xtr[idx], device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            feat = _pooled_forward(model, pos_bank, xb, ch_names, device)
            logit = head(feat).squeeze(-1)
            loss = loss_fn(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss) * len(idx)
        print(f"  epoch {ep}: mean loss = {total_loss / n:.4f}", flush=True)

    model.eval()
    embeds = {}
    with torch.no_grad():
        for name, recs in splits_X.items():
            feats = []
            for X in recs:
                for i in range(0, len(X), 32):
                    xb = _prep_batch(X[i:i + 32], device)
                    feats.append(_pooled_forward(model, pos_bank, xb, ch_names, device).cpu().numpy())
            embeds[name] = np.concatenate(feats).astype(np.float32) if feats else np.zeros((0, 512), np.float32)
    return embeds


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-nonseizure-per-patient", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = cb.subject_splits(None)
    print(f"[lora-pilot] patients train/dev/eval={len(tr_s)}/{len(dv_s)}/{len(ev_s)}", flush=True)
    # NOTE: dev/eval are ALSO bounded (all-seizure + <=N clean/patient), not
    # the full corpus splits used by the original frozen-REVE sweep -- a full
    # re-load of all 23 patients' EDFs for this differentiable, gradient-based
    # pipeline was intractable for a pilot (>7h with no progress, killed).
    # This means absolute eval N differs from docs/chbmit_epilepsy_results.md;
    # the comparison that matters here is the identity-free Delta shift under
    # otherwise-identical bounded data, not absolute-AUC parity with that doc.
    bounded_train = iter_bounded_recordings(tr_s, a.max_nonseizure_per_patient)
    bounded_dev = iter_bounded_recordings(dv_s, a.max_nonseizure_per_patient)
    bounded_eval = iter_bounded_recordings(ev_s, a.max_nonseizure_per_patient)
    print(f"[lora-pilot] bounded recordings train/dev/eval: "
          f"{len(bounded_train)}/{len(bounded_dev)}/{len(bounded_eval)} "
          f"(all-seizure + <={a.max_nonseizure_per_patient} clean/patient each)", flush=True)

    def load_all(recs):
        out = []
        for edf, events, subj in recs:
            X, starts = cb.load_windows(edf, cb.WIN_S, 100.0)
            if X is None:
                continue
            y = cb.label_windows(starts, cb.WIN_S, events)
            dur_h = (starts[-1] + cb.WIN_S) / 3600.0 if len(starts) else 0.0
            out.append(dict(X=X, y=y, starts=starts, seiz=events,
                             patient=cb._patient_id(subj), dur_h=dur_h))
        return out

    train_bounded = load_all(bounded_train)
    train_full_recs = train_bounded  # same bounded set doubles as the post-FT train embed source
    dev_recs = load_all(bounded_dev)
    eval_recs = load_all(bounded_eval)

    ch_names = cb._reve_channel_names()
    splits_X = {
        "train": [r["X"] for r in train_full_recs],
        "dev": [r["X"] for r in dev_recs],
        "eval": [r["X"] for r in eval_recs],
    }
    embeds = finetune_and_embed(
        [(r["X"], r["y"]) for r in train_bounded], splits_X, ch_names, dev, epochs=a.epochs)

    def attach_emb(recs, emb):
        i = 0
        for r in recs:
            n = len(r["X"])
            r["emb"] = emb[i:i + n]
            i += n
        return recs

    train_e = attach_emb(train_full_recs, embeds["train"])
    dev_e = attach_emb(dev_recs, embeds["dev"])
    eval_e = attach_emb(eval_recs, embeds["eval"])

    configs = [False, True] if a.both else [False]
    results = {}
    for erase in configs:
        out = cb.evaluate(train_e, dev_e, eval_e, "reve", erase_patient=erase)
        results[erase] = out
        tag = " (identity-free)" if erase else ""
        print(f"\n[lora-pilot] LoRA-REVE{tag} SzCORE:")
        print(f"  eval AUC[.1-10 FP/day] = {out['szcore_auc']:.3f}  sens@1/d={out['szcore_sens1']*100:.1f}%"
              f"  sens@10/d={out['szcore_sens10']*100:.1f}%  F1={out['op_f1']:.3f} @ {out['op_faday']:.1f} FP/day")
    if False in results and True in results:
        d = results[False]["szcore_auc"] - results[True]["szcore_auc"]
        print(f"\n[lora-pilot] LoRA-REVE identity-free Δ(eval AUC) = {d:+.3f}")


if __name__ == "__main__":
    main()
