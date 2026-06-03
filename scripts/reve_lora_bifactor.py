#!/usr/bin/env python
"""Scope C of the Muon experiment (docs/MUON_EXPERIMENT.md): LoRA fine-tune
frozen REVE on the HBN psychopathology bifactor probe — once with AdamW, once
with Muon on the LoRA matrices — and measure whether Muon's
gradient-orthogonalization pulls the five WeightWatcher-flagged α<2 attention
matrices toward the self-averaging α≈2 boundary (H1) while improving the probe
(the does-it-actually-help check).

Pipeline
--------
    eegdash HBN release -> braindecode windows @ 200 Hz
      -> frozen REVE with LoRA on transformer.layers.{2,1,0,3,10}.0.{to_qkv,to_out}
      -> mean-pool block-``--layer`` output -> linear head -> 4 bifactor dims
      -> train {LoRA matrices + head}, eval per-subject Pearson r on held-out split
      -> post-hoc: merge each LoRA delta, weight_spectral_summary(base) vs
         (merged) -> Δ(alpha_hill), Δ(participation_ratio) for H1

Optimizer split (the experiment's whole point)
----------------------------------------------
    --optimizer adamw : one AdamW over the LoRA matrices + head.
    --optimizer muon  : Muon (vendored, eeg_fm_spectral.lora.make_muon) over the
                        2-D LoRA matrices; AdamW over the head + any 1-D params.
                        Muon's update is RMS-matched, so it needs its own (larger)
                        --lr; tune it separately from --adam-lr (see the doc).

This is the torch half of the experiment; it imports torch lazily and runs
inside the PyTorch NGC SIF (REVE is gated — accept the Responsible Use
Agreement and have an HF token). The post-hoc spectral check reuses the
numpy ``weight_spectral_summary`` already in ``sae.py`` so no inline
WeightWatcher run is needed.

NOTE on the eval split: NeuralBench's canonical psychopathology eval holds out
HBN release R5 via a PredefinedSplit. Here we do a subject-level split within
whatever release is loaded (GroupShuffleSplit by subject) so the scaffold is
runnable on a single release; pass the same --release/--seed to both optimizer
runs so the split is identical. The A/B delta is what's interpretable, not the
absolute r.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the extraction pipeline's data helpers (and its torchaudio stub, which
# is installed at import time so braindecode imports cleanly in the SIF).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from extract_eeg_fm_acts import (  # noqa: E402  (after sys.path edit + stub)
    HBN_FIELDS, load_eegdash, make_windows, subject_metadata,
)
from eeg_fm_spectral.eeg_fm import REVEAdapter, REVE_BASE_ID  # noqa: E402
# Import the spectral summary from lora.py (not sae.py): sae.py imports JAX at
# top level, and this driver runs in the PyTorch SIF where JAX isn't installed.
from eeg_fm_spectral.lora import (  # noqa: E402
    DEFAULT_REVE_LORA_TARGETS, inject_lora, lora_delta, make_muon,
    weight_spectral_summary,
)


DEFAULT_TARGETS = ("externalizing", "p_factor", "internalizing", "attention")


# ---------------------------------------------------------------------------
# Data: windows -> (eeg array, per-window labels, per-window subject id)
# ---------------------------------------------------------------------------

def build_dataset(*, release, full, task, cache_dir, win_seconds, target_sfreq,
                  targets, max_windows):
    """Materialise windows into a single (N, C, T) float32 array plus aligned
    label / subject arrays, dropping windows whose subject lacks any target."""
    concat_ds, desc = load_eegdash(release=release, mini=not full,
                                   cache_dir=cache_dir, task=task)
    windows_ds = make_windows(concat_ds, win_seconds=win_seconds,
                              target_sfreq=target_sfreq)

    n = len(windows_ds)
    if max_windows is not None:
        n = min(n, max_windows)

    xs, ys, subs = [], [], []
    ch_names = None
    md_cache: dict[str, dict] = {}
    for i in range(n):
        x, _y, ind = windows_ds[i]                       # (C, T)
        try:
            sub_id = str(windows_ds.datasets[ind[0]].description["subject"])
        except Exception:
            sub_id = f"sub-{ind[0]:04d}"
        if ch_names is None:
            raw = getattr(windows_ds.datasets[ind[0]], "raw", None)
            if raw is None:
                raise AttributeError("could not read raw.ch_names off the "
                                     "windows dataset — braindecode API shift?")
            ch_names = list(raw.ch_names)
        if sub_id not in md_cache:
            md_cache[sub_id] = subject_metadata(desc, sub_id)
        md = md_cache[sub_id]
        label = np.array([md[t] for t in targets], dtype=np.float32)
        if not np.all(np.isfinite(label)):
            continue                                     # subject missing a dim
        xs.append(np.asarray(x, dtype=np.float32))
        ys.append(label)
        subs.append(sub_id)

    if not xs:
        raise RuntimeError(
            f"No windows with all targets {targets} present in release "
            f"R{release} task={task}. Check the description df has the bifactor "
            f"columns (they live on the HBN phenotype, not every release ships "
            f"them through eegdash)."
        )
    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    S = np.asarray(subs, dtype=object)
    print(f"[data] {X.shape[0]} windows, {len(set(subs))} subjects, "
          f"eeg shape (C,T)={X.shape[1:]}, targets={list(targets)}", flush=True)
    return X, Y, S, ch_names


def subject_split(subjects, *, val_frac, seed):
    """GroupShuffleSplit-by-subject -> boolean train/val masks over windows."""
    uniq = np.array(sorted(set(subjects.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_subs = set(uniq[:n_val].tolist())
    is_val = np.array([s in val_subs for s in subjects])
    if is_val.all() or (~is_val).all():
        raise RuntimeError(
            f"degenerate split: {len(uniq)} subjects, val_frac={val_frac} put "
            f"all windows on one side. Load more subjects (--full) or lower "
            f"--val-frac.")
    return ~is_val, is_val


# ---------------------------------------------------------------------------
# Model: frozen REVE + LoRA + mean-pool head
# ---------------------------------------------------------------------------

def build_model(layer, n_targets, rank, alpha, device):
    """Load REVE, inject LoRA on the 5 α<2 matrices, attach a mean-pool head.

    Returns ``(adapter, model_dict, head, wrapped, block_index)`` where
    ``wrapped`` is ``{target_name: LoRALinear}`` (for the post-hoc merge) and
    ``block_index`` is the resolved non-negative REVE block whose output the
    head pools.
    """
    import torch
    import torch.nn as nn

    adapter = REVEAdapter(layer=layer, device=device)
    model_dict = adapter.load_model(REVE_BASE_ID)
    model = model_dict["model"]
    wrapped = inject_lora(model, DEFAULT_REVE_LORA_TARGETS, rank=rank, alpha=alpha)
    model.to(device)  # inject_lora's fresh lora_A/lora_B init on CPU; base is on device

    d_model = adapter.output_dim
    head = nn.Linear(d_model, n_targets).to(device)

    n_blocks = adapter._n_blocks
    k = int(layer)
    if k < 0 and n_blocks is not None:
        k += n_blocks
    return adapter, model_dict, head, wrapped, k


def forward_pooled(model_dict, head, eeg, positions, block_index, device):
    """Frozen-REVE forward with grad through LoRA: mean-pool block output -> head.

    REVE's FlashAttention needs bf16, so the encoder runs under bf16 autocast
    (matching REVEAdapter.extract_features); the pooled rep is upcast to fp32
    before the head so the loss / Muon step stay fp32.
    """
    import torch

    model = model_dict["model"]
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device == "cuda" else
                    torch.autocast(device_type="cpu", dtype=torch.bfloat16,
                                   enabled=False))
    with autocast_ctx:
        out_layers = model(eeg, positions, return_output=True)
    rep = out_layers[block_index + 1].float().mean(dim=1)    # (B, d_model)
    return head(rep)


# ---------------------------------------------------------------------------
# Optimizer split
# ---------------------------------------------------------------------------

def build_optimizers(wrapped, head, *, optimizer, lr, adam_lr):
    """Return a list of optimizers. For ``muon`` the 2-D LoRA matrices go to the
    vendored Muon and the head (+ any 1-D) to AdamW — the mandatory hybrid. For
    ``adamw`` everything trainable goes to one AdamW."""
    import torch

    lora_params, head_params = [], list(head.parameters())
    for mod in wrapped.values():
        lora_params += [mod.lora_A, mod.lora_B]

    if optimizer == "adamw":
        opt = torch.optim.AdamW(lora_params + head_params, lr=lr)
        return [opt], {"adamw_lr": lr}
    elif optimizer == "muon":
        Muon = make_muon()
        muon = Muon([p for p in lora_params if p.ndim == 2], lr=lr)
        adam = torch.optim.AdamW(
            head_params + [p for p in lora_params if p.ndim != 2], lr=adam_lr)
        return [muon, adam], {"muon_lr": lr, "adamw_lr": adam_lr}
    raise ValueError(f"unknown optimizer {optimizer!r}")


# ---------------------------------------------------------------------------
# Eval: per-subject mean -> Pearson r per target
# ---------------------------------------------------------------------------

def pearson_per_target(preds, labels, subjects):
    """Aggregate window preds/labels to subject means, then Pearson r per dim.

    NeuralBench monitors ``val/pearsonr`` on subject-aggregated features; we
    mirror that by averaging window-level predictions within each subject before
    correlating against the (constant-per-subject) targets.
    """
    uniq = sorted(set(subjects.tolist()))
    P = np.stack([preds[subjects == s].mean(0) for s in uniq])
    L = np.stack([labels[subjects == s].mean(0) for s in uniq])
    rs = []
    for j in range(P.shape[1]):
        p, l = P[:, j], L[:, j]
        if p.std() < 1e-8 or l.std() < 1e-8:
            rs.append(float("nan"))
        else:
            rs.append(float(np.corrcoef(p, l)[0, 1]))
    return rs


# ---------------------------------------------------------------------------
# Post-hoc spectral check (H1)
# ---------------------------------------------------------------------------

def spectral_deltas(wrapped):
    """For each LoRA target: WeightWatcher-ish summary of base vs base+delta.

    Returns ``{target_name: {base: {...}, merged: {...},
    delta_alpha_hill, delta_participation_ratio}}``. ``alpha_hill`` is a Hill
    proxy for WeightWatcher's α (relative A/B only — see weight_spectral_summary
    docstring). Positive Δα toward 2.0 from below is the H1 signal.
    """
    import torch

    out = {}
    for name, mod in wrapped.items():
        with torch.no_grad():
            base = mod.base.weight.detach().float().cpu().numpy()
            delta = lora_delta(mod.lora_A.detach().float().cpu().numpy(),
                               mod.lora_B.detach().float().cpu().numpy(),
                               mod.scaling)
            merged = base + delta
        sb = weight_spectral_summary(base)
        sm = weight_spectral_summary(merged)
        out[name] = {
            "base": sb, "merged": sm,
            "delta_alpha_hill": sm["alpha_hill"] - sb["alpha_hill"],
            "delta_participation_ratio": (sm["participation_ratio"]
                                          - sb["participation_ratio"]),
        }
    return out


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args):
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    for t in targets:
        if t not in HBN_FIELDS:
            raise ValueError(f"target {t!r} not in HBN_FIELDS {HBN_FIELDS}")

    X, Y, S, ch_names = build_dataset(
        release=args.release, full=args.full, task=args.task,
        cache_dir=args.cache_dir, win_seconds=args.win_seconds,
        target_sfreq=args.target_sfreq, targets=targets,
        max_windows=args.max_windows)

    tr, va = subject_split(S, val_frac=args.val_frac, seed=args.seed)
    # Standardize targets on train stats (scale-free for Pearson, but keeps the
    # MSE well-conditioned across dims with different ranges).
    mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-8
    Yz = (Y - mu) / sd

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    adapter, model_dict, head, wrapped, block_index = build_model(
        args.layer, len(targets), args.rank, args.alpha, device)
    opts, opt_meta = build_optimizers(
        wrapped, head, optimizer=args.optimizer, lr=args.lr, adam_lr=args.adam_lr)

    # Positions are a pure function of the montage — compute once, reuse.
    pos_bank = model_dict["pos_bank"]
    with torch.no_grad():
        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device == "cuda" else
                        torch.autocast(device_type="cpu", dtype=torch.bfloat16,
                                       enabled=False))
        with autocast_ctx:
            positions_1 = pos_bank(ch_names)             # (1, C, 4) or similar

    loss_fn = torch.nn.MSELoss()
    tr_idx = np.where(tr)[0]
    n_tr = len(tr_idx)
    print(f"[train] optimizer={args.optimizer} {opt_meta}  "
          f"block={block_index} rank={args.rank} alpha={args.alpha}  "
          f"train_windows={n_tr} val_windows={int(va.sum())}", flush=True)

    head.train()
    for epoch in range(args.epochs):
        perm = np.random.permutation(tr_idx)
        running = 0.0
        t0 = time.time()
        for start in range(0, n_tr, args.batch_size):
            batch = perm[start:start + args.batch_size]
            eeg = torch.from_numpy(X[batch]).to(device)
            yb = torch.from_numpy(Yz[batch]).to(device)
            positions = positions_1.expand(eeg.size(0), -1, -1)
            pred = forward_pooled(model_dict, head, eeg, positions,
                                  block_index, device)
            loss = loss_fn(pred, yb)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in opts:
                o.step()
            running += float(loss) * len(batch)
        print(f"  [epoch {epoch}] mse={running / n_tr:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    # ---- eval ----
    head.eval()
    preds = np.empty_like(Yz)
    va_idx = np.where(va)[0]
    with torch.no_grad():
        for start in range(0, len(va_idx), args.batch_size):
            batch = va_idx[start:start + args.batch_size]
            eeg = torch.from_numpy(X[batch]).to(device)
            positions = positions_1.expand(eeg.size(0), -1, -1)
            pred = forward_pooled(model_dict, head, eeg, positions,
                                  block_index, device)
            preds[batch] = pred.float().cpu().numpy()

    rs = pearson_per_target(preds[va], (Yz[va]), S[va])
    rs_by_target = dict(zip(targets, rs))
    mean_r = float(np.nanmean(rs))
    print(f"[eval] per-target Pearson r (subject-aggregated, held-out):",
          flush=True)
    for t, r in rs_by_target.items():
        print(f"    {t:14s} r={r:+.3f}", flush=True)
    print(f"    {'mean':14s} r={mean_r:+.3f}", flush=True)

    spectral = spectral_deltas(wrapped)
    print(f"[spectral] post-hoc Δ on the {len(spectral)} LoRA targets "
          f"(alpha_hill toward 2.0 from below = H1):", flush=True)
    for name, d in spectral.items():
        print(f"    {name:34s} alpha_hill {d['base']['alpha_hill']:.2f}"
              f" -> {d['merged']['alpha_hill']:.2f} "
              f"(Δ {d['delta_alpha_hill']:+.3f})  "
              f"PR Δ {d['delta_participation_ratio']:+.1f}", flush=True)

    result = {
        "config": {
            "optimizer": args.optimizer, **opt_meta,
            "release": args.release, "full": bool(args.full), "task": args.task,
            "layer": args.layer, "block_index": block_index,
            "rank": args.rank, "alpha": args.alpha, "epochs": args.epochs,
            "batch_size": args.batch_size, "win_seconds": args.win_seconds,
            "target_sfreq": args.target_sfreq, "val_frac": args.val_frac,
            "seed": args.seed, "targets": list(targets),
            "lora_targets": list(DEFAULT_REVE_LORA_TARGETS),
        },
        "pearson_r": rs_by_target,
        "pearson_r_mean": mean_r,
        "spectral": spectral,
    }
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_prefix) + ".json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {out_prefix}.json", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    ap.add_argument("--lr", type=float, default=None,
                    help="LoRA-param LR. AdamW default 1e-3; Muon default 2e-2 "
                         "(RMS-matched — tune separately).")
    ap.add_argument("--adam-lr", type=float, default=1e-3,
                    help="AdamW LR for the head (+ 1-D params) in the muon hybrid.")
    ap.add_argument("--release", type=int, default=1, choices=range(1, 12),
                    metavar="N")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--task", default="RestingState")
    ap.add_argument("--layer", type=int, default=6,
                    help="REVE block whose mean-pooled output feeds the head "
                         "(default 6 — the healthy SAE-target block).")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--win-seconds", type=float, default=5.0)
    ap.add_argument("--target-sfreq", type=float, default=200.0)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--cache-dir", default="/data/derivatives/eegdash_cache")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 2e-2 if args.optimizer == "muon" else 1e-3

    train(args)


if __name__ == "__main__":
    sys.exit(main())
