"""Pilot: does LoRA fine-tuning of REVE change the identity-free Delta,
compared to a frozen linear probe, on HMC sleep staging?

Follow-up to scripts/lora_reve_chbmit_pilot.py, which was inconclusive:
CHB-MIT is REVE's worst dataset in the program (frozen AUC 0.014, near the
scorer's floor), so LoRA had no real signal to either preserve or destroy.
HMC is the opposite case -- REVE's CLEANEST, STRONGEST identity-trap result
(docs/hmc_sleep_results.md): frozen REVE WINS raw performance (kappa 0.898 >
classical 0.859) yet COLLAPSES hardest under identity erasure (Delta+0.490
vs classical Delta+0.085). If LoRA fine-tuning changes how much of that
kappa is identity-derived, this is where it should show up.

Same LoRA targets as the CHB-MIT pilot (to_qkv/to_out in blocks {0,1,2,3,10},
the under-trained blocks per HT-SR/WeightWatcher analysis) -- reused directly
from lora_reve_chbmit_pilot.py rather than reimplemented. Head is now a
5-class (AASM W/N1/N2/N3/REM) linear layer with cross-entropy loss, not
binary BCE.

Bounded, disclosed pilot scope: LoRA fine-tuning uses only the first 20 (of
~105) train subjects to keep the fine-tuning loop tractable -- HMC's full
train split is ~90k+ 30s epochs across 105 subjects, more than fits a single
pilot's fine-tuning budget. Dev/eval use the FULL split (same subjects as
docs/hmc_sleep_results.md's subject partition, since subject_splits() is
deterministic) so the eval-side comparison to that doc's frozen-REVE
baseline is apples-to-apples.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_sleep_hmc as hmc  # noqa: E402
import atlas_sleep_section as sl  # noqa: E402
from lora_reve_chbmit_pilot import (  # noqa: E402
    LORA_BLOCKS, _make_lora_linear_cls, _patch_lora, _pooled_forward, _prep_batch,
)

N_TRAIN_SUBJECTS_FOR_LORA = 20


def bounded_train_subjects(train_subjects, n):
    """Deterministic first-n (sorted) subset of the train subjects, so the
    LoRA fine-tuning loop stays tractable while dev/eval keep the full split
    (same subject partition as docs/hmc_sleep_results.md's frozen baseline)."""
    return set(sorted(train_subjects)[:n])


def finetune_and_embed(train_recs, splits_X, ch_names, device, epochs=1, lr=1e-3, batch=16):
    """train_recs: list of (X, y) window arrays for the bounded train subset.
    splits_X: dict name -> list of raw window arrays to embed post-finetune.
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
    n_classes = 5
    head = nn.Linear(512, n_classes).to(device)
    opt = torch.optim.Adam(lora_params + list(head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Xtr = np.concatenate([X for X, _ in train_recs])
    ytr = np.concatenate([y for _, y in train_recs]).astype(np.int64)
    n = len(Xtr)
    print(f"[lora-hmc-pilot] fine-tuning on {n} epochs across "
          f"{N_TRAIN_SUBJECTS_FOR_LORA} subjects, {epochs} epoch(s), "
          f"{len(lora_params)} LoRA tensors", flush=True)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb = _prep_batch(Xtr[idx], device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            feat = _pooled_forward(model, pos_bank, xb, ch_names, device)
            logits = head(feat)
            loss = loss_fn(logits, yb)
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
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--n-train-subjects", type=int, default=N_TRAIN_SUBJECTS_FOR_LORA)
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = hmc.subject_splits(None)
    lora_train_subjects = bounded_train_subjects(tr_s, a.n_train_subjects)
    print(f"[lora-hmc-pilot] subjects train/dev/eval={len(tr_s)}/{len(dv_s)}/{len(ev_s)}; "
          f"LoRA fine-tunes on {len(lora_train_subjects)} of the {len(tr_s)} train subjects", flush=True)

    def load_all(recs, subjects):
        out = []
        for edf, txt, subj in recs:
            if subj not in subjects:
                continue
            X, y = hmc.load_epochs(edf, txt)
            if X is None:
                continue
            out.append(dict(X=X, y=y, patient=subj))
        return out

    all_recs = hmc.iter_recordings()
    train_bounded = load_all(all_recs, lora_train_subjects)
    dev_recs = load_all(all_recs, dv_s)
    eval_recs = load_all(all_recs, ev_s)
    print(f"[lora-hmc-pilot] loaded windows: LoRA-train {sum(len(r['y']) for r in train_bounded)}, "
          f"dev {sum(len(r['y']) for r in dev_recs)}, eval {sum(len(r['y']) for r in eval_recs)}",
          flush=True)

    ch_names = hmc._reve_channel_names()
    splits_X = {
        "train": [r["X"] for r in train_bounded],
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

    train_e = attach_emb(train_bounded, embeds["train"])
    dev_e = attach_emb(dev_recs, embeds["dev"])
    eval_e = attach_emb(eval_recs, embeds["eval"])

    configs = [False, True] if a.both else [False]
    results = {}
    for erase in configs:
        out = sl.evaluate(train_e, dev_e, eval_e, "reve", erase_patient=erase)
        results[erase] = out
        tag = " (identity-free)" if erase else ""
        print(f"\n[lora-hmc-pilot] LoRA-REVE{tag} 5-class AASM staging:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        print(f"  (dev balanced-acc = {out['dev_bacc']*100:.1f}%  kappa = {out['dev_kappa']:.3f})")
    if False in results and True in results:
        d = results[False]["eval_kappa"] - results[True]["eval_kappa"]
        print(f"\n[lora-hmc-pilot] LoRA-REVE identity-free Δ(eval kappa) = {d:+.3f} "
              f"(normal {results[False]['eval_kappa']:.3f} - idfree {results[True]['eval_kappa']:.3f})")


if __name__ == "__main__":
    main()
