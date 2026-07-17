"""Matched control for lora_reve_hmc_pilot.py: frozen REVE (no LoRA), probe
fit on the SAME bounded 20-subject train set, same full dev/eval split. Isolates
whether the LoRA pilot's shrunken identity-free Delta (+0.112 vs frozen-on-full-
train's +0.490) is due to LoRA fine-tuning itself, or just the mechanical effect
of fewer train subjects giving LEACE less patient-identity signal to erase.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_sleep_hmc as hmc  # noqa: E402
import atlas_sleep_section as sl  # noqa: E402
from lora_reve_hmc_pilot import bounded_train_subjects  # noqa: E402


def main():
    import warnings; warnings.filterwarnings("ignore")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_s, dv_s, ev_s = hmc.subject_splits(None)
    lora_train_subjects = bounded_train_subjects(tr_s, 20)
    print(f"[matched-control] subjects train(bounded)/dev/eval="
          f"{len(lora_train_subjects)}/{len(dv_s)}/{len(ev_s)}", flush=True)

    embed_fn = hmc.make_embed_fn("reve", 100.0, dev)
    train = hmc.build_split(lora_train_subjects, embed_fn=embed_fn)
    devl = hmc.build_split(dv_s, embed_fn=embed_fn)
    evl = hmc.build_split(ev_s, embed_fn=embed_fn)
    print(f"  train {sum(len(r['y']) for r in train)} epochs, "
          f"dev {sum(len(r['y']) for r in devl)}, eval {sum(len(r['y']) for r in evl)}", flush=True)

    for erase in (False, True):
        out = sl.evaluate(train, devl, evl, "reve", erase_patient=erase)
        tag = " (identity-free)" if erase else ""
        print(f"\n[matched-control] frozen-REVE{tag} 5-class AASM staging:")
        print(f"  eval balanced-acc = {out['eval_bacc']*100:.1f}%  kappa = {out['eval_kappa']:.3f}")
        if erase:
            print(f"[matched-control] identity-free Δ(eval kappa) = {base_kappa - out['eval_kappa']:+.3f}")
        else:
            base_kappa = out["eval_kappa"]


if __name__ == "__main__":
    main()
