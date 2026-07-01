"""Phase 2 (GPU / NGC container): REVE-embed the natview per-TR windows from Phase 1.

For each prep_*.npz (Phase 1): normalize the per-TR windows with REVE's fit-once scaler
(+/-15 clamp), embed each TR window through REVE (layer 6, token mean-pool) -> (n_trig, 512),
save emb_{sub}_{ses}.npz. Reads only numpy (no MNE) so it runs cleanly in the container.
Run via scripts/natview_reve_embed_t9.sh. Output: /mnt/t9/natview_reve/emb_{sub}_{ses}.npz
"""
import glob
import os

import numpy as np

from emeg_fm.alljoined import ReveInputNorm
from emeg_fm.eeg_fm import REVEAdapter

OUT = "/mnt/t9/natview_reve"
LAYER = int(os.environ.get("REVE_LAYER", "6"))
BATCH = 32
NON_EEG = {"ECG", "EOG", "VEOG", "HEOG", "EMG", "GSR", "RESP", "STATUS", "STI", "TRIG"}


def main():
    adapter = REVEAdapter(layer=LAYER)
    loaded = adapter.load_model("brain-bzh/reve-base")
    try:
        vocab = set(loaded["pos_bank"].mapping.keys())
    except Exception:  # noqa: BLE001
        vocab = None
    print(f"[embed] REVE layer={LAYER} d={adapter.output_dim} "
          f"vocab={len(vocab) if vocab else 'unknown'}", flush=True)

    preps = sorted(glob.glob(f"{OUT}/prep_*.npz"))
    print(f"[embed] {len(preps)} prep files", flush=True)
    for i, pth in enumerate(preps):
        key = os.path.basename(pth)[len("prep_"):-len(".npz")]
        out = f"{OUT}/emb_{key}.npz"
        if os.path.exists(out):
            print(f"  [{i + 1}/{len(preps)}] {key} cached", flush=True)
            continue
        z = np.load(pth, allow_pickle=True)
        win = z["reve_win"]                                   # (n_trig, n_ch, 420)
        ch_names = [str(c) for c in z["ch_names"]]
        if vocab is not None:
            keep = [j for j, c in enumerate(ch_names) if c in vocab]
        else:
            keep = [j for j, c in enumerate(ch_names) if c.upper() not in NON_EEG]
        if len(keep) < 8:
            print(f"  [{i + 1}/{len(preps)}] {key} SKIP ({len(keep)} usable ch)", flush=True)
            continue
        win, ch = win[:, keep, :], [ch_names[j] for j in keep]
        norm = ReveInputNorm(sfreq_out=200.0, clamp=15.0)
        proc = norm.fit_transform(win.astype(np.float64), 200.0)
        emb = []
        for b in range(0, proc.shape[0], BATCH):
            chunk = proc[b:b + BATCH].astype(np.float32)
            feats = np.asarray(adapter.extract_features(
                loaded, {"eeg": chunk, "electrode_names": ch, "ch_names": ch}), np.float32)
            if feats.ndim == 3:
                feats = feats.mean(axis=1)
            emb.append(feats)
        emb = np.concatenate(emb, 0)                          # (n_trig, 512)
        np.savez_compressed(out, reve_emb=emb.astype(np.float32), ch_used=np.array(ch))
        print(f"  [{i + 1}/{len(preps)}] {key} reve_emb{emb.shape} ({len(ch)}ch)", flush=True)
    print("[embed] done", flush=True)


if __name__ == "__main__":
    main()
