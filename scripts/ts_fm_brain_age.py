"""Generic time-series foundation-model (TS-FM) brain-age on the SAME cohort/CV
as the classical (coffeine), NEOBA, and REVE baselines.

Tests the NeuroAtlas (2026, arXiv 2605.14698) headline — "EEG-specific FMs do not
consistently beat generic time-series FMs" — on AWAKE RESTING-STATE cohorts
(TDBRAIN / LEMON / HBN), the slice NeuroAtlas did not cover (its brain-age domain
is sleep-EEG only). If a domain-agnostic TS-FM matches frozen REVE here, that is
the same story on new ground; if REVE wins, it's a counterpoint.

Per subject: resample each resting epoch to the model's context length, run the
FROZEN TS-FM as an embedding extractor, mean-pool over channels then epochs -> one
embedding/subject; RidgeCV under identical KFold(seed) splits -> MAE / R^2. Mirrors
scripts/reve_brain_age.py exactly so numbers drop into the same benchmark table.

TS-FMs are UNIVARIATE: each EEG channel is embedded independently (fed as a
1-channel series) and the per-channel embeddings are pooled, matching NeuroAtlas's
channel-wise protocol. The model applies its own instance normalization (RevIN),
so no manual scaling is done here.

Runs in the NGC PyTorch container on /mnt/t9 (GPU + the model's pip package);
see scripts/ts_fm_brain_age_t9.sh. Add a new TS-FM by writing one Adapter class
and registering it in ADAPTERS.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np


# ---------------- TS-FM adapters ----------------
# Each adapter: load(hf_id) -> None (stores model), embed(x) where x is
# (B, C, L) float32 -> (B, D) numpy (already channel-pooled). seq_len is the
# context length the caller resamples each epoch to before calling embed.
class MomentAdapter:
    """AutonLab MOMENT-1 (T5-style patch encoder, explicit embedding mode)."""
    hf_default = "AutonLab/MOMENT-1-large"
    seq_len = 512

    def load(self, hf_id):
        import torch
        from momentfm import MOMENTPipeline
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = MOMENTPipeline.from_pretrained(
            hf_id, model_kwargs={"task_name": "embedding"})
        m.init()
        self.model = m.to(self.dev).eval()

    def embed(self, x):
        # x: (B, C, L). Feed each channel as a 1-channel series -> (B*C, 1, L),
        # embed -> (B*C, D), reshape -> (B, C, D), mean over channels.
        torch = self.torch
        B, C, L = x.shape
        xr = x.reshape(B * C, 1, L)
        outs = []
        with torch.no_grad():
            for i in range(0, xr.shape[0], 256):
                t = torch.tensor(xr[i:i + 256], dtype=torch.float32, device=self.dev)
                e = self.model(x_enc=t).embeddings      # (chunk, D)
                outs.append(np.asarray(e.float().cpu()))
        emb = np.concatenate(outs, axis=0)               # (B*C, D)
        return emb.reshape(B, C, -1).mean(axis=1)        # (B, D)


ADAPTERS = {"moment": MomentAdapter}


def _resample_time(x, target_len):
    """Resample the last axis of x (..., L) to target_len via FFT (scipy)."""
    from scipy.signal import resample
    if x.shape[-1] == target_len:
        return x
    return resample(x, target_len, axis=-1)


def _load_ages(participants):
    import pandas as pd
    df = pd.read_csv(participants, sep=None, engine="python")
    id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    ids = df[id_col].astype(str).str.replace("sub-", "", regex=False)
    return dict(zip(ids, df["age"].astype(float)))


def _subject_embedding(adapter, data, sfreq, seq_len, batch_size):
    """One embedding/subject: resample epochs to seq_len, embed, epoch-mean."""
    x = _resample_time(np.asarray(data, dtype=np.float32), seq_len)  # (n_ep, C, seq_len)
    pooled = []
    for i in range(0, x.shape[0], batch_size):
        pooled.append(adapter.embed(x[i:i + batch_size]))            # (chunk, D)
    return np.concatenate(pooled, axis=0).mean(axis=0)               # (D,)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(ADAPTERS), default="moment")
    p.add_argument("--hf-id", default=None, help="override the adapter's default HF id")
    p.add_argument("--epochs-glob", default="/mnt/t9/lemon_epo/*proc-autoreject_epo.fif")
    p.add_argument("--participants",
                   default="/data/datasets/lemon/LEMON_EEG_BIDS/participants.tsv")
    p.add_argument("--subject-regex", default=r"sub-([A-Za-z0-9]+)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=42,
                   help="KFold shuffle seed; 42 = SAME splits as the classical/REVE runs.")
    p.add_argument("--out", default="/mnt/t9/ts_fm_brain_age_emb.npz")
    args = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import mne
    mne.set_log_level("error")

    out = Path(args.out)
    if out.exists():
        z = np.load(out, allow_pickle=False)
        X, ages = z["X"], z["ages"]
        print(f"loaded cached embeddings {X.shape} from {out}")
    else:
        adapter = ADAPTERS[args.model]()
        hf_id = args.hf_id or adapter.hf_default
        adapter.load(hf_id)
        print(f"{args.model} loaded: {hf_id} (seq_len={adapter.seq_len}, dev={adapter.dev})")

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
            emb = _subject_embedding(adapter, ep.get_data(copy=False),
                                     ep.info["sfreq"], adapter.seq_len, args.batch_size)
            embs.append(emb)
            ages.append(age[sid])
            del ep
            if len(ages) % 10 == 0:
                print(f"  {len(ages)} subjects ({time.time() - t0:.0f}s)", flush=True)
            if args.max_subjects and len(ages) >= args.max_subjects:
                break

        X = np.vstack(embs).astype(np.float32)
        ages = np.asarray(ages)
        np.savez(out, X=X, ages=ages)
        print(f"\ncached embeddings -> {out}")

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print(f"\n{args.model} brain-age: {X.shape[0]} subjects, d={X.shape[1]}")
    print(f"dummy-mean MAE = {np.mean(np.abs(ages - ages.mean())):.2f} yr")
    print(f"KFold(n_splits={args.n_splits}, shuffle=True, random_state={args.seed})")
    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 25)))
    sc = cross_validate(reg, X, ages, cv=cv, scoring=("neg_mean_absolute_error", "r2"))
    mae = -sc["test_neg_mean_absolute_error"]
    r2 = sc["test_r2"]
    print(f"{args.model}+RidgeCV  MAE = {mae.mean():.2f} +/- {mae.std():.2f} yr   "
          f"R^2 = {r2.mean():.3f} +/- {r2.std():.3f}")


if __name__ == "__main__":
    main()
