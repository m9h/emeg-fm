"""REVE foundation-model brain-age on the SAME cohort/CV as the classical and
NEOBA baselines.

For each subject's resting epochs we extract frozen REVE features (one block,
mean-pooled over tokens, then averaged across epochs -> one embedding per
subject), then fit a ridge brain-age head under 10-fold CV. Reporting MAE / R^2
on identical KFold splits makes REVE directly comparable to coffeine
filterbank-riemann and NEOBA on this cohort (the brain-age benchmark goal).

Preprocessing follows REVE's published input contract (reve.yaml) via
:class:`emeg_fm.alljoined.ReveInputNorm`: resample 200 Hz -> per-recording
per-channel z-score (fit once over the subject's epochs) -> clamp +/-15.

Runs inside the Docker NGC PyTorch 26.05 container on /mnt/t9 (GPU + transformers);
see scripts/reve_brain_age_t9.sh for the launcher.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np


def _load_ages(participants):
    import pandas as pd

    df = pd.read_csv(participants, sep=None, engine="python")
    id_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    ids = df[id_col].astype(str).str.replace("sub-", "", regex=False)
    return dict(zip(ids, df["age"].astype(float)))


def _subject_embedding(adapter, loaded, norm, data, sfreq, ch_names, batch_size):
    """One embedding per subject: REVE block features, token- then epoch-pooled."""
    proc = norm.fit_transform(np.asarray(data, dtype=np.float64), sfreq)
    pooled = []
    for i in range(0, proc.shape[0], batch_size):
        chunk = proc[i:i + batch_size].astype(np.float32)
        feats = adapter.extract_features(
            loaded,
            {"eeg": chunk, "electrode_names": ch_names, "ch_names": ch_names},
        )
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 3:               # (B, P, D) -> mean over tokens
            feats = feats.mean(axis=1)
        pooled.append(feats)
    return np.concatenate(pooled, axis=0).mean(axis=0)  # mean over epochs -> (D,)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs-glob",
                   default="/mnt/t9/lemon_epo/*proc-autoreject_epo.fif")
    p.add_argument("--participants",
                   default="/data/datasets/lemon/LEMON_EEG_BIDS/participants.tsv")
    p.add_argument("--subject-regex", default=r"sub-([A-Za-z0-9]+)")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=0,
                   help="KFold shuffle seed. Pass 42 to land on the SAME splits "
                        "as compute_benchmark_age_prediction (HBN/TUAB/etc.).")
    p.add_argument("--egi-remap", action="store_true",
                   help="Remap EGI GSN-HydroCel labels (E1..E128, e.g. HBN) onto "
                        "REVE's electrode vocabulary by nearest 3D montage "
                        "position; standard 10-20 labels pass through unchanged.")
    p.add_argument("--out", default="/mnt/t9/lemon_epo/reve_lemon_emb.npz")
    args = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import mne
    mne.set_log_level("error")

    from emeg_fm.eeg_fm import REVEAdapter
    from emeg_fm.alljoined import ReveInputNorm

    out = Path(args.out)
    if out.exists():
        z = np.load(out, allow_pickle=False)
        X, ages = z["X"], z["ages"]
        print(f"loaded cached embeddings {X.shape} from {out}")
    else:
        age = _load_ages(args.participants)
        pat = re.compile(args.subject_regex)
        files = sorted(Path("/").glob(args.epochs_glob.lstrip("/")))

        adapter = REVEAdapter(layer=args.layer)
        loaded = adapter.load_model("brain-bzh/reve-base")
        print(f"REVE loaded: d_model={adapter.output_dim}, layer={args.layer}")

        reve_vocab = None
        if args.egi_remap:
            from emeg_fm.eeg_fm import _labram_map_ch_names
            reve_vocab = list(getattr(loaded["pos_bank"], "mapping", []) or [])
            if not reve_vocab:
                raise SystemExit("--egi-remap: REVE pos_bank exposes no .mapping "
                                 "vocabulary to remap onto")
            print(f"EGI remap on: {len(reve_vocab)}-electrode REVE vocab")

        embs, ages, t0 = [], [], time.time()
        for f in files:
            m = pat.search(f.name)
            if not m:
                continue
            sid = m.group(1)
            if sid not in age or not np.isfinite(age[sid]):
                continue
            ep = mne.read_epochs(f, preload=True, verbose="error")
            names = (_labram_map_ch_names(ep.ch_names, vocab=reve_vocab)
                     if reve_vocab else ep.ch_names)
            emb = _subject_embedding(
                adapter, loaded, ReveInputNorm(),
                ep.get_data(copy=False), ep.info["sfreq"], names,
                args.batch_size,
            )
            embs.append(emb)
            ages.append(age[sid])
            del ep
            if len(ages) % 10 == 0:
                print(f"  {len(ages)} subjects ({time.time() - t0:.0f}s)",
                      flush=True)
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

    print(f"\nREVE brain-age: {X.shape[0]} subjects, d={X.shape[1]}")
    print(f"dummy-mean MAE = {np.mean(np.abs(ages - ages.mean())):.2f} yr")
    print(f"KFold(n_splits={args.n_splits}, shuffle=True, random_state={args.seed})")
    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 25)))
    sc = cross_validate(reg, X, ages, cv=cv,
                        scoring=("neg_mean_absolute_error", "r2"))
    mae = -sc["test_neg_mean_absolute_error"]
    r2 = sc["test_r2"]
    print(f"REVE+RidgeCV  MAE = {mae.mean():.2f} +/- {mae.std():.2f} yr   "
          f"R^2 = {r2.mean():.3f} +/- {r2.std():.3f}")


if __name__ == "__main__":
    main()
