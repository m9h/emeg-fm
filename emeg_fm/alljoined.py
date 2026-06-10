"""Data-prep helpers for the Alljoined-1.6M EEG→image dataset.

Alljoined ships per-subject epochs as a *pickled dict* ``.npy``
(``preprocessed_eeg/sub-XX/preprocessed_eeg_{train,test}_flat.npy``):

    preprocessed_eeg_data : (n_trials, 32, 250) float64, microvolts
    ch_names              : list[32]  (standard 10-20 labels)
    configs               : {sfreq: 250, tmin: -0.2, tmax: 1.0, ...}
    times                 : (250,)

These helpers turn that into REVE-ready input: load the dict, average
repeated trials per stimulus image (the test partition shows ~200 shared
images many times → trial-averaging buys SNR), and resample/normalize into
REVE's expected distribution (200 Hz, per-channel z-score, clamp ±15 — the
same recipe as ``scripts/extract_eeg_fm_acts.py``). Pure numpy/scipy so the
transforms are unit-testable without torch or the foundation model.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly
from math import gcd


def topk_retrieval(pred, gallery, ks=(1, 5), labels=None) -> dict:
    """Cosine top-k retrieval accuracy of predicted vs. true embeddings.

    For each query (a decoded embedding), rank every gallery embedding by
    cosine similarity and record where the correct gallery item lands. This
    is the standard EEG/fMRI-to-image decoding metric: a prediction "hits"
    at k if the true image is among the k most similar gallery entries.

    Parameters
    ----------
    pred : (n_query, d) predicted embeddings.
    gallery : (n_gallery, d) candidate (true) embeddings.
    ks : iterable of ints — the k values to report (``top{k}`` keys).
    labels : (n_query,) int, optional. ``labels[i]`` is the gallery row that
        is the correct match for query ``i``. If omitted, identity matching
        is assumed (query ``i`` ↔ gallery ``i``), which requires
        ``n_query == n_gallery``.

    Returns
    -------
    dict with ``top{k}`` accuracies (floats in [0, 1]), ``median_rank``
    (1-indexed), ``ranks`` (int array, the correct item's rank per query),
    and ``chance_top1`` (= 1 / n_gallery).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gallery = np.asarray(gallery, dtype=np.float64)
    n_query, n_gallery = pred.shape[0], gallery.shape[0]

    if labels is None:
        if n_query != n_gallery:
            raise ValueError(
                f"identity matching needs n_query == n_gallery, got "
                f"{n_query} and {n_gallery}; pass explicit `labels`."
            )
        labels = np.arange(n_query)
    else:
        labels = np.asarray(labels).astype(int)

    pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    gn = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    sim = pn @ gn.T                                       # (n_query, n_gallery)

    correct = sim[np.arange(n_query), labels]
    # 1-indexed rank = how many gallery items beat the correct one, + 1.
    ranks = 1 + np.sum(sim > correct[:, None], axis=1)

    out = {f"top{k}": float(np.mean(ranks <= k)) for k in ks}
    out["median_rank"] = float(np.median(ranks))
    out["ranks"] = ranks.astype(int)
    out["chance_top1"] = 1.0 / n_gallery
    return out


def load_subject_npy(path: str) -> dict:
    """Load a pickled-dict Alljoined epoch file into a plain dict.

    Returns ``{"eeg": (n,32,250), "ch_names": [...], "sfreq": float,
    "times": (...)}``.
    """
    raw = np.load(path, allow_pickle=True)
    d = raw.item() if isinstance(raw, np.ndarray) else raw
    cfg = d.get("configs", {}) or {}
    return {
        "eeg": np.asarray(d["preprocessed_eeg_data"]),
        "ch_names": list(d["ch_names"]),
        "sfreq": float(cfg.get("sfreq", 250)),
        "times": np.asarray(d.get("times")) if d.get("times") is not None else None,
    }


def average_by_image(eeg: np.ndarray, image_ids) -> tuple:
    """Average trials that share a stimulus image.

    Parameters
    ----------
    eeg : (n_trials, C, T)
    image_ids : (n_trials,) array-like of hashable ids (str or int).

    Returns
    -------
    (averaged, unique_ids, counts):
        averaged   : (n_unique, C, T) mean over each image's trials
        unique_ids : (n_unique,) sorted unique ids, aligned to ``averaged``
        counts     : (n_unique,) trials averaged per image
    """
    eeg = np.asarray(eeg)
    ids = np.asarray(image_ids)
    unique_ids, inverse, counts = np.unique(
        ids, return_inverse=True, return_counts=True
    )
    n_unique = unique_ids.shape[0]
    out = np.zeros((n_unique,) + eeg.shape[1:], dtype=np.float64)
    np.add.at(out, inverse, eeg.astype(np.float64))
    out /= counts.reshape((n_unique,) + (1,) * (eeg.ndim - 1))
    return out, unique_ids, counts


def preprocess_for_reve(
    eeg: np.ndarray,
    sfreq_in: float,
    sfreq_out: float = 200.0,
    clamp: float = 15.0,
) -> np.ndarray:
    """Resample to ``sfreq_out`` then per-channel z-score + clamp.

    Matches REVE's training input distribution (NeuralBench ``reve.yaml``:
    200 Hz, StandardScaler, clamp 15). Resampling uses polyphase filtering
    along the time axis; z-score is computed per (trial, channel) over time.
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    if int(round(sfreq_in)) != int(round(sfreq_out)):
        g = gcd(int(round(sfreq_out)), int(round(sfreq_in)))
        up = int(round(sfreq_out)) // g
        down = int(round(sfreq_in)) // g
        eeg = resample_poly(eeg, up=up, down=down, axis=-1)

    mu = eeg.mean(axis=-1, keepdims=True)
    sigma = eeg.std(axis=-1, keepdims=True) + 1e-8
    eeg = (eeg - mu) / sigma
    np.clip(eeg, -clamp, clamp, out=eeg)
    return eeg
