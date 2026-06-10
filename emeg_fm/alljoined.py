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


def _resample_time(eeg: np.ndarray, sfreq_in: float, sfreq_out: float) -> np.ndarray:
    """Polyphase resample along the time axis (no-op if rates match)."""
    if int(round(sfreq_in)) == int(round(sfreq_out)):
        return eeg
    g = gcd(int(round(sfreq_out)), int(round(sfreq_in)))
    up = int(round(sfreq_out)) // g
    down = int(round(sfreq_in)) // g
    return resample_poly(eeg, up=up, down=down, axis=-1)


def preprocess_for_reve(
    eeg: np.ndarray,
    sfreq_in: float,
    sfreq_out: float = 200.0,
    clamp: float = 15.0,
) -> np.ndarray:
    """Resample to ``sfreq_out`` then per-(epoch,channel) z-score + clamp.

    NOTE: this is the *per-epoch* normalization — convenient and stateless, but
    it is **not** REVE's canonical input contract. The canonical recipe
    (``reve.yaml`` via neuralset) fits one ``StandardScaler`` over the whole
    continuous recording, so per-epoch amplitude differences survive and the
    ±``clamp`` actually fires. Under this per-epoch z-score every window is
    forced to unit variance, so the clamp is effectively a no-op. Prefer
    :class:`ReveInputNorm` (fit-once, frozen) for contract-faithful and
    stream==batch-consistent normalization; this function is retained for
    stateless offline use on already-filtered data.
    """
    eeg = _resample_time(np.asarray(eeg, dtype=np.float64), sfreq_in, sfreq_out)
    mu = eeg.mean(axis=-1, keepdims=True)
    sigma = eeg.std(axis=-1, keepdims=True) + 1e-8
    eeg = (eeg - mu) / sigma
    np.clip(eeg, -clamp, clamp, out=eeg)
    return eeg


class ReveInputNorm:
    """REVE's canonical input contract (``reve.yaml``) as a fit-once transform.

    Layer-2 of the preprocessing split: turns raw epochs into REVE's training
    distribution. The order mirrors neuralset's ``_preprocess_raw``:
    resample → (per-channel ``StandardScaler``) → clamp. Crucially the scaler
    is **fit once** over a reference set (the calibration block, standing in for
    neuralset's whole continuous recording) and then **frozen**, so:

    * every online epoch is normalized with the *same* per-channel mean/std the
      calibration data established — the relative amplitude of a given epoch
      survives, and the ±``clamp`` is meaningful rather than a no-op;
    * :meth:`transform` is exactly per-epoch independent, so transforming a
      batch equals stacking single-epoch transforms (stream==batch parity).

    Bandpass / notch (Layer-3, the device bridge) are intentionally *not* here:
    Alljoined ships pre-filtered, and live-device filtering has edge-effect
    semantics that differ between a continuous stream and a single epoch.

    Parameters
    ----------
    sfreq_out, clamp : REVE input rate (200 Hz) and clamp bound (±15).
    """

    def __init__(self, sfreq_out: float = 200.0, clamp: float = 15.0):
        self.sfreq_out = float(sfreq_out)
        self.clamp = float(clamp)
        self.mean_ = None      # (C,)
        self.std_ = None       # (C,)
        self._sfreq_in = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    def fit(self, eeg: np.ndarray, sfreq_in: float) -> "ReveInputNorm":
        """Estimate per-channel mean/std over the reference set ``eeg``.

        ``eeg`` is ``(N, C, T)`` or ``(C, T)``; statistics pool over every
        non-channel axis (all trials and all time samples), matching a
        ``StandardScaler`` fit on the concatenated recording.
        """
        eeg = np.asarray(eeg, dtype=np.float64)
        if eeg.ndim == 2:
            eeg = eeg[None, ...]
        eeg = _resample_time(eeg, sfreq_in, self.sfreq_out)
        axes = tuple(i for i in range(eeg.ndim) if i != eeg.ndim - 2)  # all but C
        self.mean_ = eeg.mean(axis=axes)
        self.std_ = eeg.std(axis=axes) + 1e-8
        self._sfreq_in = float(sfreq_in)
        return self

    def transform(self, eeg: np.ndarray, sfreq_in: float | None = None) -> np.ndarray:
        """Apply the frozen per-channel scaler + clamp to ``eeg``.

        ``eeg`` is ``(N, C, T)`` or ``(C, T)``; the channel axis must match the
        fitted statistics. ``sfreq_in`` defaults to the fitted input rate.
        """
        if self.mean_ is None:
            raise RuntimeError("ReveInputNorm not fitted — call fit() first")
        eeg = np.asarray(eeg, dtype=np.float64)
        squeeze = eeg.ndim == 2
        if squeeze:
            eeg = eeg[None, ...]
        eeg = _resample_time(eeg, sfreq_in if sfreq_in is not None else self._sfreq_in,
                             self.sfreq_out)
        if eeg.shape[-2] != self.mean_.shape[0]:
            raise ValueError(
                f"channel count {eeg.shape[-2]} != fitted {self.mean_.shape[0]}")
        eeg = (eeg - self.mean_[:, None]) / self.std_[:, None]
        np.clip(eeg, -self.clamp, self.clamp, out=eeg)
        return eeg[0] if squeeze else eeg

    def fit_transform(self, eeg: np.ndarray, sfreq_in: float) -> np.ndarray:
        return self.fit(eeg, sfreq_in).transform(eeg, sfreq_in)
