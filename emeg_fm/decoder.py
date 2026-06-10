"""Per-subject streaming EEG→image decoder (frozen REVE + ridge head).

This is the realtime sibling of the offline ``scripts/extract_alljoined_reve.py``
smoke. The model stack is identical and the *only* per-subject learning is the
ridge head — which is exactly what makes "fine-tune to a new subject in under
10 minutes" tractable: the foundation model (REVE) stays frozen, and a
closed-form ridge from REVE features → CLIP image embeddings is fit in
sub-second time on whatever calibration trials the subject produced.

Two phases:

    1. **Calibrate** — collect epochs for a labelled calibration block, average
       repeats per image (SNR, as in the offline smoke), then ``fit`` the ridge.
    2. **Online** — for each new epoch, ``predict_embedding`` then ``retrieve``
       the top-k nearest gallery images by cosine similarity.

REVE + CLIP need torch and live downstream of an NGC PyTorch SIF, so they are
imported lazily. For unit tests, inject ``feature_fn`` to bypass REVE entirely.
"""
from __future__ import annotations

import numpy as np

from emeg_fm.alljoined import (
    ReveInputNorm, preprocess_for_reve, average_by_image, topk_retrieval,
)


class StreamingReveDecoder:
    """Frozen-REVE features + per-subject ridge → CLIP gallery retrieval.

    Parameters
    ----------
    gallery : (n_gallery, d_clip) CLIP image embeddings (the candidate set).
    gallery_ids : (n_gallery,) ids aligned to ``gallery`` rows; ``retrieve``
        returns these. For an :class:`emeg_fm.stimuli.ImageStimulusSet` these
        are the integer marker codes.
    model_id, layer : REVE checkpoint + block index (defaults match the smoke).
    sfreq_out, clamp : REVE input normalization (200 Hz, ±15 — reve.yaml).
    ridge_alpha : ridge regularization (1000 worked well on Alljoined).
    feature_fn : optional ``(epochs (N,C,T), ch_names) -> (N, d_feat)`` override.
        When given, REVE is never loaded — used for testing and for plugging in
        an alternative backbone.
    """

    def __init__(self, gallery, gallery_ids, *,
                 model_id="brain-bzh/reve-base", layer=6,
                 sfreq_out=200.0, clamp=15.0, ridge_alpha=1000.0,
                 device=None, reve_batch=32, feature_fn=None):
        self.gallery = np.asarray(gallery, dtype=np.float64)
        self.gallery_ids = np.asarray(gallery_ids)
        self.id_to_row = {int(g): i for i, g in enumerate(self.gallery_ids)}
        self.model_id = model_id
        self.layer = layer
        self.sfreq_out = sfreq_out
        self.clamp = clamp
        self.ridge_alpha = ridge_alpha
        self.device = device
        self.reve_batch = reve_batch
        self._feature_fn = feature_fn
        self._adapter = None
        self._loaded = None
        self._scaler = None
        self._ridge = None
        # Layer-2 REVE input contract: fit-once per-channel scaler, frozen at
        # calibration and reused for every online epoch (reve.yaml parity).
        self._norm = None

    # -- feature extraction --------------------------------------------------

    def load(self):
        """Eagerly load REVE (no-op if a ``feature_fn`` was injected)."""
        if self._feature_fn is not None or self._adapter is not None:
            return self
        from emeg_fm.eeg_fm import REVEAdapter
        self._adapter = REVEAdapter(layer=self.layer, device=self.device)
        self._loaded = self._adapter.load_model(self.model_id)
        return self

    def features(self, epochs, ch_names) -> np.ndarray:
        """``(N, C, T)`` raw epochs → ``(N, d_feat)`` pooled features.

        Applies the REVE input normalization (resample→z-score→clamp) then a
        frozen REVE forward, mean-pooling tokens. ``ch_names`` is the montage
        passed to reve-positions.
        """
        epochs = np.asarray(epochs, dtype=np.float64)
        if epochs.ndim == 2:
            epochs = epochs[None, ...]
        # sfreq_in is read per-call; epochs from the same source share it.
        if self._norm is not None and self._norm.is_fitted:
            # Frozen calibration scaler → identical transform for calib & online.
            proc = self._norm.transform(epochs, sfreq_in=self._sfreq_in)
        else:
            # Pre-calibration / stateless fallback (per-epoch z-score).
            proc = preprocess_for_reve(epochs, sfreq_in=self._sfreq_in,
                                       sfreq_out=self.sfreq_out, clamp=self.clamp)
        if self._feature_fn is not None:
            return np.asarray(self._feature_fn(proc, ch_names), dtype=np.float32)

        self.load()
        pooled = []
        for i in range(0, proc.shape[0], self.reve_batch):
            chunk = proc[i:i + self.reve_batch]
            feats = self._adapter.extract_features(
                self._loaded,
                {"eeg": chunk, "electrode_names": ch_names, "ch_names": ch_names},
            )
            feats = np.asarray(feats, dtype=np.float32)
            if feats.ndim == 3:                       # (B, P, D) → mean tokens
                feats = feats.mean(axis=1)
            pooled.append(feats)
        return np.concatenate(pooled, axis=0)

    _sfreq_in = 250.0   # overwritten by fit_from_trials / set_sfreq_in

    def set_sfreq_in(self, sfreq_in):
        self._sfreq_in = float(sfreq_in)
        return self

    # -- calibration (the <10-min per-subject fine-tune) --------------------

    def fit(self, X_feat, codes):
        """Fit the ridge head: standardized REVE features → CLIP gallery rows.

        ``codes`` are marker codes; their gallery embeddings are the targets.
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X_feat, dtype=np.float64)
        rows = np.array([self.id_to_row[int(c)] for c in codes], dtype=int)
        Y = self.gallery[rows]
        self._scaler = StandardScaler().fit(X)
        self._ridge = Ridge(alpha=self.ridge_alpha).fit(self._scaler.transform(X), Y)
        return self

    def fit_from_trials(self, trials, average=True):
        """Calibrate from a list of :class:`emeg_fm.streaming.Trial`.

        With ``average=True`` (default), repeated presentations of the same
        image are trial-averaged before feature extraction — the SNR trick the
        offline smoke relied on. Returns a small calibration report dict.
        """
        trials = list(trials)
        if not trials:
            raise ValueError("no calibration trials collected")
        self.set_sfreq_in(trials[0].sfreq)
        ch_names = trials[0].ch_names
        epochs = np.stack([t.epoch for t in trials], axis=0)
        codes = np.array([t.code for t in trials], dtype=int)

        if average:
            avg, uniq, counts = average_by_image(epochs, codes)
            epochs, codes = avg, uniq
        else:
            counts = np.ones(len(codes))

        # only keep codes that exist in the gallery
        keep = np.array([int(c) in self.id_to_row for c in codes], dtype=bool)
        epochs, codes = epochs[keep], codes[keep]
        if len(codes) < 2:
            raise ValueError(
                f"only {len(codes)} calibration image(s) overlap the gallery — "
                f"need ≥2 to fit a ridge"
            )
        # Fit the REVE input scaler on the calibration block and freeze it; the
        # online path reuses these per-channel stats (reve.yaml parity).
        self._norm = ReveInputNorm(sfreq_out=self.sfreq_out, clamp=self.clamp)
        self._norm.fit(epochs, self._sfreq_in)
        X = self.features(epochs, ch_names)
        self.fit(X, codes)
        return {
            "n_trials": len(trials),
            "n_fit_samples": int(len(codes)),
            "n_images": int(len(np.unique(codes))),
            "mean_trials_per_image": float(np.mean(counts[keep] if average else counts)),
            "d_feat": int(X.shape[1]),
            "averaged": bool(average),
        }

    @property
    def is_fitted(self) -> bool:
        return self._ridge is not None

    # -- online inference ----------------------------------------------------

    def predict_embedding(self, trial_or_epoch, ch_names=None) -> np.ndarray:
        """One epoch → predicted CLIP embedding ``(d_clip,)``."""
        if self._ridge is None:
            raise RuntimeError("decoder not calibrated — call fit_from_trials first")
        if hasattr(trial_or_epoch, "sfreq"):
            self.set_sfreq_in(trial_or_epoch.sfreq)
        epoch, ch = self._unpack(trial_or_epoch, ch_names)
        X = self.features(epoch[None, ...], ch)
        return self._ridge.predict(self._scaler.transform(X))[0]

    def retrieve(self, trial_or_epoch, ch_names=None, k=5):
        """Top-k gallery ids for one epoch.

        Returns ``[(image_id, cosine_score), ...]`` sorted by descending score.
        """
        pred = self.predict_embedding(trial_or_epoch, ch_names)
        gn = self.gallery / (np.linalg.norm(self.gallery, axis=1, keepdims=True) + 1e-12)
        pn = pred / (np.linalg.norm(pred) + 1e-12)
        sim = gn @ pn
        top = np.argsort(-sim)[:k]
        return [(int(self.gallery_ids[i]), float(sim[i])) for i in top]

    def evaluate(self, trials, ks=(1, 5, 10)):
        """Batch retrieval metrics over labelled trials (offline-equivalent)."""
        trials = list(trials)
        self.set_sfreq_in(trials[0].sfreq)
        ch_names = trials[0].ch_names
        epochs = np.stack([t.epoch for t in trials], axis=0)
        codes = np.array([t.code for t in trials], dtype=int)
        X = self.features(epochs, ch_names)
        pred = self._ridge.predict(self._scaler.transform(X))
        labels = np.array([self.id_to_row[int(c)] for c in codes], dtype=int)
        return topk_retrieval(pred, self.gallery, ks=ks, labels=labels)

    @staticmethod
    def _unpack(trial_or_epoch, ch_names):
        if hasattr(trial_or_epoch, "epoch"):              # a Trial
            return trial_or_epoch.epoch, trial_or_epoch.ch_names
        if ch_names is None:
            raise ValueError("pass ch_names when giving a raw epoch array")
        return np.asarray(trial_or_epoch), ch_names
