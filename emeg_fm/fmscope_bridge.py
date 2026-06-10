"""Bridge emeg_fm's frozen EEG-FM adapters into the vendored FMScope audit.

FMScope (``fmscope/``, arXiv 2606.06647, "The Identity Trap in EEG Foundation
Models") audits whether a frozen model's accuracy rides on subject-identity
leakage. Its two input contracts (``fmscope.api``) are structural:

    FMExtractor  : callable (B, C, T) -> (B, embed_dim), with ``embed_dim``
    CohortAdapter: iter_recordings() -> (subject_id, label, windows)

This module supplies both for our stack: a ``REVEExtractor`` wrapping
:class:`emeg_fm.eeg_fm.REVEAdapter` (mean-pooled tokens), and
``alljoined_cohort`` which packs an Alljoined subject into an in-memory cohort
keyed by subject + stimulus super-category. Torch and the FM are imported
lazily so this module loads without them (the diagnostics run inside the SIF).
"""

from __future__ import annotations

import numpy as np


class REVEExtractor:
    """Adapt :class:`emeg_fm.eeg_fm.REVEAdapter` to FMScope's ``FMExtractor``.

    FMScope calls ``extractor(windows)`` with just the EEG batch, so the
    montage (``ch_names``) is fixed at construction. Returns mean-pooled
    ``(B, d_model)`` frozen features.
    """

    def __init__(self, ch_names, layer: int = 6, model_id: str = "brain-bzh/reve-base"):
        from emeg_fm.eeg_fm import REVEAdapter  # lazy: torch lives downstream

        self.ch_names = list(ch_names)
        self._adapter = REVEAdapter(layer=layer)
        self._loaded = self._adapter.load_model(model_id)
        self.embed_dim = int(self._adapter.output_dim)

    def __call__(self, x):
        if hasattr(x, "detach"):              # torch.Tensor → numpy
            x = x.detach().cpu().numpy()
        feats = self._adapter.extract_features(
            self._loaded,
            {"eeg": np.asarray(x), "electrode_names": self.ch_names,
             "ch_names": self.ch_names},
        )
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 3:                   # (B, P, D) → mean over tokens
            feats = feats.mean(axis=1)
        return feats


def alljoined_cohort(eeg_npy, stim_parquet, partition="stim_test",
                     label_col="super_category", sfreq_out=200.0):
    """Build a FMScope ``InMemoryCohort`` from one Alljoined subject.

    Each recording is one stimulus image's averaged epoch; ``subject_id`` is
    constant (single-subject) and ``label`` is the integer-encoded stimulus
    ``label_col`` — so the identity-vs-task diagnostics have a real task axis.
    Windows are preprocessed into REVE's 200 Hz z-scored/clamped distribution.
    """
    import pandas as pd
    from fmscope.data.adapters import InMemoryCohort
    from emeg_fm.alljoined import (
        load_subject_npy, average_by_image, preprocess_for_reve,
    )

    rec = load_subject_npy(eeg_npy)
    eeg, ch_names, sfreq = rec["eeg"], rec["ch_names"], rec["sfreq"]

    stim = pd.read_parquet(stim_parquet)
    stim = stim[stim["partition"] == partition]
    if "dropped" in stim.columns:
        stim = stim[~stim["dropped"].astype(bool)]
    stim = stim.reset_index(drop=True)
    if len(stim) != eeg.shape[0]:
        raise ValueError(
            f"trial/stim misalignment: {eeg.shape[0]} epochs vs {len(stim)} "
            f"kept '{partition}' rows — use experiment_metadata_categories.parquet."
        )

    image_files = [str(p) for p in stim["image_path"].tolist()]
    avg, uniq_files, _counts = average_by_image(eeg, image_files)
    proc = preprocess_for_reve(avg, sfreq_in=sfreq, sfreq_out=sfreq_out)

    # Label each unique image by its stimulus category (first row per image).
    first = stim.drop_duplicates("image_path").set_index("image_path")
    labels_raw = [first.loc[f, label_col] if label_col in stim.columns else 0
                  for f in uniq_files]
    classes = {c: i for i, c in enumerate(sorted(set(map(str, labels_raw))))}
    labels = [classes[str(c)] for c in labels_raw]

    recordings = [(0, int(lbl), proc[i][None, ...])    # (n_windows=1, C, T)
                  for i, lbl in enumerate(labels)]
    cohort = InMemoryCohort(recordings, n_channels=proc.shape[1], sfreq=sfreq_out)
    cohort.ch_names = ch_names                          # bridge to REVEExtractor
    return cohort
