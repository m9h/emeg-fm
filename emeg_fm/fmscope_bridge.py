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

    def _inputs(self, x):
        if hasattr(x, "detach"):              # torch.Tensor → numpy
            x = x.detach().cpu().numpy()
        return {"eeg": np.asarray(x), "electrode_names": self.ch_names,
                "ch_names": self.ch_names}

    def __call__(self, x):
        feats = self._adapter.extract_features(self._loaded, self._inputs(x))
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 3:                   # (B, P, D) → mean over tokens
            feats = feats.mean(axis=1)
        return feats

    def all_layer_feats(self, x):
        """Mean-pooled per-block features in one forward pass.

        Returns ``(n_blocks, B, d_model)`` — REVE's native
        ``return_output=True`` all-layers list (embedding dropped). Used by
        :func:`reve_layer_probe` instead of fmscope's hook-based ``layer_probe``
        because REVE's blocks are bare ``ModuleList([attn, ff])`` with the
        residual add in the parent forward (no hookable block module).
        """
        return self._adapter.extract_all_layers(
            self._loaded, self._inputs(x), pool_tokens=True,
        )


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


# ----------------------------------------------------------------------------
# Diagnostics #3 (FOOOF aperiodic role) and #4 (layer-wise probe), REVE-native.
#
# fmscope.diagnostics.layer_probe hooks per-block submodules; REVE can't be
# hooked at block granularity (each block is a bare ModuleList([attn, ff]) with
# the residual add in the parent forward). So we reproduce its *scoring* exactly
# — StratifiedGroupKFold label probe + recording-level GroupKFold subject probe,
# StandardScaler + balanced LogisticRegression(C=1.0), balanced accuracy — but
# source features from REVE's native return_output=True all-layers list
# (one forward, every depth) via REVEExtractor.all_layer_feats.
# ----------------------------------------------------------------------------


def _label_probe_ba(feats, labels_arr, sids_arr, *, n_folds, seed):
    """Subject-grouped label balanced accuracy (mirrors fmscope label probe)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    try:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        bas = []
        for tr, te in cv.split(feats, labels_arr, groups=sids_arr):
            sc = StandardScaler().fit(feats[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
            clf.fit(sc.transform(feats[tr]), labels_arr[tr])
            pred = clf.predict(sc.transform(feats[te]))
            bas.append(balanced_accuracy_score(labels_arr[te], pred))
        return float(np.mean(bas))
    except ValueError:
        return float("nan")


def _subject_probe_ba(feats, sids_arr, rec_arr, *, n_folds, seed):
    """Recording-level re-identification balanced accuracy (mirrors fmscope)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    sub_rec_counts = {s: np.unique(rec_arr[sids_arr == s]).size
                      for s in np.unique(sids_arr)}
    keep = {s for s, n in sub_rec_counts.items() if n >= 2}
    if len(keep) < 2:
        return float("nan")
    mask = np.isin(sids_arr, list(keep))
    feats_k, sids_k, rec_k = feats[mask], sids_arr[mask], rec_arr[mask]
    try:
        n_groups = int(np.unique(rec_k).size)
        cv = GroupKFold(n_splits=min(n_folds, n_groups))
        bas = []
        for tr, te in cv.split(feats_k, sids_k, groups=rec_k):
            train_subs = set(sids_k[tr].tolist())
            te = te[np.isin(sids_k[te], list(train_subs))]
            if te.size == 0:
                continue
            sc = StandardScaler().fit(feats_k[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
            clf.fit(sc.transform(feats_k[tr]), sids_k[tr])
            pred = clf.predict(sc.transform(feats_k[te]))
            bas.append(balanced_accuracy_score(sids_k[te], pred))
        return float(np.mean(bas)) if bas else float("nan")
    except ValueError:
        return float("nan")


def _iter_windows(cohort, *, max_windows_per_recording=None):
    """Yield ``(rec_idx, sid, label, windows)`` with optional per-recording cap."""
    for rec_idx, (sid, label, windows) in enumerate(cohort.iter_recordings()):
        windows = np.asarray(windows, dtype=np.float32)
        if (max_windows_per_recording is not None
                and windows.shape[0] > max_windows_per_recording):
            idx = np.linspace(0, windows.shape[0] - 1,
                              max_windows_per_recording).astype(int)
            windows = windows[idx]
        yield rec_idx, sid, label, windows


def reve_layer_probe(extractor, cohort, *, batch_size=16, n_folds=3, seed=0,
                     max_windows_per_recording=None):
    """Depth-wise label + subject linear probe over REVE's transformer blocks.

    Mirrors :func:`fmscope.diagnostics.layer_probe.layer_probe` but uses
    ``extractor.all_layer_feats`` (one forward, all depths) instead of forward
    hooks. Returns ``{"per_depth": [{"depth", "depth_fraction",
    "label_ba_mean", "subject_ba_mean"}, ...], "n_layers", "elapsed_s",
    "n_windows", "n_subjects"}`` — depth ``k`` is REVE block ``k`` and
    ``depth_fraction = (k + 1) / n_blocks``.
    """
    import time

    feats_per_layer = None
    sids, labels, rec_ids = [], [], []
    t0 = time.time()
    for rec_idx, sid, label, windows in _iter_windows(
            cohort, max_windows_per_recording=max_windows_per_recording):
        for i in range(0, windows.shape[0], batch_size):
            layer_feats = extractor.all_layer_feats(windows[i:i + batch_size])
            if feats_per_layer is None:
                feats_per_layer = [[] for _ in range(layer_feats.shape[0])]
            for d in range(layer_feats.shape[0]):
                feats_per_layer[d].append(np.asarray(layer_feats[d],
                                                     dtype=np.float32))
        sids.extend([sid] * windows.shape[0])
        labels.extend([label] * windows.shape[0])
        rec_ids.extend([rec_idx] * windows.shape[0])

    if feats_per_layer is None:
        raise ValueError("cohort yielded no windows")
    sids_arr = np.asarray(sids)
    labels_arr = np.asarray(labels)
    rec_arr = np.asarray(rec_ids)
    n_layers = len(feats_per_layer)

    per_depth = []
    for d in range(n_layers):
        feats = np.concatenate(feats_per_layer[d], axis=0)
        per_depth.append({
            "depth": d,
            "depth_fraction": (d + 1) / n_layers,
            "label_ba_mean": _label_probe_ba(feats, labels_arr, sids_arr,
                                             n_folds=n_folds, seed=seed),
            "subject_ba_mean": _subject_probe_ba(feats, sids_arr, rec_arr,
                                                 n_folds=n_folds, seed=seed),
        })
    return {
        "per_depth": per_depth,
        "n_layers": n_layers,
        "elapsed_s": time.time() - t0,
        "n_windows": int(len(sids)),
        "n_subjects": int(np.unique(sids_arr).size),
    }


def fooof_role(extractor, cohort, *, sfreq=200.0, batch_size=16, n_folds=3,
               seed=0, max_windows_per_recording=None, mode="aperiodic_removed"):
    """1/f-role diagnostic: label/subject BA drop after aperiodic ablation.

    Re-extracts the extractor's pooled features on the original windows and on
    FOOOF-ablated windows (``mode="aperiodic_removed"`` by default), then scores
    both with the same label and subject probes. The paper's "1/f role" columns
    are ``state_drop_mean = orig.label_BA − ablated.label_BA`` and
    ``subject_drop_mean = orig.subject_BA − ablated.subject_BA`` — ready to pass
    as ``AuditConfig.oneoverf``.
    """
    from fmscope.preprocess.fooof_ablation import fooof_ablate

    orig_list, abl_list = [], []
    sids, labels, rec_ids = [], [], []
    for rec_idx, sid, label, windows in _iter_windows(
            cohort, max_windows_per_recording=max_windows_per_recording):
        ablated = fooof_ablate(windows, sfreq, mode=mode)
        for i in range(0, windows.shape[0], batch_size):
            orig_list.append(np.asarray(extractor(windows[i:i + batch_size]),
                                        dtype=np.float32))
            abl_list.append(np.asarray(extractor(ablated[i:i + batch_size]),
                                       dtype=np.float32))
        sids.extend([sid] * windows.shape[0])
        labels.extend([label] * windows.shape[0])
        rec_ids.extend([rec_idx] * windows.shape[0])

    sids_arr = np.asarray(sids)
    labels_arr = np.asarray(labels)
    rec_arr = np.asarray(rec_ids)
    orig = np.concatenate(orig_list, axis=0)
    abl = np.concatenate(abl_list, axis=0)

    orig_label = _label_probe_ba(orig, labels_arr, sids_arr, n_folds=n_folds, seed=seed)
    abl_label = _label_probe_ba(abl, labels_arr, sids_arr, n_folds=n_folds, seed=seed)
    orig_subj = _subject_probe_ba(orig, sids_arr, rec_arr, n_folds=n_folds, seed=seed)
    abl_subj = _subject_probe_ba(abl, sids_arr, rec_arr, n_folds=n_folds, seed=seed)

    return {
        "state_drop_mean": orig_label - abl_label,
        "subject_drop_mean": orig_subj - abl_subj,
        "orig_label_ba": orig_label,
        "ablated_label_ba": abl_label,
        "orig_subject_ba": orig_subj,
        "ablated_subject_ba": abl_subj,
        "mode": mode,
    }
