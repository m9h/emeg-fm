"""Bridge a MOABB BCI cohort into the vendored FMScope identity-trap audit.

MOABB (NeuroTechX) is the standard small-N BCI benchmark harness. Its
motor-imagery datasets are subject-rich and trial-poor — exactly where a frozen
EEG-FM's apparent task-skill is most at risk of riding on subject-identity
leakage, which is the failure mode FMScope (arXiv 2606.06647) is built to
surface.

:func:`build_moabb_cohort` runs one MOABB paradigm/dataset, groups the trials by
``subject × class`` into FMScope ``InMemoryCohort`` recordings (each recording
carries a single label, as the audit expects), and normalizes each window into
REVE's input distribution. It mirrors
:func:`emeg_fm.fmscope_bridge.alljoined_cohort`; pair the returned cohort with
:class:`emeg_fm.fmscope_bridge.REVEExtractor`.

MOABB is imported lazily so this module loads without it (the audit runs inside
the SIF where moabb is installed to a ``--target`` dir on ``sys.path``).
"""

from __future__ import annotations

import numpy as np


# Datasets known to mix subjects recorded at incompatible native sampling
# rates -- resampling each to the same target sfreq rounds to different
# per-trial sample counts, so combining them into one cohort array raises a
# broadcast/shape mismatch downstream (e.g. "operands could not be broadcast
# together with shapes (602,) (601,)"). MOABB's own PhysionetMI docstring:
# "Subject 88 was recorded at 128 Hz instead of 160 Hz like all other
# subjects. Loading subject 88 together with other subjects will cause
# errors in any paradigm."
KNOWN_INCOMPATIBLE_SUBJECTS = {
    "PhysionetMotorImagery": {88},
}


def _exclude_incompatible_subjects(dataset, subjects):
    """Drop subjects with a documented incompatible native sampling rate for
    this dataset, so the cohort's per-subject arrays stay a uniform shape."""
    code = getattr(dataset, "code", type(dataset).__name__)
    bad = KNOWN_INCOMPATIBLE_SUBJECTS.get(code, set())
    if not bad:
        return subjects
    dropped = sorted(set(subjects) & bad)
    if dropped:
        print(f"[moabb_cohort] {code}: excluding known-incompatible subject(s) "
              f"{dropped} (mismatched native sampling rate breaks resample-to-"
              f"common-length across the cohort)", flush=True)
    return [s for s in subjects if s not in bad]


def _is_epoch_concat_mismatch(exc) -> bool:
    """True if this ValueError is one of MNE/MOABB's single-call
    ``concatenate_epochs`` failure modes that ``_get_data_per_subject``
    fixes: differing channel counts ("nchan"), or differing per-trial epoch
    lengths across subjects (surfaces as a numpy broadcast error deep inside
    ``mne.epochs._concatenate_epochs``'s ``np.allclose(epochs.times, ...)``
    check -- real per-recording timing jitter, not necessarily a sampling-
    rate difference)."""
    msg = str(exc)
    return "nchan" in msg or "broadcast" in msg


def _crop_to_common_length(arrays):
    """Crop the time axis (last dim) of each array to the shortest one, so
    per-subject epoch arrays with a 1-sample (or few) length mismatch can
    still be concatenated. Trial count (axis 0) is untouched."""
    min_t = min(a.shape[-1] for a in arrays)
    lengths = sorted({a.shape[-1] for a in arrays})
    if len(lengths) > 1:
        print(f"[moabb_cohort] cropping per-subject epoch lengths to common "
              f"min {min_t} samples (lengths seen: {lengths}) -- per-recording "
              f"timing jitter, not a real signal difference", flush=True)
    return [a[..., :min_t] for a in arrays]


def _get_data_per_subject(paradigm, dataset, subjects):
    """Pool MOABB trials across subjects on their common-channel intersection.

    Fallback for datasets whose per-subject channel counts differ (which breaks
    MOABB's single-call ``concatenate_epochs``), or whose per-subject epoch
    lengths differ by a sample or two (real per-recording timing jitter).
    Channels are restricted to the intersection and reordered to a single
    stable order, and epoch lengths are cropped to the shortest, so the
    stacked array is consistent for REVE.
    """
    parts, common = [], None
    for s in subjects:
        ep, ys, m = paradigm.get_data(dataset, subjects=[s], return_epochs=True)
        chs = list(ep.ch_names)
        common = set(chs) if common is None else (common & set(chs))
        parts.append((ep, ys, m))
    if not parts or not common:
        raise ValueError("no common channels across subjects for this dataset")
    order = [c for c in parts[0][0].ch_names if c in common]  # stable order
    Xs, ys_all, subj_all = [], [], []
    for ep, ys, m in parts:
        ep = ep.copy().pick(order)
        Xs.append(ep.get_data())
        ys_all.append(np.asarray(ys))
        subj_all.append(np.asarray(m["subject"]))
    Xs = _crop_to_common_length(Xs)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys_all, axis=0)
    subj = np.concatenate(subj_all, axis=0)
    return X, y, subj, order


def build_moabb_cohort(
    dataset=None,
    paradigm=None,
    *,
    subjects=None,
    sfreq_out: float = 200.0,
    fmin: float = 0.5,
    fmax: float = 99.5,
    clamp: float = 15.0,
    normalize: bool = True,
):
    """Build a FMScope ``InMemoryCohort`` from any MOABB motor-imagery dataset.

    Defaults to BNCI2014_001 + LeftRightImagery (9 subjects, 22 ch, 2-class left/
    right hand) when ``dataset``/``paradigm`` are omitted, so existing callers
    keep working. Pass any MOABB ``dataset`` instance and a compatible
    ``paradigm`` instance to loop the identity-trap audit across the MOABB set
    (see ``scripts/moabb_identity_leaderboard.py``).

    Each ``(subject, class)`` group becomes one recording whose windows are the
    individual trials, normalized per-subject with REVE's frozen-scaler + clamp
    contract (:class:`emeg_fm.alljoined.ReveInputNorm`). The broadband
    ``fmin``/``fmax`` and ``resample`` are pushed into MOABB so the trials match
    REVE's 200 Hz / 0.5–99.5 Hz training pipeline rather than the narrowband
    8–32 Hz MI default.

    Parameters
    ----------
    dataset : moabb.datasets.base.BaseDataset or None
        MOABB dataset instance; ``None`` = ``BNCI2014_001()``.
    paradigm : moabb.paradigms.base.BaseParadigm or None
        Compatible paradigm instance. ``None`` = ``LeftRightImagery`` with the
        REVE-contract band/resample. If given, ``fmin``/``fmax``/``resample`` on
        the supplied paradigm are used as-is (caller's responsibility).
    subjects : list[int] or None
        MOABB subject ids; ``None`` = the dataset's full ``subject_list``.
    sfreq_out : float
        Target sampling rate (REVE: 200 Hz). Passed to MOABB ``resample``.
    fmin, fmax : float
        Bandpass edges handed to the default paradigm (REVE contract 0.5–99.5).
    clamp : float
        Post-z-score clamp bound (REVE contract ±15).

    Returns
    -------
    InMemoryCohort
        ``recordings = [(subject_id, label, windows (n_trials, C, T) f32), ...]``
        with ``.ch_names`` set for :class:`REVEExtractor`.
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery
    from fmscope.data.adapters import InMemoryCohort
    from emeg_fm.alljoined import ReveInputNorm

    if dataset is None:
        dataset = BNCI2014_001()
    if paradigm is None:
        paradigm = LeftRightImagery(fmin=fmin, fmax=fmax, resample=sfreq_out)
    if subjects is None:
        subjects = list(dataset.subject_list)
    subjects = _exclude_incompatible_subjects(dataset, subjects)

    try:
        epochs, y, meta = paradigm.get_data(
            dataset, subjects=subjects, return_epochs=True)
        ch_names = list(epochs.ch_names)
        # (n_trials, C, T). MOABB returns Volts; the per-channel z-score below is
        # scale-invariant, so no µV rescale is needed (clamp acts in std units).
        X = epochs.get_data()
        subj = np.asarray(meta["subject"])
    except ValueError as e:
        if not _is_epoch_concat_mismatch(e):
            raise
        # Some datasets vary channel count across subjects, or per-subject
        # epoch length by a sample or two (timing jitter), so MOABB's
        # single-call concatenate_epochs fails. Fetch per subject and pool on
        # the common-channel intersection / cropped-to-shortest length instead.
        X, y, subj, ch_names = _get_data_per_subject(paradigm, dataset, subjects)
    C = X.shape[1]
    classes = sorted({str(v) for v in y})           # e.g. ['left_hand','right_hand']
    label_map = {c: i for i, c in enumerate(classes)}
    labels = np.asarray([label_map[str(v)] for v in y])

    recordings: list[tuple[int, int, np.ndarray]] = []
    for s in sorted({int(v) for v in subj}):
        s_mask = subj == s
        if not s_mask.any():
            continue
        # Frozen per-subject scaler over all of this subject's trials (stands in
        # for neuralset's whole-recording StandardScaler) so cross-trial
        # amplitude survives and the clamp is meaningful. normalize=False emits
        # RAW windows instead, for FMs with their own input contract (e.g.
        # LuMamba: resample-256 + per-channel IQR, applied in its extractor).
        norm = (ReveInputNorm(sfreq_out=sfreq_out, clamp=clamp).fit(
            X[s_mask], sfreq_in=sfreq_out) if normalize else None)
        for lbl in range(len(classes)):
            idx = np.where(s_mask & (labels == lbl))[0]
            if idx.size == 0:
                continue
            windows = (norm.transform(X[idx], sfreq_in=sfreq_out)
                       if normalize else X[idx]).astype(np.float32)
            recordings.append((int(s), int(lbl), windows))

    cohort = InMemoryCohort(recordings, n_channels=C, sfreq=sfreq_out)
    cohort.ch_names = ch_names
    return cohort
