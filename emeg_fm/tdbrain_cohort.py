"""Bridge the TDBRAIN clinical EEG archive into the FMScope identity-trap audit
as a TRAIT cohort (one diagnosis per subject).

Unlike the MOABB motor-imagery cells (within-subject left/right *state*, layout
``"W,C"``), TDBRAIN carries exactly one psychiatric label per subject — the
*trait* regime where subject-axis erasure (LEACE) can genuinely destroy the
label, so identity-free balanced accuracy is diagnostic here rather than
saturating at the within-subject ceiling that made BA degenerate on the MI cells.

Default contrast: **MDD vs ADHD** — the best-powered pair in the cohort. The
released, labelled set is ≈378 MDD vs ≈253 ADHD (Synapse Wiki); the paper's
headline totals (van Dijk et al. 2022, Sci Data) are higher (≈426 / ≈271)
because ~30% of clinical/outcome labels are blinded for prospective validation,
so the downloadable N is the smaller set. Either way MDD > ADHD (~3:2). Healthy
controls are marginal (not a main released indication; ≈47 in the paper), so
MDD-vs-HC is heavily imbalanced and underpowered for LOSO. The diagnosis labels
live in the DUA-gated ``participants.tsv`` (Synapse syn26468893, downloadable
only via the brainclinics.com ORCID portal — Synapse itself does not yet grant
download), NOT in the EEG zips — point ``participants_tsv`` at it once staged.

Each subject -> one recording: ses-1 resting EEG (restEC by default), the 26
scalp channels (the 7 trailing EOG/ECG/EMG channels mislabeled as EEG in the
BrainVision header are dropped), segmented into fixed-length windows and
normalized with REVE's frozen-scaler+clamp contract
(:class:`emeg_fm.alljoined.ReveInputNorm`). Pair the returned cohort with
:class:`emeg_fm.fmscope_bridge.REVEExtractor` and feed ``audit_cell`` with
``cell_layout="T,C"``.

mne is imported lazily so this module loads without it (label parsing and tests
need no EEG stack).
"""

from __future__ import annotations

import glob
import os

import numpy as np

# The 26 scalp electrodes (standard 10-20), matching the brain-age TDBRAIN
# config. The 7 trailing header channels (VPVA/VNVB/HPHL/HNHR = EOG, Erbs/
# OrbOcc/Mass = ECG/EMG) are mislabeled as EEG and excluded.
TDBRAIN_SCALP_26 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]


def _sid_to_int(pid: str):
    """``"sub-19681349"`` -> ``19681349``; ``None`` if not parseable."""
    pid = (pid or "").strip()
    if pid.startswith("sub-"):
        pid = pid[4:]
    try:
        return int(pid)
    except ValueError:
        return None


def load_trait_labels(
    participants_tsv: str,
    classes=("MDD", "ADHD"),
    *,
    label_col: str = "indication",
) -> dict[int, int]:
    """Map ``subject_id (int) -> class index`` for the requested trait classes.

    Reads TDBRAIN's ``participants.tsv`` and keeps only subjects whose
    ``label_col`` (default ``"indication"``) exactly matches one of ``classes``
    (case-insensitive). The class index follows ``classes`` order, so
    ``classes=("MDD", "ADHD")`` gives MDD=0, ADHD=1.

    TDBRAIN has one row per *session*, so a subject can appear multiple times;
    the first row wins (ses-1), matching the brain-age "ses-1 only" convention.
    """
    import csv

    order = [c.strip().upper() for c in classes]
    idx = {c: i for i, c in enumerate(order)}
    out: dict[int, int] = {}
    with open(participants_tsv, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            val = (r.get(label_col) or "").strip().upper()
            if val not in idx:
                continue
            sid = _sid_to_int(r.get("participant_id", ""))
            if sid is None or sid in out:  # first (ses-1) row wins
                continue
            out[sid] = idx[val]
    return out


def _find_vhdr(bids_root: str, sid: int, task: str):
    """First BrainVision header for ``sub-<sid>`` and ``task`` (ses-1 preferred)."""
    pat = os.path.join(
        bids_root, f"sub-{sid}", "ses-*", "eeg",
        f"sub-{sid}_ses-*_task-{task}_eeg.vhdr",
    )
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None


def _load_windows(vhdr, channels, sfreq_out, clamp, epoch_sec):
    """Load one recording, pick the scalp montage, window it, REVE-normalize.

    Returns ``(windows (n_win, C, T) float32, ch_names)`` or ``(None, present)``
    if the file lacks the full montage or is too short for one window.
    """
    import mne
    from emeg_fm.alljoined import ReveInputNorm

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="error")
    present = [c for c in channels if c in raw.ch_names]
    if len(present) != len(channels):  # keep n_channels uniform across the cohort
        return None, present
    raw.pick(present)
    data = raw.get_data()                       # (C, T), Volts
    sfreq_in = float(raw.info["sfreq"])         # 500 Hz native
    win = int(round(epoch_sec * sfreq_in))
    n = data.shape[1] // win
    if n == 0:
        return None, present
    segs = np.stack([data[:, i * win:(i + 1) * win] for i in range(n)])  # (n,C,T)
    # Frozen per-recording scaler (contract-faithful: cross-window amplitude
    # survives so the ±clamp is meaningful) — mirrors the MOABB cohort builder.
    norm = ReveInputNorm(sfreq_out=sfreq_out, clamp=clamp).fit(segs, sfreq_in=sfreq_in)
    windows = norm.transform(segs, sfreq_in=sfreq_in).astype(np.float32)
    return windows, present


def build_tdbrain_cohort(
    bids_root: str = "/mnt/t9/tdbrain/bids",
    participants_tsv: str = "/mnt/t9/tdbrain/participants.tsv",
    *,
    classes=("MDD", "ADHD"),
    label_col: str = "indication",
    task: str = "restEC",
    channels=None,
    sfreq_out: float = 200.0,
    clamp: float = 15.0,
    epoch_sec: float = 10.0,
    max_per_class: int | None = None,
    subjects=None,
):
    """Build a FMScope ``InMemoryCohort`` from TDBRAIN as a trait cohort.

    Parameters
    ----------
    bids_root : str
        TDBRAIN BIDS tree (``sub-*/ses-*/eeg/*.vhdr``).
    participants_tsv : str
        DUA-gated label table (Synapse syn26468893). Must contain
        ``participant_id`` + ``label_col``.
    classes : tuple[str, ...]
        Trait classes to contrast, in label-index order (default MDD vs ADHD).
    label_col : str
        Diagnosis column — ``"indication"`` (presenting complaint, dense) or
        ``"formal Dx"`` (confirmed, sparse). Default ``"indication"``.
    task : str
        Resting condition: ``"restEC"`` (eyes-closed, canonical) or ``"restEO"``.
    channels : list[str] or None
        Scalp montage; ``None`` = :data:`TDBRAIN_SCALP_26`.
    sfreq_out, clamp, epoch_sec : float
        REVE input rate (200 Hz), clamp bound (±15), window length (10 s).
    max_per_class : int or None
        Cap recordings per class (e.g. to balance MDD against ADHD); ``None`` =
        use all. Subjects are taken in sorted id order.
    subjects : iterable[int|str] or None
        Restrict to these subject ids (``"sub-123"`` or ``123``); ``None`` = all
        labelled subjects.

    Returns
    -------
    InMemoryCohort
        ``recordings = [(subject_id, label, windows (n_win, C, T) f32), ...]``,
        one recording per subject, with ``.ch_names`` set for ``REVEExtractor``.
    """
    from fmscope.data.adapters import InMemoryCohort

    channels = list(channels) if channels is not None else list(TDBRAIN_SCALP_26)
    labels = load_trait_labels(participants_tsv, classes, label_col=label_col)
    if subjects is not None:
        keep = {_sid_to_int(s) if isinstance(s, str) else int(s) for s in subjects}
        labels = {k: v for k, v in labels.items() if k in keep}

    recordings: list[tuple[int, int, np.ndarray]] = []
    per_class = {i: 0 for i in range(len(classes))}
    used_c = None
    for sid in sorted(labels):
        lbl = labels[sid]
        if max_per_class is not None and per_class[lbl] >= max_per_class:
            continue
        vhdr = _find_vhdr(bids_root, sid, task)
        if vhdr is None:
            continue
        windows, _present = _load_windows(vhdr, channels, sfreq_out, clamp, epoch_sec)
        if windows is None or windows.shape[0] == 0:
            continue
        recordings.append((int(sid), int(lbl), windows))
        per_class[lbl] += 1
        used_c = windows.shape[1]

    if not recordings:
        raise ValueError(
            f"no usable TDBRAIN recordings for classes={classes} "
            f"(labelled subjects={len(labels)}, bids_root={bids_root}, task={task})"
        )
    cohort = InMemoryCohort(recordings, n_channels=used_c, sfreq=sfreq_out)
    cohort.ch_names = channels
    return cohort
