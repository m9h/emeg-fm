"""Pure helpers for the FMScope paper from-raw reproduction (task #81).

The orchestration (download, ICA/ICLabel/autoreject preprocessing, REVE
extraction, the audit) lives in ``scripts/reproduce_paper_cohort.py`` and runs
in the Docker NGC container. The two pure, correctness-critical pieces live here
so they are unit-tested on CPU:

- :func:`eegbci_mi_vs_rest_label` — the ds004362 (PhysioNet eegmmidb) contrast:
  interleaved rest (T0) vs fist motor imagery (T1/T2).
- :func:`order_for_pooled_grouping` — order windows so each ``(subject, label)``
  is one contiguous run, so ``subject_axis_erasure``'s default
  ``_segment_recordings`` pools exactly one recording per ``(subject, condition)``
  — matching the paper's grouping (verified to reproduce its numbers, task #80).
"""
from __future__ import annotations

import numpy as np

# eegbci imagery runs (4/8/12): MI of fists. T0 = rest between trials.
_EEGBCI_MI_VS_REST = {"T0": 0, "T1": 1, "T2": 1}


def eegbci_mi_vs_rest_label(description) -> int | None:
    """Map an eegbci annotation description to the MI-vs-rest binary label.

    Returns 0 for rest (T0), 1 for fist motor imagery (T1/T2), ``None`` for
    anything else (so unknown annotations are dropped, not mislabelled).
    """
    return _EEGBCI_MI_VS_REST.get(str(description).strip(), None)


_DS002893_TONES = {"low_tone", "high_tone"}
_DS002893_ROLE = {"frequent_stimulus": 0, "infrequent_stimulus": 1}  # standard / target


def ds002893_tone_label(row) -> int | None:
    """ds002893 auditory-P300 contrast: attended-auditory standard vs target.

    The paper decodes attended-auditory tones, infrequent (target=1) vs frequent
    (standard=0). Keeps only attended auditory tone events; everything else
    (visual, unattended, button presses, cues) returns ``None`` (dropped).
    """
    if str(row.get("focus_modality", "")).lower() != "auditory":
        return None
    if str(row.get("attention_status", "")).lower() != "attended":
        return None
    if str(row.get("event_type", "")) not in _DS002893_TONES:
        return None
    return _DS002893_ROLE.get(str(row.get("task_role", "")), None)


_DS004148_TASKS = {"eyesclosed": 0, "mathematic": 1}


def ds004148_task_label(task) -> int | None:
    """ds004148 contrast: eyes-closed rest (0) vs mental arithmetic (1).

    Each is a separate continuous task recording; other tasks (eyesopen,
    memory, music) return ``None``.
    """
    return _DS004148_TASKS.get(str(task), None)


def order_for_pooled_grouping(subject, label) -> np.ndarray:
    """Permutation that sorts windows by ``(subject, label)`` (subject primary).

    Applying it makes every ``(subject, label)`` group a single contiguous run,
    so the default ``_segment_recordings`` in ``subject_axis_erasure`` pools one
    recording per ``(subject, condition)`` — the paper's grouping. Stable within
    a group (preserves trial order).
    """
    subject = np.asarray(subject)
    label = np.asarray(label)
    return np.lexsort((label, subject))  # primary key = subject, secondary = label
