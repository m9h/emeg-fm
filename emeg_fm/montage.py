"""Channel-montage presets and a headless validator for live device sessions.

REVE is montage-agnostic: ``brain-bzh/reve-positions`` maps each electrode
*label* to a 3D coordinate, so a live session only has to hand REVE a list of
labels that resolve to known geometry. The cheapest way to fail a typo'd or
legacy label (``T3`` for ``T7``) is *before* the session — without a GPU or the
gated model — by checking each label against MNE's ``standard_1020`` geometry,
which is the same 10-20 system reve-positions derive from.

Contents
--------
* :data:`EMOTIV_EPOCFLEX32` — Emotiv EPOC Flex 32-channel saline-cap default
  layout, in standard 10-20 names.
* :func:`validate_montage` — resolve labels against MNE ``standard_1020``,
  applying the common legacy aliases (T3/T4/T5/T6), returning the canonical
  names or reporting which labels are unknown.
"""
from __future__ import annotations

# Emotiv EPOC Flex 32-channel saline default layout (standard 10-20 names).
# This is the gel/saline-cap default electrode set; all labels live in the
# 10-20 system so reve-positions can place every one.
EMOTIV_EPOCFLEX32 = [
    "Fp1", "Fp2", "AF3", "AF4",
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "PO3", "PO4",
    "O1", "Oz", "O2",
]

# Pre-2000s 10-20 names that newer standards renamed. MNE's standard_1020 only
# knows the modern names, so map the legacy ones through before lookup.
_LEGACY_ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}

PRESETS = {
    "epocflex32": EMOTIV_EPOCFLEX32,
}


def canonicalize(label: str) -> str:
    """Trim whitespace and apply legacy 10-20 aliases (T3→T7, …)."""
    key = str(label).strip()
    return _LEGACY_ALIASES.get(key.upper(), key)


def get_preset(name: str) -> list[str]:
    """Return a named montage preset (case-insensitive). Raises on unknown."""
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key not in PRESETS:
        raise KeyError(
            f"unknown montage preset {name!r}; have {sorted(PRESETS)}"
        )
    return list(PRESETS[key])


def channel_coords3d(ch_names, montages=("standard_1005", "GSN-HydroCel-128")):
    """Resolve channel labels to 3D xyz coords across a list of MNE montages.

    For LuMamba's topology-agnostic ``encode(x, channel_locations)``: MOABB uses
    10-20 names (``standard_1005``), HBN uses EGI ``E1..E128`` (``GSN-HydroCel-128``).
    Each label is matched case-insensitively (with legacy 10-20 aliases applied)
    against the montages in order; the first hit wins. Unresolvable labels are
    dropped, so callers must subset their data by the returned indices.

    Returns ``(keep_idx, coords)`` where ``keep_idx`` are the input positions that
    resolved (in order) and ``coords`` is the ``(len(keep_idx), 3)`` float array.
    """
    import numpy as np
    try:
        import mne
    except ImportError as e:  # pragma: no cover — ships with braindecode
        raise ImportError("channel_coords3d needs MNE; install 'mne'.") from e

    banks = []
    for mname in montages:
        pos = mne.channels.make_standard_montage(mname).get_positions()["ch_pos"]
        banks.append({k.upper(): v for k, v in pos.items()})

    keep, coords = [], []
    for i, label in enumerate(ch_names):
        canon = canonicalize(label).upper()
        for bank in banks:
            if canon in bank:
                keep.append(i)
                coords.append(bank[canon])
                break
    return keep, np.asarray(coords, dtype=np.float32).reshape(-1, 3)


def validate_montage(ch_names, *, standard="standard_1020", strict=True):
    """Check every label resolves to MNE ``standard`` geometry.

    Labels are matched case-insensitively after applying the legacy 10-20
    aliases (:func:`canonicalize`). This is a headless pre-flight — it needs
    only MNE, no GPU and no gated REVE checkpoint.

    Parameters
    ----------
    ch_names : sequence of str
        Montage labels to check (e.g. :data:`EMOTIV_EPOCFLEX32`).
    standard : str
        MNE standard montage to resolve against (default ``standard_1020`` —
        the system reve-positions derive from).
    strict : bool
        If True (default) raise ``ValueError`` listing any unresolved labels;
        if False return them instead.

    Returns
    -------
    resolved : list[str]
        Canonical label for each input (legacy aliases applied), in order.
    unresolved : list[str]
        Input labels with no match. Only returned when ``strict=False``.
    """
    try:
        import mne
    except ImportError as e:  # pragma: no cover — ships with braindecode
        raise ImportError(
            "validate_montage needs MNE (ships with braindecode); install 'mne'."
        ) from e

    pos = mne.channels.make_standard_montage(standard).get_positions()["ch_pos"]
    known = {name.upper(): name for name in pos}

    resolved, unresolved = [], []
    for label in ch_names:
        canon = canonicalize(label)
        hit = known.get(canon.upper())
        if hit is None:
            unresolved.append(str(label))
        else:
            resolved.append(hit)

    if unresolved and strict:
        raise ValueError(
            f"{len(unresolved)} montage label(s) not in MNE {standard}: "
            f"{unresolved}. Check for typos or legacy names (T3/T4/T5/T6 are "
            f"aliased automatically)."
        )
    if strict:
        return resolved
    return resolved, unresolved
