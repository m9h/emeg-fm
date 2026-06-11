"""Layer-3 device bridge: clean raw consumer-headset EEG into REVE-ready epochs.

The streaming split has three layers:

    Layer-1  acquisition   — LSL inlet → ring buffer → time-locked epochs
                             (:mod:`emeg_fm.streaming`)
    Layer-3  device bridge — THIS module: remove mains, drift and a bad common
                             reference from noisy raw device samples
    Layer-2  REVE contract — resample 200 Hz → frozen per-channel scaler →
                             clamp ±15 (:class:`emeg_fm.alljoined.ReveInputNorm`)

Layer-3 runs *before* Layer-2, mirroring REVE's canonical order
(``reve.yaml`` via neuralset: notch → bandpass → resample → scale → clamp).
It owns only the device-idiosyncratic cleanup that a benchmark dataset already
had baked in but a live Emotiv wet cap does not:

    * **re-reference** (optional common-average) — kill shared cap noise;
    * **notch** the mains line (50 Hz EU / 60 Hz US) and optional harmonics;
    * **band-pass** ``[highpass, lowpass]`` with Nyquist-aware low-cut — the
      ``[0.5, 99.5]`` REVE was trained on, automatically dropping the lowpass
      when the device's Nyquist is below it (e.g. 128 Hz EPOC → highpass-only).

Resampling, scaling and clamping are deliberately *not* here — they are the
Layer-2 contract, fit-once on calibration so calibration and online epochs get
identical treatment. Keeping the two layers separate is what preserves the
stream==batch parity the decoder relies on.

Why it stays off the replay path
--------------------------------
Alljoined ships pre-filtered (band-passed + 60 Hz notched). Re-filtering it
would double-apply, so the replay smoke and the offline pipeline use **no
bridge**; the bridge is constructed only for the live Emotiv session. Filtering
is zero-phase (``sosfiltfilt`` / ``filtfilt``) and applied along the time axis,
so a batch ``(N, C, T)`` equals stacking single ``(C, T)`` transforms.

Short-epoch caveat
------------------
A 0.5 Hz high-pass has a long impulse response; zero-phase filtering of a short
epoch (≈1 s) has edge transients. For live use prefer filtering the longer
ring-buffer window (pre-roll from ``tmin`` plus post-roll) rather than the bare
stimulus window — :meth:`process` accepts any length, so feed it the widest
window the buffer holds.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt


class DeviceBridge:
    """Configurable Layer-3 front-end (re-reference → notch → band-pass).

    Parameters
    ----------
    sfreq : device sampling rate (Hz).
    highpass, lowpass : band-pass edges (Hz). ``lowpass`` is dropped (high-pass
        only) when it is ``>= 0.99 * Nyquist`` — exactly neuralset's behaviour,
        and what makes a 128 Hz EPOC (Nyquist 64) work with REVE's 99.5 cut.
        Either edge may be ``None`` to disable that side.
    notch : mains frequency to notch (Hz), or ``None`` to skip. 60 in the US,
        50 in the EU.
    notch_harmonics : also notch integer multiples of ``notch`` below Nyquist.
    notch_q : quality factor of the notch (higher = narrower).
    reref : ``None`` (default, matches reve.yaml), ``"average"`` for a common
        average reference, or a list of channel indices/labels whose mean is
        subtracted.
    order : Butterworth order for the band-pass.
    """

    def __init__(self, sfreq, *, highpass=0.5, lowpass=99.5, notch=60.0,
                 notch_harmonics=False, notch_q=30.0, reref=None, order=4):
        self.sfreq = float(sfreq)
        self.highpass = highpass
        self.notch = notch
        self.notch_harmonics = bool(notch_harmonics)
        self.notch_q = float(notch_q)
        self.reref = reref
        self.order = int(order)
        nyq = self.sfreq / 2.0

        # Drop a lowpass at/above Nyquist (neuralset semantics).
        lp = lowpass
        if lp is not None and lp >= 0.99 * nyq:
            lp = None
        self.lowpass = lp

        # Pre-design the band-pass SOS once.
        self._sos = self._design_bandpass(self.highpass, self.lowpass, nyq)

        # Pre-design notch (b, a) for the mains line + optional harmonics.
        self._notches = []
        if self.notch is not None:
            freqs = [self.notch]
            if self.notch_harmonics:
                k = 2
                while k * self.notch < 0.99 * nyq:
                    freqs.append(k * self.notch)
                    k += 1
            for f0 in freqs:
                if 0.0 < f0 < nyq:
                    b, a = iirnotch(w0=f0, Q=self.notch_q, fs=self.sfreq)
                    self._notches.append((b, a))

    @classmethod
    def for_emotiv(cls, sfreq=128.0, mains=60.0, **kw):
        """Preset for an Emotiv 32-ch wet cap (EmotivPRO/CyKit LSL).

        EPOC-class devices stream at 128 Hz (sometimes 256 Hz); at 128 Hz the
        99.5 Hz REVE lowpass sits above Nyquist and is automatically dropped to
        a high-pass-only band, leaving the mains notch to do the heavy lifting.
        """
        return cls(sfreq, notch=mains, **kw)

    def _design_bandpass(self, hp, lp, nyq):
        if hp is None and lp is None:
            return None
        if hp is not None and lp is not None:
            return butter(self.order, [hp, lp], btype="band", fs=self.sfreq,
                          output="sos")
        if hp is not None:
            return butter(self.order, hp, btype="high", fs=self.sfreq,
                          output="sos")
        return butter(self.order, lp, btype="low", fs=self.sfreq, output="sos")

    def _resolve_reref_idx(self, n_ch, ch_names=None):
        if self.reref is None or self.reref == "average":
            return None
        idx = []
        upper = ({str(n).strip().upper(): i for i, n in enumerate(ch_names)}
                 if ch_names is not None else None)
        for r in self.reref:
            if isinstance(r, str):
                if upper is None:
                    raise ValueError("string reref needs ch_names")
                idx.append(upper[str(r).strip().upper()])
            else:
                idx.append(int(r))
        return np.asarray(idx, dtype=int)

    def process(self, eeg: np.ndarray, ch_names=None) -> np.ndarray:
        """Clean ``eeg`` ``(C, T)`` or ``(N, C, T)``; returns the same shape.

        Order: re-reference → notch(es) → band-pass, all zero-phase along time.
        ``ch_names`` is only needed for a label-based ``reref`` list.
        """
        eeg = np.asarray(eeg, dtype=np.float64)
        squeeze = eeg.ndim == 2
        if squeeze:
            eeg = eeg[None, ...]
        if eeg.ndim != 3:
            raise ValueError(f"expected (C,T) or (N,C,T), got {eeg.shape}")
        n_ch, n_t = eeg.shape[-2], eeg.shape[-1]

        # 1. re-reference
        if self.reref == "average":
            eeg = eeg - eeg.mean(axis=-2, keepdims=True)
        elif self.reref is not None:
            ridx = self._resolve_reref_idx(n_ch, ch_names)
            eeg = eeg - eeg[:, ridx, :].mean(axis=-2, keepdims=True)

        # filtfilt/sosfiltfilt need padlen < signal length; clamp for short epochs.
        def _pad(default):
            return min(default, max(0, n_t - 1))

        # 2. notch(es)
        for b, a in self._notches:
            pad = _pad(3 * max(len(a), len(b)))
            eeg = filtfilt(b, a, eeg, axis=-1, padlen=pad)

        # 3. band-pass
        if self._sos is not None:
            pad = _pad(3 * self._sos.shape[0] * 2)
            eeg = sosfiltfilt(self._sos, eeg, axis=-1, padlen=pad)

        return eeg[0] if squeeze else eeg

    # Callable so it can be dropped in anywhere a (C,T)->(C,T) fn is expected.
    __call__ = process
