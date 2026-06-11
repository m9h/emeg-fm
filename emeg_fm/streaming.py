"""Live acquisition bridge for realtime EEG→image decoding.

The deployment target is **NeuroTechX EEG-ExPy** for stimulus presentation.
EEG-ExPy's runtime contract is two Lab Streaming Layer (LSL) outlets:

    * an **EEG** stream  (type ``"EEG"``, ``C`` channels @ ``sfreq``) — produced
      by the device backend. For an Emotiv 32-ch wet cap this is EmotivPRO's
      LSL output (or CyKit); EEG-ExPy itself can also push muse/openbci/brainflow
      boards. Either way it is just an LSL "EEG" inlet to us.
    * a **Markers** stream (type ``"Markers"``, 1 channel, irregular) — pushed by
      the presentation script at each stimulus onset, value = an integer marker
      *code* that maps back to the shown image (see :mod:`emeg_fm.stimuli`).

This module owns nothing about the headset or PsychoPy. It subscribes to those
two LSL streams, keeps a short ring buffer of EEG samples, and turns every
marker into a fixed time-locked epoch ``(C, T)``. Those epochs feed
:class:`emeg_fm.decoder.StreamingReveDecoder` for the per-subject ridge
calibration and online retrieval.

``pylsl`` is imported lazily so the pure epoching logic (and the file-replay
source) is importable and unit-testable on a box with no LSL / no headset.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np


@dataclass
class Trial:
    """One time-locked epoch produced by an acquisition source.

    Attributes
    ----------
    code : int          — stimulus marker code (maps to an image id).
    epoch : np.ndarray  — ``(C, T)`` EEG, channel order = ``ch_names``.
    ch_names : list[str]— channel labels for ``epoch`` (passed to REVE as the
                          montage so reve-positions can map to 3D coords).
    sfreq : float       — sampling rate of ``epoch`` (Hz).
    onset_ts : float    — LSL timestamp of the marker (seconds).
    """

    code: int
    epoch: np.ndarray
    ch_names: list
    sfreq: float
    onset_ts: float


def select_channels(epoch: np.ndarray, src_names, want_names) -> np.ndarray:
    """Reorder/subset ``epoch`` (C, T) from ``src_names`` to ``want_names``.

    Channel-label match is case-insensitive (LSL/Emotiv labels are sometimes
    upper-cased, REVE/10-20 labels mixed-case). Raises if a requested channel
    is not present in the source stream.
    """
    src_upper = {str(n).strip().upper(): i for i, n in enumerate(src_names)}
    idx = []
    for n in want_names:
        key = str(n).strip().upper()
        if key not in src_upper:
            raise KeyError(
                f"requested channel {n!r} not in stream channels {list(src_names)}"
            )
        idx.append(src_upper[key])
    return epoch[np.asarray(idx, dtype=int), :]


class RingBuffer:
    """Fixed-capacity, timestamped per-sample buffer for one EEG stream.

    Stores the most recent ``maxlen_s`` seconds of samples and slices epochs
    by LSL timestamp. Pure numpy/collections — no LSL — so epoching is
    deterministic and unit-testable.
    """

    def __init__(self, n_channels: int, sfreq: float, maxlen_s: float = 30.0):
        self.n_channels = int(n_channels)
        self.sfreq = float(sfreq)
        self.maxlen = max(1, int(round(maxlen_s * sfreq)))
        self._ts: collections.deque = collections.deque(maxlen=self.maxlen)
        self._data: collections.deque = collections.deque(maxlen=self.maxlen)

    def push(self, timestamps, samples) -> None:
        """Append a chunk. ``samples`` is ``(n, C)``, ``timestamps`` is ``(n,)``."""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[None, :]
        timestamps = np.asarray(timestamps, dtype=np.float64).ravel()
        if samples.shape[0] != timestamps.shape[0]:
            raise ValueError(
                f"chunk mismatch: {samples.shape[0]} samples vs "
                f"{timestamps.shape[0]} timestamps"
            )
        if samples.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {samples.shape[1]}"
            )
        for t, s in zip(timestamps, samples):
            self._ts.append(float(t))
            self._data.append(s)

    @property
    def latest_ts(self) -> float:
        return self._ts[-1] if self._ts else float("-inf")

    @property
    def earliest_ts(self) -> float:
        return self._ts[0] if self._ts else float("inf")

    def __len__(self) -> int:
        return len(self._ts)

    def ready_for(self, onset_ts: float, tmax: float) -> bool:
        """Have we buffered samples through ``onset_ts + tmax``?"""
        return self.latest_ts >= onset_ts + tmax

    def epoch(self, onset_ts: float, tmin: float, tmax: float,
              max_gap_s: float | None = None) -> np.ndarray:
        """Slice the epoch ``[onset_ts+tmin, onset_ts+tmax)`` → ``(C, T)``.

        Selects every buffered sample whose timestamp falls in the window, in
        arrival order. Raises if the buffer does not yet extend through the
        window end (call :meth:`ready_for` first) or if the window start has
        already aged out of the ring. ``max_gap_s`` optionally rejects epochs
        with a dropout larger than that many seconds between consecutive
        samples (dropped LSL packets).
        """
        t0, t1 = onset_ts + tmin, onset_ts + tmax
        if self.latest_ts < t1:
            raise ValueError(
                f"buffer not ready: latest_ts={self.latest_ts:.4f} < {t1:.4f}"
            )
        if self.earliest_ts > t0:
            raise ValueError(
                f"window start {t0:.4f} aged out (earliest={self.earliest_ts:.4f})"
            )
        ts = np.fromiter(self._ts, dtype=np.float64, count=len(self._ts))
        sel = np.where((ts >= t0) & (ts < t1))[0]
        if sel.size == 0:
            raise ValueError(f"no samples in window [{t0:.4f}, {t1:.4f})")
        data = [self._data[i] for i in sel]
        if max_gap_s is not None and sel.size > 1:
            gaps = np.diff(ts[sel])
            if float(gaps.max()) > max_gap_s:
                raise ValueError(
                    f"dropout {gaps.max()*1e3:.0f} ms > {max_gap_s*1e3:.0f} ms "
                    f"in epoch window"
                )
        return np.stack(data, axis=1).astype(np.float32)   # (C, T)


class _EpochAssembler:
    """Shared marker→epoch state machine used by live and replay sources.

    Holds the ring buffer and a queue of pending markers; ``drain`` emits a
    :class:`Trial` for every pending marker whose post-onset window has fully
    arrived. Decoupled from LSL so the replay source reuses the exact epoching.
    """

    def __init__(self, ch_names, sfreq, tmin, tmax, montage=None,
                 buffer_s=30.0, max_gap_s=None):
        self.src_names = list(ch_names)
        self.sfreq = float(sfreq)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.montage = list(montage) if montage else None
        self.out_names = self.montage or self.src_names
        self.max_gap_s = max_gap_s
        self.buffer = RingBuffer(len(self.src_names), sfreq, buffer_s)
        self._pending: collections.deque = collections.deque()

    def push_eeg(self, timestamps, samples):
        self.buffer.push(timestamps, samples)

    def push_marker(self, code, onset_ts):
        self._pending.append((int(code), float(onset_ts)))

    def drain(self):
        """Yield Trials for every pending marker whose window is complete."""
        while self._pending and self.buffer.ready_for(self._pending[0][1], self.tmax):
            code, onset_ts = self._pending.popleft()
            try:
                ep = self.buffer.epoch(onset_ts, self.tmin, self.tmax,
                                       self.max_gap_s)
            except ValueError as e:                       # aged out / dropout
                print(f"[acq] skip marker {code}@{onset_ts:.3f}: {e}", flush=True)
                continue
            if self.montage is not None:
                ep = select_channels(ep, self.src_names, self.montage)
            yield Trial(code=code, epoch=ep, ch_names=self.out_names,
                        sfreq=self.sfreq, onset_ts=onset_ts)


class LSLAcquisition:
    """Subscribe to EEG-ExPy's EEG + Markers LSL streams and emit epochs.

    Parameters
    ----------
    tmin, tmax : epoch window relative to marker onset (s). Default −0.2 → 1.0
        matches the Alljoined epoching the offline smoke validated.
    montage : optional list of channel labels to subset/reorder the stream to
        (e.g. the exact 32 electrodes you placed on the Emotiv cap). If omitted,
        the stream's own channel labels are used and passed straight to REVE
        (REVE is montage-agnostic via reve-positions).
    eeg_type / marker_type / eeg_name / marker_name : LSL resolution hints.

    Use as a context manager; iterate :meth:`stream_epochs`.
    """

    def __init__(self, *, tmin=-0.2, tmax=1.0, montage=None,
                 eeg_type="EEG", marker_type="Markers",
                 eeg_name=None, marker_name=None,
                 buffer_s=30.0, max_gap_s=0.1, resolve_timeout=10.0):
        self.tmin, self.tmax = tmin, tmax
        self.montage = montage
        self.eeg_type, self.marker_type = eeg_type, marker_type
        self.eeg_name, self.marker_name = eeg_name, marker_name
        self.buffer_s, self.max_gap_s = buffer_s, max_gap_s
        self.resolve_timeout = resolve_timeout
        self._eeg_inlet = None
        self._marker_inlet = None
        self._asm = None
        self.sfreq = None
        self.ch_names = None

    # -- LSL plumbing (lazy import; not exercised by the unit tests) ---------

    @staticmethod
    def _resolve(pylsl, *, stype, name, timeout):
        if name:
            streams = pylsl.resolve_byprop("name", name, timeout=timeout)
        else:
            streams = pylsl.resolve_byprop("type", stype, timeout=timeout)
        if not streams:
            raise RuntimeError(
                f"no LSL stream found (type={stype!r}, name={name!r}) within "
                f"{timeout}s — is the device/EEG-ExPy presentation running?"
            )
        return streams[0]

    @staticmethod
    def _channel_labels(info):
        labels, ch = [], info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(ch.child_value("label") or f"ch{len(labels)}")
            ch = ch.next_sibling()
        return labels

    def connect(self):
        if self._eeg_inlet is not None:           # idempotent: already resolved
            return self
        import pylsl

        eeg_info = self._resolve(pylsl, stype=self.eeg_type, name=self.eeg_name,
                                 timeout=self.resolve_timeout)
        mrk_info = self._resolve(pylsl, stype=self.marker_type,
                                 name=self.marker_name,
                                 timeout=self.resolve_timeout)
        self._eeg_inlet = pylsl.StreamInlet(eeg_info, max_chunklen=0,
                                            recover=True)
        self._marker_inlet = pylsl.StreamInlet(mrk_info, recover=True)
        full = self._eeg_inlet.info()
        self.sfreq = full.nominal_srate()
        self.ch_names = self._channel_labels(full)
        self._asm = _EpochAssembler(
            self.ch_names, self.sfreq, self.tmin, self.tmax,
            montage=self.montage, buffer_s=self.buffer_s,
            max_gap_s=self.max_gap_s,
        )
        print(f"[acq] EEG '{full.name()}' {len(self.ch_names)}ch @ {self.sfreq}Hz; "
              f"markers '{mrk_info.name()}'", flush=True)
        return self

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        for inlet in (self._eeg_inlet, self._marker_inlet):
            try:
                if inlet is not None:
                    inlet.close_stream()
            except Exception:
                pass

    def stream_epochs(self, max_trials=None, idle_timeout=None):
        """Generator of :class:`Trial`. Pulls EEG + markers, emits completed epochs.

        ``max_trials`` stops after N epochs; ``idle_timeout`` (s) stops if no
        new marker arrives for that long (end of a calibration block).
        """
        import pylsl

        n = 0
        last_event = pylsl.local_clock()
        while True:
            samples, stamps = self._eeg_inlet.pull_chunk(timeout=0.0,
                                                          max_samples=1024)
            if stamps:
                self._asm.push_eeg(stamps, samples)
            m_sample, m_ts = self._marker_inlet.pull_sample(timeout=0.02)
            if m_sample is not None:
                code = int(m_sample[0])
                if code != 0:                              # 0 reserved = non-stim
                    self._asm.push_marker(code, m_ts)
                    last_event = pylsl.local_clock()
            for trial in self._asm.drain():
                yield trial
                n += 1
                if max_trials is not None and n >= max_trials:
                    return
            if idle_timeout is not None and not self._asm._pending and \
                    (pylsl.local_clock() - last_event) > idle_timeout:
                return


class FileReplaySource:
    """Replay a pre-epoched Alljoined subject as a :class:`Trial` stream.

    Lets the full streaming/decoder/retrieval path be validated end-to-end
    **without a headset** — it feeds the exact same ``Trial`` interface the
    live :class:`LSLAcquisition` emits, so ``run_streaming_decode.py --replay``
    exercises calibration + online decode against real EEG before a session.

    Each yielded epoch is one trial-averaged image (the offline smoke's unit),
    ``code`` = the unique-image index, ``ch_names`` = the subject's montage.
    """

    def __init__(self, eeg_npy, stim_parquet, partition="stim_test",
                 montage=None, max_images=None, seed=0):
        self.eeg_npy = eeg_npy
        self.stim_parquet = stim_parquet
        self.partition = partition
        self.montage = montage
        self.max_images = max_images
        self.seed = seed
        self.ch_names = None
        self.sfreq = None
        self.code_to_image = {}

    def stream_epochs(self, max_trials=None, idle_timeout=None):
        import pandas as pd
        from emeg_fm.alljoined import load_subject_npy, average_by_image
        import os

        rec = load_subject_npy(self.eeg_npy)
        eeg, ch_names, sfreq = rec["eeg"], rec["ch_names"], rec["sfreq"]
        self.ch_names = self.montage or ch_names
        self.sfreq = sfreq

        stim = pd.read_parquet(self.stim_parquet)
        stim = stim[stim["partition"] == self.partition]
        if "dropped" in stim.columns:
            stim = stim[~stim["dropped"].astype(bool)]
        stim = stim.reset_index(drop=True)
        if len(stim) != eeg.shape[0]:
            raise ValueError(
                f"trial/stim misalignment: {eeg.shape[0]} epochs vs {len(stim)} "
                f"kept '{self.partition}' rows."
            )
        image_files = [os.path.basename(p) for p in stim["image_path"].tolist()]
        avg, uniq, counts = average_by_image(eeg, image_files)

        order = np.arange(avg.shape[0])
        if self.max_images is not None and avg.shape[0] > self.max_images:
            order = np.argsort(-counts)[:self.max_images]
        rng = np.random.default_rng(self.seed)
        rng.shuffle(order)

        n = 0
        t = 0.0
        for code in order:
            ep = avg[code]
            if self.montage is not None:
                ep = select_channels(ep, ch_names, self.montage)
            self.code_to_image[int(code)] = str(uniq[code])
            yield Trial(code=int(code), epoch=ep.astype(np.float32),
                        ch_names=self.ch_names, sfreq=sfreq, onset_ts=t)
            t += (self.tmax_tmin())
            n += 1
            if max_trials is not None and n >= max_trials:
                return

    @staticmethod
    def tmax_tmin():
        return 1.2
