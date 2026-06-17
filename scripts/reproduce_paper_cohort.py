#!/usr/bin/env python
"""From-raw independent reproduction of an FMScope paper generalization cohort.

Verifies we can reproduce the paper's REVE identity-trap numbers end-to-end from
RAW EEG — not just from the bundled frozen features (task #80 already confirmed
the audit recomputes the paper's numbers byte-/near-exactly from those). Here we
download raw, preprocess, extract REVE ourselves, and audit.

Target (default): **ds004362** = PhysioNet eegmmidb, motor imagery (fist, T1/T2)
vs interleaved rest (T0), runs 4/8/12 — the paper's own MI cell (the analogue of
our MOABB MI flagship). Paper REVE: raw label-BA 0.728 -> erased 0.994 (pooled
subject×condition erasure).

IMPORTANT — this is a *directional* reproduction, not byte-exact:
  * The paper's ``preprocess_v2`` config (exact ICLabel prob thresholds, autoreject
    params) is not published, so our ICA+ICLabel+autoreject+CAR is best-effort.
  * REVE column only (no LaBraM/CBraMod extractor in emeg_fm).
Success = the pooled erasure shows the same DIRECTION (raw ~0.7 -> erased ~0.95+,
identity dominates and erasure lifts), reported against the paper number with the
gap stated, never tuned to match.

Runtime: Docker NGC PyTorch 26.05 + uv on /mnt/t9 (REVE GPU + gated checkpoint;
mne_icalabel + autoreject uv-installed in the launcher). See the .sbatch.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

# Paper LEACE references (reve), pooled subject×condition erasure.
PAPER_REVE = {
    "ds004362": {"label_ba_raw": 0.7278, "label_ba_erased": 0.9944,
                 "subj_ba_linear_pre": 0.9961, "n_subj": 60},
    "ds002893": {"label_ba_raw": 0.8561, "label_ba_erased": 1.0,
                 "subj_ba_linear_pre": 0.7649, "n_subj": 44},
    "ds004148": {"label_ba_raw": 0.6778, "label_ba_erased": 0.8167,
                 "subj_ba_linear_pre": 0.9815, "n_subj": 60},
}

ONEURO_DIR = os.environ.get("ONEURO_DIR", "/mnt/t9/openneuro")


def _preprocess_subject(raw, *, sfreq_out, l_freq=0.5, h_freq=45.0, notch=60.0):
    """Best-effort preprocess_v2: bandpass+notch -> ICA+ICLabel reject -> CAR.

    AutoReject is applied at the epoch stage by the caller. Returns the cleaned,
    CAR-referenced, montaged raw at its native sfreq (resampled later per-epoch).
    """
    import mne
    from mne.preprocessing import ICA

    raw.load_data()
    raw.notch_filter(notch, verbose="ERROR")
    raw.filter(l_freq, h_freq, verbose="ERROR")

    n_excl = -1  # -1 => ICA step skipped (deviation), >=0 => components removed
    try:
        from mne_icalabel import label_components
        # ICLabel expects an extended-infomax ICA on 1 Hz high-passed data.
        ica_raw = raw.copy().filter(1.0, None, verbose="ERROR")
        ica = ICA(n_components=0.99, method="infomax",
                  fit_params=dict(extended=True), max_iter="auto", random_state=97)
        ica.fit(ica_raw, verbose="ERROR")
        labels = label_components(ica_raw, ica, method="iclabel")
        artifact = {"eye blink", "muscle artifact", "heart beat",
                    "line noise", "channel noise"}
        excl = [i for i, (lab, p) in
                enumerate(zip(labels["labels"], labels["y_pred_proba"]))
                if lab in artifact and p > 0.8]
        ica.exclude = excl
        ica.apply(raw, verbose="ERROR")
        n_excl = len(excl)
    except ImportError:
        # PREPROCESSING DEVIATION: mne_icalabel unavailable -> no ICA artifact
        # removal. Loudly flagged; the run is then a further approximation of
        # preprocess_v2 (bandpass+notch+autoreject+CAR only).
        print("[DEVIATION] mne_icalabel not importable -> skipping ICA/ICLabel",
              flush=True)
    raw.set_eeg_reference("average", verbose="ERROR")
    return raw, n_excl


def _load_ds004362(n_subjects, sfreq_out, win_sec, max_per_class):
    """Download + preprocess eegmmidb; return (X, y, subj, ch_names).

    X: (n_windows, C, T) float32 at sfreq_out; y in {0=rest, 1=MI}; subj per win.
    """
    import mne
    from mne.datasets import eegbci
    from emeg_fm.paper_repro import eegbci_mi_vs_rest_label
    try:
        from autoreject import AutoReject
    except ImportError:
        AutoReject = None
        print("[DEVIATION] autoreject not importable -> skipping epoch rejection",
              flush=True)

    runs = [4, 8, 12]  # fist motor-imagery runs
    Xs, ys, subs = [], [], []
    ch_names = None
    done = 0
    for sid in range(1, 110):
        if done >= n_subjects:
            break
        try:
            fns = eegbci.load_data(sid, runs, update_path=True, verbose="ERROR")
            raws = [mne.io.read_raw_edf(f, preload=True, verbose="ERROR") for f in fns]
            raw = mne.concatenate_raws(raws, verbose="ERROR")
            eegbci.standardize(raw)
            raw.set_montage("standard_1005", verbose="ERROR")
            raw, n_excl = _preprocess_subject(raw, sfreq_out=sfreq_out)

            events, ev_id = mne.events_from_annotations(raw, verbose="ERROR")
            # Map annotation ids -> binary MI/rest; drop unknowns.
            keep = {code: eegbci_mi_vs_rest_label(desc)
                    for desc, code in ev_id.items()
                    if eegbci_mi_vs_rest_label(desc) is not None}
            ep = mne.Epochs(raw, events, event_id={d: c for d, c in ev_id.items()
                                                   if c in keep},
                            tmin=0.0, tmax=win_sec, baseline=None,
                            preload=True, verbose="ERROR")
            ep.resample(sfreq_out, verbose="ERROR")
            if AutoReject is not None:
                ar = AutoReject(n_jobs=1, verbose=False, random_state=11)
                ep = ar.fit_transform(ep, return_log=False)

            y = np.array([keep[c] for c in ep.events[:, 2]])
            X = ep.get_data(copy=True).astype(np.float32)
            # Cap per class per subject (match the paper's balanced trial budget).
            if max_per_class:
                sel = []
                for lbl in (0, 1):
                    ix = np.where(y == lbl)[0]
                    if len(ix) > max_per_class:
                        ix = ix[:max_per_class]
                    sel.append(ix)
                sel = np.sort(np.concatenate(sel))
                X, y = X[sel], y[sel]
            if len(np.unique(y)) < 2:
                print(f"[skip] subj {sid}: one class after cleaning", flush=True)
                continue
            Xs.append(X); ys.append(y); subs.append(np.full(len(y), sid))
            ch_names = ep.ch_names
            done += 1
            print(f"[ok] subj {sid}: {len(y)} win ({np.bincount(y)}), "
                  f"ICA excl={n_excl}  [{done}/{n_subjects}]", flush=True)
        except Exception as exc:  # noqa: BLE001 — isolate per-subject failures
            print(f"[fail] subj {sid}: {type(exc).__name__}: {exc}", flush=True)
    if not Xs:
        raise RuntimeError("no subjects survived preprocessing")
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(subs), ch_names)


def _openneuro_subjects(ds):
    """Download participants.tsv and return (dest, subject_id list)."""
    import openneuro
    dest = os.path.join(ONEURO_DIR, ds)
    os.makedirs(dest, exist_ok=True)
    try:
        openneuro.download(dataset=ds, target_dir=dest, include=["participants.tsv"])
    except Exception as e:  # noqa: BLE001
        print(f"  (participants.tsv: {type(e).__name__})", flush=True)
    pt = os.path.join(dest, "participants.tsv")
    subs = ([r["participant_id"] for r in csv.DictReader(open(pt), delimiter="\t")]
            if os.path.exists(pt) else [])
    return dest, subs


def _dl(ds, dest, inc):
    """Download one include prefix if not already present on /mnt/t9."""
    import openneuro
    full = os.path.join(dest, inc)
    if os.path.isdir(full) and any(os.scandir(full)):
        return True
    try:
        openneuro.download(dataset=ds, target_dir=dest, include=[inc])
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  (dl {inc}: {type(e).__name__})", flush=True)
        return False


def _load_ds002893(n_subjects, sfreq_out, win_sec, max_per_class):
    """Auditory P300 oddball: attended-auditory target vs standard, 1 s windows."""
    import mne
    from emeg_fm.paper_repro import ds002893_tone_label
    dest, subs = _openneuro_subjects("ds002893")
    Xs, ys, ss, ch_names, done = [], [], [], None, 0
    for sub in subs:
        if done >= n_subjects:
            break
        if not _dl("ds002893", dest, f"{sub}/eeg"):
            continue
        try:
            setf = glob.glob(f"{dest}/{sub}/eeg/*_eeg.set")
            evf = glob.glob(f"{dest}/{sub}/eeg/*_events.tsv")
            if not setf or not evf:
                print(f"[skip] {sub}: missing set/events", flush=True)
                continue
            raw = mne.io.read_raw_eeglab(setf[0], preload=True, verbose="ERROR")
            raw.pick("eeg", verbose="ERROR")
            raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
            raw, n_excl = _preprocess_subject(raw, sfreq_out=sfreq_out, notch=60.0)
            sf = raw.info["sfreq"]
            ev = []
            for r in csv.DictReader(open(evf[0]), delimiter="\t"):
                lab = ds002893_tone_label(r)
                if lab is None:
                    continue
                ev.append([int(round(float(r["onset"]) * sf)), 0, lab + 1])
            if not ev:
                print(f"[skip] {sub}: no attended-auditory tones", flush=True)
                continue
            epo = mne.Epochs(raw, np.array(ev),
                             event_id={"standard": 1, "target": 2},
                             tmin=0.0, tmax=win_sec, baseline=None,
                             preload=True, verbose="ERROR")
            epo.resample(sfreq_out, verbose="ERROR")
            y = epo.events[:, 2] - 1
            X = epo.get_data(copy=True).astype(np.float32)
            n1 = int((y == 1).sum())
            if n1 == 0 or (y == 0).sum() == 0:
                print(f"[skip] {sub}: single class", flush=True)
                continue
            # Subsample frequent (standard) to target count (paper protocol).
            sel = np.sort(np.concatenate([np.where(y == 0)[0][:n1],
                                          np.where(y == 1)[0]]))
            X, y = X[sel], y[sel]
            Xs.append(X); ys.append(y); ss.append(np.full(len(y), len(ss)))
            ch_names = epo.ch_names; done += 1
            print(f"[ok] {sub}: {len(y)} win ({np.bincount(y)}) excl={n_excl} "
                  f"[{done}/{n_subjects}]", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {sub}: {type(exc).__name__}: {exc}", flush=True)
    if not Xs:
        raise RuntimeError("no subjects survived preprocessing")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), ch_names


def _load_ds004148(n_subjects, sfreq_out, win_sec, max_per_class):
    """Eyes-closed vs mental arithmetic (session1), 5 s sliding windows."""
    import mne
    from emeg_fm.paper_repro import ds004148_task_label
    dest, subs = _openneuro_subjects("ds004148")
    Xs, ys, ss, ch_names, done = [], [], [], None, 0
    for sub in subs:
        if done >= n_subjects:
            break
        if not _dl("ds004148", dest, f"{sub}/ses-session1/eeg"):
            continue
        try:
            subX, subY = [], []
            for task in ("eyesclosed", "mathematic"):
                lbl = ds004148_task_label(task)
                vh = glob.glob(f"{dest}/{sub}/ses-session1/eeg/*task-{task}_eeg.vhdr")
                if not vh:
                    continue
                raw = mne.io.read_raw_brainvision(vh[0], preload=True, verbose="ERROR")
                raw.pick("eeg", verbose="ERROR")
                raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
                raw, _ = _preprocess_subject(raw, sfreq_out=sfreq_out, notch=50.0)
                epo = mne.make_fixed_length_epochs(raw, duration=win_sec,
                                                   overlap=0.0, preload=True,
                                                   verbose="ERROR")
                epo.resample(sfreq_out, verbose="ERROR")
                X = epo.get_data(copy=True).astype(np.float32)
                if max_per_class and len(X) > max_per_class:
                    X = X[:max_per_class]
                subX.append(X); subY.append(np.full(len(X), lbl))
                ch_names = epo.ch_names
            if len(subX) < 2:
                print(f"[skip] {sub}: missing a task", flush=True)
                continue
            X, y = np.concatenate(subX), np.concatenate(subY)
            Xs.append(X); ys.append(y); ss.append(np.full(len(y), len(ss)))
            done += 1
            print(f"[ok] {sub}: {len(y)} win ({np.bincount(y)}) "
                  f"[{done}/{n_subjects}]", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {sub}: {type(exc).__name__}: {exc}", flush=True)
    if not Xs:
        raise RuntimeError("no subjects survived preprocessing")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), ch_names


LOADERS = {
    "ds004362": _load_ds004362,
    "ds002893": _load_ds002893,
    "ds004148": _load_ds004148,
}
# Per-dataset default window length (s) — overridden only if --win-sec is passed.
WIN_SEC = {"ds004362": 4.0, "ds002893": 1.0, "ds004148": 5.0}


def _build_cohort(X, y, subj, ch_names, sfreq_out, clamp):
    from fmscope.data.adapters import InMemoryCohort
    from emeg_fm.alljoined import ReveInputNorm
    recordings = []
    for s in sorted({int(v) for v in subj}):
        m = subj == s
        norm = ReveInputNorm(sfreq_out=sfreq_out, clamp=clamp).fit(
            X[m], sfreq_in=sfreq_out)
        for lbl in (0, 1):  # one recording per (subject, condition) — paper grouping
            idx = np.where(m & (y == lbl))[0]
            if idx.size == 0:
                continue
            w = norm.transform(X[idx], sfreq_in=sfreq_out).astype(np.float32)
            recordings.append((int(s), int(lbl), w))
    coh = InMemoryCohort(recordings, n_channels=X.shape[1], sfreq=sfreq_out)
    coh.ch_names = ch_names
    return coh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ds004362", choices=list(PAPER_REVE))
    ap.add_argument("--n-subjects", type=int, default=60)
    ap.add_argument("--sfreq-out", type=float, default=200.0)
    ap.add_argument("--win-sec", type=float, default=0.0,
                    help="window length (s); 0 = per-dataset default "
                         "(ds004362 4.0, ds002893 1.0, ds004148 5.0)")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="cap windows/class/subject (0 = no cap)")
    ap.add_argument("--clamp", type=float, default=15.0)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--model", default="brain-bzh/reve-base")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir",
                    default=os.path.expanduser("~/dev/emeg-fm/results/paper_repro"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from emeg_fm.fmscope_bridge import REVEExtractor
    from fmscope.verdict.audit import _extract_features
    from fmscope.diagnostics.erasure import subject_axis_erasure
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    win_sec = args.win_sec or WIN_SEC[args.dataset]
    X, y, subj, ch_names = LOADERS[args.dataset](
        args.n_subjects, args.sfreq_out, win_sec, args.max_per_class or None)
    print(f"[cohort] {args.dataset}: {len(y)} windows, "
          f"{np.unique(subj).size} subj, {X.shape[1]} ch, device={device}", flush=True)

    coh = _build_cohort(X, y, subj, ch_names, args.sfreq_out, args.clamp)
    ext = REVEExtractor(ch_names=coh.ch_names, layer=args.layer, model_id=args.model)
    feats, sids, labels, _ = _extract_features(
        ext, coh, batch_size=args.batch_size, device=device)

    pooled = subject_axis_erasure(feats, sids, labels, cv="stratified-kfold")
    n = len(labels)
    per_trial = subject_axis_erasure(
        feats, sids, labels, window_recording=np.arange(n),
        rec_labels=labels, rec_pids=sids, cv="stratified-kfold")

    ref = PAPER_REVE[args.dataset]
    out = {
        "dataset": args.dataset, "model": args.model, "layer": args.layer,
        "n_subjects": int(np.unique(subj).size), "n_windows": int(n),
        "paper_reve": ref,
        "ours_pooled": {"raw": pooled.label_ba_raw, "erased": pooled.label_ba_erased,
                        "delta": pooled.label_ba_delta,
                        "subj_pre": pooled.subj_ba_linear_pre,
                        "subj_post": pooled.subj_ba_linear_post,
                        "interpretable": pooled.interpretable},
        "ours_per_trial": {"raw": per_trial.label_ba_raw,
                           "erased": per_trial.label_ba_erased,
                           "delta": per_trial.label_ba_delta,
                           "interpretable": per_trial.interpretable},
        "raw_gap_vs_paper": pooled.label_ba_raw - ref["label_ba_raw"],
        "erased_gap_vs_paper": pooled.label_ba_erased - ref["label_ba_erased"],
    }
    path = os.path.join(args.out_dir, f"reproduce_{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== from-raw reproduction vs paper (REVE, pooled) ===", flush=True)
    print(f"  paper  : raw={ref['label_ba_raw']:.3f} erased={ref['label_ba_erased']:.3f}", flush=True)
    print(f"  ours   : raw={pooled.label_ba_raw:.3f} erased={pooled.label_ba_erased:.3f} "
          f"delta={pooled.label_ba_delta:+.3f} subj {pooled.subj_ba_linear_pre:.2f}->"
          f"{pooled.subj_ba_linear_post:.2f}", flush=True)
    print(f"  gap    : raw={out['raw_gap_vs_paper']:+.3f} erased={out['erased_gap_vs_paper']:+.3f}", flush=True)
    print(f"  per-tr : raw={per_trial.label_ba_raw:.3f} erased={per_trial.label_ba_erased:.3f} "
          f"delta={per_trial.label_ba_delta:+.3f} interp={per_trial.interpretable}", flush=True)
    print(f"[done] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
