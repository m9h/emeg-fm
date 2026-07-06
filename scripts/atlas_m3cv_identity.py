"""NeuroTechX-Atlas identity-trap extension: M3CV (Huang et al., "M3CV: A
Multi-Subject, Multi-Session, and Multi-Task Database for EEG-Based
Biometrics", IEEE TIFS 2022 / Kaggle "eeg-biometric-competition").

M3CV is itself a biometric-verification competition -- decoding subject
identity IS its native task. That makes a naive "decode identity, then erase
identity, watch it collapse" test circular (LEACE guarantees exactly that by
construction for a linear probe). The complementary, non-circular use of this
cohort for our identity-trap program: M3CV also labels each epoch with one of
13 experimental CONDITIONS (task/state) per subject. We decode CONDITION
(the cognitive/task variable) under LOSO across the 95 enrollment subjects,
with subject IDENTITY as the nuisance axis erased via LEACE -- exactly the
same contract as every other atlas_*_extra loader (extract/loso/
within_subject_ba from atlas_cogneuro_section). This gives a large
(57,851-epoch, 95-subject, 13-class) independent multi-task cohort for the
identity-free leaderboard, distinct in kind from ERP/BCI/sleep/epilepsy.

Data: /data/datasets/eeg_fmri/m3cv/{Enrollment,Enrollment_Info.csv,...}.
Each epoch is a .mat file with a single 'epoch_data' array (65, 1000) --
65th channel is a reference/extra channel, dropped per the competition's own
Baseline/Fun_Preprocess.m (chan_select = 1:64). Sampling rate 250 Hz, 4 s
epochs, per Baseline/Demo.m ('fs', 250).

REVE is deliberately NOT supported for this dataset: M3CV ships no
channels.tsv/electrodes.tsv/channel-order documentation (unlike ds002718/
ds004883/ds003061/ds006018), so there is no honest way to map the 64 channel
indices to real 10-20 names for REVE's name-based lookup -- guessing a
Neuroscan/Curry montage order would silently mislabel channels. Same
engineering decision as SeizeIT2's proprietary wearable-electrode case:
classical-only, raise clearly rather than force a shaky mapping.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_cogneuro_section import loso, within_subject_ba  # noqa: E402
from atlas_bci_section import embed_classical  # noqa: E402

ROOT = os.environ.get("M3CV_ROOT", "/data/datasets/eeg_fmri/m3cv")
N_CONDITIONS = 13


def _read_info(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_m3cv_enrollment(max_subjects=None, max_per_cell=None, seed=42):
    """Enrollment split: 95 subjects, single session, 13 conditions, known
    subject id for every epoch -- the cleanest split for a condition-decode
    task since Calibration/Testing mix sessions and (for Testing) mask some
    subject ids by design of the verification competition."""
    import scipy.io as sio
    rng = np.random.RandomState(seed)
    rows = _read_info(os.path.join(ROOT, "Enrollment_Info.csv"))
    by_cell = {}
    for row in rows:
        key = (row["subject"], row["condition"])
        by_cell.setdefault(key, []).append(row)

    subs = sorted({row["subject"] for row in rows})
    if max_subjects:
        subs = subs[:max_subjects]
    subs = set(subs)

    all_X, all_y, all_subj = [], [], []
    for (subj, cond), cell_rows in by_cell.items():
        if subj not in subs:
            continue
        if max_per_cell and len(cell_rows) > max_per_cell:
            idx = rng.choice(len(cell_rows), max_per_cell, replace=False)
            cell_rows = [cell_rows[i] for i in idx]
        y_val = int(cond) - 1
        for row in cell_rows:
            mat_path = os.path.join(ROOT, "Enrollment", f"{row['EpochID']}.mat")
            if not os.path.exists(mat_path):
                continue
            d = sio.loadmat(mat_path)["epoch_data"][:64, :].astype(np.float32)
            all_X.append(d)
            all_y.append(y_val)
            all_subj.append(subj)
    X = np.stack(all_X)
    return X, np.asarray(all_y, int), np.asarray(all_subj)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["logbandpower"],
                    help="classical only -- see module docstring for why REVE is unsupported")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--max-per-cell", type=int, default=None,
                    help="cap epochs per (subject,condition) cell -- full cohort is 57,851 epochs")
    a = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")

    print(f"[m3cv-identity] model={a.model} max_subjects={a.max_subjects} "
          f"max_per_cell={a.max_per_cell}", flush=True)
    X, y, subj = load_m3cv_enrollment(a.max_subjects, a.max_per_cell)
    n_subj = len(set(subj.tolist()))
    print(f"  {len(y)} epochs, {n_subj} subjects, {N_CONDITIONS} conditions, "
          f"d_in={X.shape[1]}ch x {X.shape[2]}samp", flush=True)

    emb = embed_classical(X, sfreq=250.0)
    losoc = loso(emb, y, subj, N_CONDITIONS, erase_identity=False)
    idf = loso(emb, y, subj, N_CONDITIONS, erase_identity=True)
    wsub = within_subject_ba(emb, y, subj, N_CONDITIONS)
    print(f"\n[m3cv-identity] {a.model}: LOSO={losoc*100:.1f}% "
          f"within={wsub*100:.1f}% idfree={idf*100:.1f}% idΔ={(losoc-idf)*100:+.1f}")


if __name__ == "__main__":
    main()
