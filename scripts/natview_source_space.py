"""Source-space EEG->BOLD on natview -- the Valdes-Sosa nPCD step.

Instead of scalp occipital CHANNELS (which mix in non-visual sources), project the EEG
to the occipital CORTICAL SURFACE via an MNE minimum-norm inverse built on the natview
FreeSurfer recon, then predict occipital BOLD from occipital-SOURCE alpha power. This is
the principled asymmetric fusion (EEG-informed-fMRI done in source space) that the
scalp-band-power baseline approximates crudely.

Conductor: 3-layer SPHERE model (make_sphere_model) -- no FS binaries / watershed BEM
needed. Coreg: standard_1005 montage fitted to the subject's estimated fiducials.
Trigger-aligned (R128, shared clock). Reuses the tested emeg_fm.natview primitives and
the scalp pipeline's BOLD/eval helpers for a clean scalp-vs-source head-to-head.
Usage: natview_source_space.py [task]
"""
import os
import sys
import tempfile

import mne
import numpy as np
from scipy.signal import hilbert

from emeg_fm import natview as nv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natview_eeg_to_bold import (  # noqa: E402  reuse the scalp pipeline's helpers
    BANDS, NV, TR, bandpower_per_tr, clean_bold, evaluate, find_pair, mcflirt,
    volume_trigger_samples,
)

mne.set_log_level("ERROR")
OCC_ROIS = ("pericalcarine", "lateraloccipital", "cuneus", "lingual")  # aparc visual cortex
LAMBDA2 = 1.0 / 9.0  # SNR=3 regularization


def _relink(src, dst):
    if os.path.lexists(dst):
        os.unlink(dst)
    os.symlink(src, dst)


def prepare_subjects_dir(subjects_dir, subject):
    """MNE read_talxfm needs mri/T1.mgz; the natview recon staged only brain.mgz. Build a
    symlinked mirror under /mnt/t9 with T1.mgz -> brain.mgz (identical conformed geometry),
    never writing into /data NFS. Returns a usable subjects_dir."""
    if os.path.exists(f"{subjects_dir}/{subject}/mri/T1.mgz"):
        return subjects_dir
    sub = os.path.basename(subjects_dir.rstrip("/"))
    mirror, src, dst = f"/mnt/t9/natview_fs/{sub}", f"{subjects_dir}/{subject}", None
    dst = f"{mirror}/{subject}"
    os.makedirs(f"{dst}/mri", exist_ok=True)
    for name in ("surf", "label", "stats", "bem"):
        if os.path.isdir(f"{src}/{name}"):
            _relink(f"{src}/{name}", f"{dst}/{name}")
    for entry in os.listdir(f"{src}/mri"):
        _relink(f"{src}/mri/{entry}", f"{dst}/mri/{entry}")
    _relink(f"{src}/mri/brain.mgz", f"{dst}/mri/T1.mgz")
    print(f"[source] T1.mgz absent -> symlinked mirror SUBJECTS_DIR at {mirror}", flush=True)
    return mirror


def build_inverse(raw, subjects_dir, subject):
    """Sphere-model minimum-norm inverse + merged occipital cortical label."""
    raw.set_montage("standard_1005", match_case=False)
    raw.set_eeg_reference("average", projection=True)
    src = mne.setup_source_space(subject, spacing="oct6", subjects_dir=subjects_dir,
                                 add_dist=False)
    sphere = mne.make_sphere_model("auto", "auto", raw.info)
    # No watershed head surface -> coregister from fiducials only (MNI-estimated MRI fids
    # from the recon's Talairach xfm, matched to the montage's head-coord fiducials).
    fids = mne.coreg.get_mni_fiducials(subject, subjects_dir=subjects_dir)
    # template montage vs subject anatomy -> ~1 cm fiducial residual is expected; tolerate
    # it (label-averaged source power, not precise localization).
    trans = mne.coreg.coregister_fiducials(raw.info, fids, tol=0.05)
    fwd = mne.make_forward_solution(raw.info, trans, src, sphere, eeg=True, meg=False)
    cov = mne.make_ad_hoc_cov(raw.info)
    # depth weighting divides by per-source leadfield norm -> NaN for the sphere model's
    # zero-norm sources; disable it (depth is a BEM refinement, not needed here).
    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2, depth=None)
    labels = mne.read_labels_from_annot(subject, "aparc", subjects_dir=subjects_dir)
    occ = [lb for lb in labels if any(lb.name.startswith(r) for r in OCC_ROIS)]
    merged = occ[0]
    for lb in occ[1:]:
        merged = merged + lb
    print(f"[source] oct6 src, sphere BEM, occipital label = {len(occ)} aparc ROIs "
          f"({'+'.join(sorted({r for r in OCC_ROIS}))})", flush=True)
    return inv, merged


def source_bandpower(raw, inv, label, trig, n_tr):
    """Per-band occipital-SOURCE power, binned by R128 volume triggers."""
    feats = []
    for lo, hi in BANDS.values():
        rb = raw.copy().filter(lo, hi, verbose=False)
        stc = mne.minimum_norm.apply_inverse_raw(rb, inv, lambda2=LAMBDA2, method="dSPM",
                                                 label=label, verbose=False)
        env = np.abs(hilbert(stc.data.mean(0))) ** 2   # mean over occipital sources -> power
        feats.append(nv.bin_by_triggers(env, trig, n_tr))
    m = min(len(f) for f in feats)
    return np.column_stack([f[:m] for f in feats])


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "inscapes"
    pair = find_pair()
    if pair is None:
        sys.exit(f"no EEG+BOLD pair for task={task}")
    sub, ses, eeg_path, bold_path = pair
    subjects_dir = prepare_subjects_dir(f"{NV}/freesurfer/{sub}", ses)
    print(f"[pair] {sub} {ses} task-{task} | FS subjects_dir={subjects_dir} subject={ses}",
          flush=True)

    with tempfile.TemporaryDirectory(dir="/mnt/t9") as wd:
        print("[fmri] mcflirt ...", flush=True)
        d, par, affine = mcflirt(bold_path, wd)
    brain = nv.brain_mask(d)
    occ_mask = nv.occipital_mask(brain, affine)
    y_occ = clean_bold(d, occ_mask, par)
    print(f"[fmri] {d.shape[3]} TRs | occipital {int(occ_mask.sum())} vox", flush=True)

    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    trig = volume_trigger_samples(eeg_path, raw.info["sfreq"])
    occ_ch = nv.select_occipital_channels(raw.ch_names)
    print(f"[sync] {len(trig)} R128 triggers | first {trig[0] / raw.info['sfreq']:.1f}s",
          flush=True)

    print(f"\n=== natview {sub} {ses} task-{task}: SCALP vs SOURCE -> occipital BOLD ===")
    Xs = bandpower_per_tr(raw, trig, d.shape[3], occ_ch)   # scalp occipital channels
    evaluate("scalp occ-ch -> OCCIPITAL", Xs, y_occ)

    inv, occ_label = build_inverse(raw, subjects_dir, ses)
    Xv = source_bandpower(raw, inv, occ_label, trig, d.shape[3])   # occipital cortical source
    evaluate("SOURCE occ-cortex -> OCCIPITAL", Xv, y_occ)


if __name__ == "__main__":
    main()
