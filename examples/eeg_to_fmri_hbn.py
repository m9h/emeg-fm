#!/usr/bin/env python3
"""EEG -> fMRI cross-subject prediction on HBN -- the cases we can ABSOLUTELY test.

HBN has resting EEG (BIDS_EEG) AND resting fMRI (C-PAC) for the same subjects, but
in SEPARATE sessions. So the honest, testable question is **cross-subject between-
modality prediction**, NOT within-subject simultaneous alpha-BOLD fusion (HBN has
no simultaneous EEG-fMRI). Concretely: does a subject's EEG predict their resting
fMRI functional connectivity, across subjects, above a permutation null?

This is the empirical, HBN-grounded counterpart to Valdes-Sosa's MODEL-driven
EEG/fMRI fusion (neurojax reference_valdes_sosa_multimodal): instead of fitting the
generative neural-mass -> EEG + BOLD cascade, we test whether the two modalities are
statistically linked across subjects -- the first thing the data can answer.

Cohort: EEG subjects (cmi_bids_*) INTERSECT C-PAC func subjects (both resting).
  EEG  : s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_*/sub-<S>/eeg/...RestingState_eeg.set
  fMRI : s3://fcp-indi/data/Projects/HBN/CPAC_preprocessed/sub-<S>_ses-1/  (MNI BOLD)

Pipeline (each stage is a hook so the research cron can flesh it out incrementally):
  1. EEG feature  : REVE FM embedding of resting EEG  (reuse scripts/reve_brain_age.py
                    REVE adapter), or band-power / source connectivity.
  2. fMRI feature : resting FC -- parcellate the C-PAC BOLD (Schaefer/aparc) -> the
                    upper-triangle correlation vector.
  3. Cross-subject model : PLS / CCA / RidgeCV predicting fMRI-FC from EEG-embedding,
                    subject-grouped CV.
  4. Permutation null : shuffle the EEG<->fMRI subject pairing; the true cross-modal
                    canonical r / R^2 must exceed the null distribution.

Testable hypotheses (the research cron extends this list):
  H1. EEG REVE embedding predicts resting fMRI FC above chance (cross-subject CCA).
  H2. EEG alpha-band connectivity predicts fMRI DMN/visual-network FC
      (Valdes-Sosa: alpha<->BOLD positive frontal/thalamus, negative occipital).
  H3. EEG-derived brain-age agrees with fMRI-derived brain-age within subject.
"""
from __future__ import annotations

import argparse

import numpy as np

HBN_S3 = "s3://fcp-indi/data/Projects/HBN"


# --- cohort: EEG ∩ C-PAC (resting), the only subjects where EEG->fMRI is testable -
def intersect_eeg_fmri(eeg_ids: list[str], cpac_ids: list[str]) -> list[str]:
    """Bare NDAR ids present in BOTH the EEG and the C-PAC func cohorts."""
    norm = lambda s: s.replace("sub-", "").replace("_ses-1", "")
    return sorted(set(map(norm, eeg_ids)) & set(map(norm, cpac_ids)))


# --- fMRI feature: resting functional connectivity ---------------------------
def fc_vector(roi_timeseries: np.ndarray) -> np.ndarray:
    """Upper-triangle Fisher-z FC from an (n_rois, n_timepoints) C-PAC ROI series."""
    c = np.corrcoef(roi_timeseries)
    iu = np.triu_indices_from(c, k=1)
    return np.arctanh(np.clip(c[iu], -0.999, 0.999))


# --- cross-modal test: canonical correlation + permutation null --------------
def crossmodal_cca_permutation(X_eeg: np.ndarray, Y_fmri: np.ndarray, *,
                               n_perm: int = 1000, n_comp: int = 1, seed: int = 0):
    """First canonical correlation between EEG (X) and fMRI (Y) features, with a
    subject-shuffling permutation null. Returns (true_r, p_value, null)."""
    from sklearn.cross_decomposition import CCA

    def first_r(X, Y):
        cca = CCA(n_components=n_comp).fit(X, Y)
        xc, yc = cca.transform(X, Y)
        return float(np.corrcoef(xc[:, 0], yc[:, 0])[0, 1])

    rng = np.random.default_rng(seed)
    true_r = first_r(X_eeg, Y_fmri)
    null = np.array([first_r(X_eeg, Y_fmri[rng.permutation(len(Y_fmri))])
                     for _ in range(n_perm)])
    p = float((np.sum(null >= true_r) + 1) / (n_perm + 1))
    return true_r, p, null


# --- hooks the research cron fleshes out (heavy I/O / model extraction) -------
def eeg_reve_embedding(subject: str) -> np.ndarray:  # noqa: D401
    """REVE FM embedding of a subject's resting EEG. TODO: wire emeg_fm.adapters /
    scripts/reve_brain_age.py REVE extractor over the staged cmi_bids RestingState .set."""
    raise NotImplementedError("research-cron: wire the REVE resting-EEG embedding")


def fmri_resting_fc(subject: str) -> np.ndarray:
    """Resting FC vector from the subject's C-PAC BOLD. TODO: parcellate the MNI
    bandpassed BOLD (Schaefer) -> ROI series -> fc_vector()."""
    raise NotImplementedError("research-cron: wire the C-PAC resting-FC extraction")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eeg-ids", help="file: EEG subject ids (one per line)")
    p.add_argument("--cpac-ids", help="file: C-PAC subject ids (one per line)")
    p.add_argument("--n-perm", type=int, default=1000)
    a = p.parse_args()
    if not (a.eeg_ids and a.cpac_ids):
        print(__doc__)
        print("\nProvide --eeg-ids and --cpac-ids to compute the testable cohort.")
        return
    eeg = [l.strip() for l in open(a.eeg_ids) if l.strip()]
    cpac = [l.strip() for l in open(a.cpac_ids) if l.strip()]
    cohort = intersect_eeg_fmri(eeg, cpac)
    print(f"[cohort] EEG ∩ C-PAC resting: {len(cohort)} subjects -- the EEG->fMRI testbed")
    # X = np.array([eeg_reve_embedding(s) for s in cohort])
    # Y = np.array([fmri_resting_fc(s) for s in cohort])
    # r, pval, _ = crossmodal_cca_permutation(X, Y, n_perm=a.n_perm)
    # print(f"[H1] cross-subject EEG->fMRI canonical r={r:.3f}  p={pval:.4f}")


if __name__ == "__main__":
    main()
