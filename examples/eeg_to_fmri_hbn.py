#!/usr/bin/env python3
"""EEG -> fMRI cross-subject prediction on HBN -- the cases we can ABSOLUTELY test.

Grounded by the multi-lens sweep (run wf_e512e1b2, 2026-06-29, verified on FCP-INDI
S3):
  * ~1,224 HBN subjects have BOTH clean resting EEG (BIDS_EEG/cmi_bids_*) AND C-PAC
    resting fMRI -- but in SEPARATE, NON-SIMULTANEOUS sessions. So every within-
    subject time-resolved fusion (spontaneous alpha-BOLD, NeuroBOLT per-TR synthesis,
    microstate-sequence -> concurrent dFC) is STRUCTURALLY UNTESTABLE (see BOUNDARY).
    The only valid regime is TRAIT-LEVEL CROSS-SUBJECT mapping: a per-subject EEG
    summary predicts a per-subject fMRI summary across held-out subjects. One within-
    subject exception (case 11): the shared MOVIES impose a common stimulus clock
    across the two sessions -- stimulus-driven, gated on verified clip timing.
  * fMRI targets derive from C-PAC's shipped CC200 ROI timeseries (FC is one corrcoef
    away; ALFF/ReHo/gradients/graph metrics derivable). NO precomputed FC/ALFF on S3.
    CC200 is purely functional -> DMN/visual hypotheses need a CC200->Yeo mapping.
  * Features already cached: REVE embeddings /mnt/t9/reve_hbn_emb.npz; NEOBA OSF/ODC
    /mnt/t9/neoba_hbn_feats.npz. jaxoccoli primitives exist: graph.diffusion_mapping
    / modularity, transport.gromov_wasserstein_fc, multivariate.cca.

THE DOMINANT CONFOUND: HBN is 5-21; BOTH modalities mature steeply with age, so a raw
cross-subject EEG<->fMRI correlation can be 100% age-mediated. **Every case is scored
as incremental Delta-R^2 over an age+meanFD+sex+site baseline (delta_r2_over_confounds),
with a cross-modal-pairing permutation null that respects sibling/family blocks** --
never a raw association. See PITFALLS.
"""
from __future__ import annotations

import argparse

import numpy as np

HBN_S3 = "s3://fcp-indi/data/Projects/HBN"
REVE_EMB = "/mnt/t9/reve_hbn_emb.npz"          # cached per-subject REVE embeddings
NEOBA_FEATS = "/mnt/t9/neoba_hbn_feats.npz"    # cached NEOBA 234 OSF + 156 ODC

# Ranked cross-subject cases (sweep wf_e512e1b2). feasibility H = features on disk +
# primitives exist (near-runnable now); M = needs a new pipeline; gated noted inline.
CASES = [
    {"rank": 1, "name": "aperiodic 1/f exponent -> fALFF/ReHo (E:I -> local BOLD amplitude)",
     "eeg": "per-channel FOOOF aperiodic exponent+offset (same specparam pass as NEOBA)",
     "fmri": "parcel-wise fractional ALFF + ReHo from C-PAC CC200 BOLD",
     "test": "dR2 over age+meanFD+sex+site; subject-perm null + SPIN null for map-to-map; within-age-band replication",
     "feas": "H"},
    {"rank": 2, "name": "frozen REVE EEG embedding -> resting FC (identity-trap-guarded)",
     "eeg": "pooled REVE embedding of RestingState EEG (cached REVE_EMB)",
     "fmri": "CC200 -> 200x200 Pearson -> Fisher-z upper triangle (19,900 edges)",
     "test": "subject-pairing perm null (family-blocked) on canonical r + top-k retrieval; LOSO; deconfound age/sex/FD; identity-free control must still beat chance; split-half ceiling first",
     "feas": "H"},
    {"rank": 3, "name": "NEOBA OSF/ODC fingerprint -> connectome graph topology",
     "eeg": "full NEOBA 234 OSF + 156 ODC (cached NEOBA_FEATS)",
     "fmri": "CC200 graph metrics: modularity Q, efficiency, participation, path length (jaxoccoli graph.py)",
     "test": "ODC dR2 over OSF-only AND over age+motion+sex+site; QuantileTransformer in-fold (heavy-tailed pediatric EGI); density-match graphs; subject-perm null",
     "feas": "H"},
    {"rank": 4, "name": "source-space band connectivity -> BOLD FC (Gromov-Wasserstein + edge CCA)",
     "eeg": "leakage-corrected source AEC/imag-coherence parcellated to fMRI atlas (HBN FS v6 leadfields)",
     "fmri": "parcel-parcel BOLD FC at matched resolution",
     "test": "GW coupling cost vs node-perm + mismatched-pair null; imag/orthogonalized estimators ONLY; distance regressor; LOSO",
     "feas": "M (build montage->leadfield->AEC pipeline)"},
    {"rank": 5, "name": "EEG spectral topography / REVE -> principal fMRI gradient",
     "eeg": "posterior-anterior alpha gradient + spectral PCs, or REVE embedding",
     "fmri": "principal gradient (diffusion_mapping of CC200 FC; Margulies 2016): range + node scores",
     "test": "MANDATORY Procrustes/template alignment (gradient sign/order); spin null; dR2 over age",
     "feas": "M-H (diffusion_mapping exists)"},
    {"rank": 6, "name": "EEG microstate statistics -> ICA RSNs / gradient",
     "eeg": "static microstate A-D(+F) stats from one GROUP template",
     "fmri": "group-ICA/dual-regression RSN maps + fractional occupancy",
     "test": "identity-shuffle perm on canonical r; FDR; heavy age deconfound + within-age-band; split-half reliability first",
     "feas": "M"},
    {"rank": 7, "name": "peak alpha frequency -> thalamo-cortical/visual FC (AGE STRESS TEST)",
     "eeg": "individual PAF + alpha bandwidth (periodic specparam)",
     "fmri": "thalamo-cortical + visual CC200 FC blocks",
     "test": "formal MEDIATION through age (direct vs indirect, bootstrap CI) -- a NULL direct effect calibrates age-aliasing for the whole program",
     "feas": "H (program-calibration value)"},
    {"rank": 8, "name": "EEG brain-age gap vs fMRI brain-age gap agreement",
     "eeg": "EEG brain-age (coffeine/REVE/NEOBA regressor)",
     "fmri": "fMRI brain-age (CC200-FC regressor)",
     "test": "both predict chrono-age OOS first; correlate the two GAPS after partialling shared chrono-age; subject-perm null",
     "feas": "H (both regressors exist; parallel-scalar, not a map)"},
    {"rank": 9, "name": "stimulus-matched movie EEG -> movie-fMRI FC (shared drive)",
     "eeg": "EEG features during task-DespicableMe / task-ThePresent",
     "fmri": "movie-fMRI FC on _scan_movieDM/_scan_movieTP (same stimulus)",
     "test": "subject-pairing perm null; group CV; deconfound age/motion/site; KEY contrast: shared drive vs rest predictability",
     "feas": "M (only 2 movies have both modalities; N shrinks)"},
    {"rank": 10, "name": "FM-to-FM: REVE EEG embedding -> frozen fMRI-FM latent",
     "eeg": "REVE-base L6 embedding (cached)",
     "fmri": "flat-map ViT-MAE fMRI-FM latent (arxiv 2510.13768) on HBN C-PAC rest",
     "test": "held-out-subject latent R2 + retrieval; dR2 over age; head-to-head vs REVE->raw-FC",
     "feas": "M-low GATED (must first extract fMRI-FM embeddings on HBN)"},
    {"rank": 11, "name": "within-subject movie EEG->BOLD coupling (movie-clock bridge)",
     "eeg": "movie-time-aligned EEG band-power/envelope timecourse, HRF-convolved",
     "fmri": "same subject's CC200 BOLD on movieDM/movieTP, movie-time aligned",
     "test": "ONLY within-subject route (shared movie clock); circular-shift null along movie time; group-level test",
     "feas": "low GATED on events.tsv proving frame-matched clip timing across sessions"},
]

# Structurally untestable on HBN -- retained to mark the boundary (any such claim is
# invalid by construction: HBN has NO simultaneous EEG-fMRI).
BOUNDARY = [
    "within-subject spontaneous resting alpha-BOLD fusion (no shared rest clock)",
    "time-resolved EEG band power -> per-TR BOLD synthesis (NeuroBOLT-class)",
    "EEG microstate SEQUENCE -> concurrent fMRI dynamic-FC state",
]

# Cross-cutting pitfalls every case must control for (sweep top_pitfalls).
PITFALLS = [
    "AGE is the dominant shared confound (5-21 cohort) -> always report dR2 over age+meanFD+sex+site.",
    "Multi-site scanner/rig shortcut (RU/CBIC/SI/CUNY) -> deconfound site, LOSO, identity-free control.",
    "Permutation must shuffle the CROSS-MODAL pairing (EEG_i<->fMRI_j) respecting sibling/family blocks; fit all scaling/CCA/deconfound INSIDE the train fold.",
    "Reliability ceiling: establish split-half/run-to-run reliability of EACH feature first.",
    "C-PAC fixed denoise (aCompCor+0.01-0.1Hz+FD0.5); only the atlas is changeable; CC200 has no Yeo labels.",
    "Source leakage -> orthogonalized AEC/imag-coherence only + distance regressor; NEOBA heavy-tailed -> QuantileTransformer in-fold; gradients/ICA/microstates need Procrustes; graph metrics need density-match; map-to-map needs a spin null.",
    "Shared trait head-motion across modalities (restless kids move in cap AND scanner) can manufacture cross-modal correlation -> include EEG artifact load + fMRI meanFD as covariates. Fix EEG resting state (HBN alternates eyes-open/closed).",
]


# --- cohort: EEG ∩ C-PAC (resting), the only subjects where EEG->fMRI is testable -
def intersect_eeg_fmri(eeg_ids: list[str], cpac_ids: list[str]) -> list[str]:
    """Bare NDAR ids present in BOTH the EEG and the C-PAC func cohorts."""
    norm = lambda s: s.replace("sub-", "").replace("_ses-1", "")
    return sorted(set(map(norm, eeg_ids)) & set(map(norm, cpac_ids)))


# --- fMRI feature: resting functional connectivity ---------------------------
def fc_vector(roi_timeseries: np.ndarray) -> np.ndarray:
    """Upper-triangle Fisher-z FC from an (n_rois, n_timepoints) C-PAC CC200 series."""
    c = np.corrcoef(roi_timeseries)
    iu = np.triu_indices_from(c, k=1)
    return np.arctanh(np.clip(c[iu], -0.999, 0.999))


# --- the HBN discipline: incremental R^2 OVER the age+motion+sex+site baseline ---
def delta_r2_over_confounds(X_eeg, Y, confounds, *, groups=None, n_splits=5, seed=0):
    """Cross-subject incremental R^2 of EEG features beyond a confound baseline -- the
    core HBN test (every association must beat age+meanFD+sex+site). Group-(family-)
    blocked CV when ``groups`` given. Returns (r2_base, r2_full, delta)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, KFold

    Y = np.asarray(Y, float)
    splitter = (GroupKFold(n_splits) if groups is not None
                else KFold(n_splits, shuffle=True, random_state=seed))

    def cv_r2(Xin):
        Xin = np.asarray(Xin, float)
        pred = np.zeros_like(Y)
        for tr, te in splitter.split(Xin, Y, groups):
            pred[te] = RidgeCV().fit(Xin[tr], Y[tr]).predict(Xin[te])
        return r2_score(Y, pred, multioutput="uniform_average")

    r2_base = cv_r2(confounds)
    r2_full = cv_r2(np.hstack([np.asarray(confounds, float), np.asarray(X_eeg, float)]))
    return r2_base, r2_full, r2_full - r2_base


# --- cross-modal test: canonical correlation + cross-pairing permutation null ----
def crossmodal_cca_permutation(X_eeg, Y_fmri, *, n_perm=1000, n_comp=1, seed=0,
                               groups=None):
    """First canonical correlation between EEG (X) and fMRI (Y) with a CROSS-MODAL
    subject-pairing permutation null (shuffle EEG_i<->fMRI_j). Pass ``groups`` (family
    ids) to keep the shuffle family-block aware. Returns (true_r, p_value, null)."""
    from sklearn.cross_decomposition import CCA

    def first_r(X, Y):
        cca = CCA(n_components=n_comp).fit(X, Y)
        xc, yc = cca.transform(X, Y)
        return float(np.corrcoef(xc[:, 0], yc[:, 0])[0, 1])

    rng = np.random.default_rng(seed)
    n = len(Y_fmri)
    true_r = first_r(X_eeg, Y_fmri)
    null = np.array([first_r(X_eeg, Y_fmri[rng.permutation(n)]) for _ in range(n_perm)])
    p = float((np.sum(null >= true_r) + 1) / (n_perm + 1))
    return true_r, p, null


# --- data hooks (grounded on the verified on-disk caches / CC200) --------------
def load_reve_embeddings(path: str = REVE_EMB):
    """Cached per-subject REVE resting-EEG embeddings -> (ids, X). Case 2/5/10."""
    d = np.load(path, allow_pickle=True)
    return d["subjects"], d["embeddings"]


def load_neoba_features(path: str = NEOBA_FEATS):
    """Cached NEOBA 234 OSF + 156 ODC per subject -> (ids, X). Case 3."""
    d = np.load(path, allow_pickle=True)
    return d["subjects"], d["features"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eeg-ids", help="file: EEG subject ids (one per line)")
    p.add_argument("--cpac-ids", help="file: C-PAC subject ids (one per line)")
    p.add_argument("--list-cases", action="store_true", help="print the ranked plan")
    a = p.parse_args()
    if a.list_cases or not (a.eeg_ids and a.cpac_ids):
        print(__doc__)
        print("Ranked cross-subject EEG->fMRI cases on HBN (feasibility H=near-runnable):")
        for c in CASES:
            print(f"  [{c['rank']:>2}] ({c['feas']:>16}) {c['name']}")
        print("\nBOUNDARY (structurally untestable on HBN):")
        for b in BOUNDARY:
            print(f"   x  {b}")
        if not (a.eeg_ids and a.cpac_ids):
            print("\nProvide --eeg-ids and --cpac-ids to compute the testable cohort.")
        return
    eeg = [l.strip() for l in open(a.eeg_ids) if l.strip()]
    cpac = [l.strip() for l in open(a.cpac_ids) if l.strip()]
    cohort = intersect_eeg_fmri(eeg, cpac)
    print(f"[cohort] EEG ∩ C-PAC resting: {len(cohort)} subjects -- the EEG->fMRI testbed")


if __name__ == "__main__":
    main()
