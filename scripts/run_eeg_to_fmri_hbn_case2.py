"""Case #2: cross-subject REVE EEG embedding -> resting CC200 FC on HBN.

The first EMPIRICAL EEG->fMRI result for the NeuroTechX white paper §5. Aligns the
cached REVE embeddings (2537 x 512, ids recovered + age-validated to 1e-15) with
C-PAC CC200 resting FC (pulled per subject) on the EEG∩C-PAC cohort, and tests
whether a subject's frozen EEG-FM embedding is cross-subject-linked to their BOLD
functional connectome, with the rigor the sweep prescribed:

  1. reliability ceiling  : run-1 vs run-2 FC split-half (bounds any cross-modal r)
  2. RAW                  : cross-subject CCA r1 + subject-pairing permutation null
  3. DECONFOUNDED         : same after removing age+age^2+sex+meanFD+site(release) --
                            the "identity-free" result; age dominates this 5-21
                            developmental cohort, so this is the headline
  4. incremental ridge R^2: predict FC-PCs from confounds vs confounds+EEG (KFold CV)

Caveat: HBN sibling/family ids are not in participants.tsv, so the permutation/CV are
subject-level, not family-blocked -- a known (mildly optimistic) refinement.

Run after pull_hbn_cc200.py writes /mnt/t9/hbn_cc200_fc.npz.
"""
import os
import sys

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(0)
N_PERM = 1000

# Principled PCA rank: reuse the ecosystem's Gavish-Donoho optimal hard threshold
# (smni-cmi clean.gavish_donoho_rank -- documented for exactly MIGP/PLSC/CCA dims,
# NOT ICA). Replaces the ad-hoc KX=KY=50. Falls back to the same omega(beta)*median
# rule inline if the cross-repo import is unavailable.
sys.path.insert(0, os.path.expanduser("~/Workspace/smni-cmi/src"))
try:
    from smni_cmi.clean import gavish_donoho_rank
except Exception:
    gavish_donoho_rank = None


def _gd_inline(s, m, n):
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return int(max(2, (s > omega * np.median(s)).sum()))


def gd_rank(M):
    """Gavish-Donoho optimal rank of a z-scored data matrix (prefers the smni-cmi
    function; identical omega(beta)*median rule inline as fallback)."""
    s = np.linalg.svd(StandardScaler().fit_transform(M), compute_uv=False)
    if gavish_donoho_rank is not None:
        try:
            return max(2, min(int(gavish_donoho_rank(s, M.shape[0], M.shape[1])),
                              len(s) - 1))
        except Exception:
            pass
    return min(_gd_inline(s, M.shape[0], M.shape[1]), len(s) - 1)

# --- load + align on the EEG order ------------------------------------------
emb = np.load("/mnt/t9/reve_hbn_emb.npz")
X_all, ages_all = emb["X"], emb["ages"]
eeg_ids = [l.strip() for l in open("/mnt/t9/hbn_reve_ids.txt") if l.strip()]
meta = np.load("/mnt/t9/hbn_reve_meta.npz", allow_pickle=True)
sex_by = dict(zip(meta["ids"], meta["sex"]))
rel_by = dict(zip(meta["ids"], meta["release"]))
age_by = dict(zip(eeg_ids, ages_all))
x_by = dict(zip(eeg_ids, X_all))

fc = np.load("/mnt/t9/hbn_cc200_fc.npz", allow_pickle=True)
fc_ids = list(fc["ids"])
fc_by = dict(zip(fc_ids, fc["fc"]))
fd_by = dict(zip(fc_ids, fc["meanfd"]))
r1_by = dict(zip(fc_ids, fc["fc_run1"]))
r2_by = dict(zip(fc_ids, fc["fc_run2"]))

cohort = [s for s in eeg_ids if s in fc_by]
print(f"aligned EEG∩FC cohort: {len(cohort)} subjects", flush=True)
X = np.array([x_by[s] for s in cohort])
Y = np.array([fc_by[s] for s in cohort], float)
R1 = np.array([r1_by[s] for s in cohort], float)
R2 = np.array([r2_by[s] for s in cohort], float)

# clean FC nans (dead CC200 ROIs -> nan edges): drop poorly-covered subjects, then
# impute remaining nan edges by the across-subject edge mean.
Y[~np.isfinite(Y)] = np.nan
keep = np.isnan(Y).mean(1) < 0.05
print(f"FC coverage: dropping {int((~keep).sum())} subjects with >5% nan edges",
      flush=True)
cohort = [s for s, k in zip(cohort, keep) if k]
X, Y, R1, R2 = X[keep], Y[keep], R1[keep], R2[keep]
col_mean = np.nan_to_num(np.nanmean(Y, 0))
ii = np.where(np.isnan(Y))
Y[ii] = np.take(col_mean, ii[1])
Y = np.nan_to_num(Y, nan=0.0)
R1 = np.nan_to_num(R1, nan=0.0)
R2 = np.nan_to_num(R2, nan=0.0)

age = np.array([age_by[s] for s in cohort])
fd = np.array([fd_by[s] for s in cohort], float)
fd = np.nan_to_num(fd, nan=np.nanmedian(fd))
sex = np.array([1.0 if str(sex_by.get(s, "")).strip().upper() in ("M", "1", "1.0")
                else 0.0 for s in cohort])
rels = sorted({str(rel_by.get(s, "NA")) for s in cohort})
rel = np.array([[1.0 if str(rel_by.get(s, "NA")) == r else 0.0 for r in rels]
                for s in cohort])
C = np.column_stack([age, age ** 2, sex, fd, rel])
print(f"confounds: age, age^2, sex, meanFD, {len(rels)} release dummies "
      f"-> C={C.shape}; age {age.min():.1f}-{age.max():.1f}", flush=True)

# --- principled PCA rank (Gavish-Donoho) instead of the ad-hoc 50 -----------
KX, KY = gd_rank(X), gd_rank(Y)
src = "smni-cmi gavish_donoho_rank" if gavish_donoho_rank else "inline GD rule"
print(f"[rank] Gavish-Donoho optimal rank ({src}): REVE KX={KX}, FC KY={KY} "
      f"(was ad-hoc 50)", flush=True)

# --- 1. reliability ceiling -------------------------------------------------
shr = np.array([np.corrcoef(R1[i], R2[i])[0, 1] for i in range(len(cohort))])
print(f"[ceiling] FC split-half reliability (run1 vs run2): "
      f"{np.nanmean(shr):.3f} +/- {np.nanstd(shr):.3f}", flush=True)

# --- helpers ----------------------------------------------------------------
def zt(M):
    return StandardScaler().fit_transform(M)


def residualize(M, Cc):
    return M - LinearRegression().fit(Cc, M).predict(Cc)


def first_cca_r(A, B):
    cca = CCA(n_components=1, max_iter=500).fit(A, B)
    a, b = cca.transform(A, B)
    return float(np.corrcoef(a[:, 0], b[:, 0])[0, 1])


def perm_p(A, B, true_r, n=N_PERM):
    null = np.array([first_cca_r(A, B[RNG.permutation(len(B))]) for _ in range(n)])
    return float((np.sum(null >= true_r) + 1) / (n + 1)), null


# --- 2. RAW cross-subject CCA ----------------------------------------------
Xp = PCA(KX, random_state=0).fit_transform(zt(X))
Yp = PCA(KY, random_state=0).fit_transform(zt(Y))
r_raw = first_cca_r(Xp, Yp)
p_raw, _ = perm_p(Xp, Yp, r_raw)
print(f"[RAW]      cross-subject CCA r1 = {r_raw:.3f}   perm-p = {p_raw:.4f}", flush=True)

# --- 3. DECONFOUNDED ("identity-free") CCA ----------------------------------
Xrp = PCA(KX, random_state=0).fit_transform(residualize(zt(X), C))
Yrp = PCA(KY, random_state=0).fit_transform(residualize(zt(Y), C))
r_dec = first_cca_r(Xrp, Yrp)
p_dec, _ = perm_p(Xrp, Yrp, r_dec)
print(f"[DECONF]   age+sex+FD+site-removed CCA r1 = {r_dec:.3f}   perm-p = {p_dec:.4f}",
      flush=True)

# --- 4. incremental ridge R^2 over the confound baseline (KFold CV) ---------
def cv_r2(Xin, Yt):
    cv = KFold(5, shuffle=True, random_state=0)
    pred = np.zeros_like(Yt)
    for tr, te in cv.split(Xin):
        sc = StandardScaler().fit(Xin[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(Xin[tr]), Yt[tr])
        pred[te] = m.predict(sc.transform(Xin[te]))
    return r2_score(Yt, pred, multioutput="uniform_average")


Yp_full = PCA(KY, random_state=0).fit_transform(zt(Y))
r2_base = cv_r2(C, Yp_full)
r2_full = cv_r2(np.hstack([C, Xp]), Yp_full)
print(f"[deltaR2]  FC-PCs from confounds R2={r2_base:.3f} -> +EEG R2={r2_full:.3f} "
      f"(dR2={r2_full - r2_base:+.3f})", flush=True)

np.savez("/mnt/t9/hbn_eeg_fmri_case2_result.npz",
         n=len(cohort), reliability=float(np.nanmean(shr)),
         r_raw=r_raw, p_raw=p_raw, r_dec=r_dec, p_dec=p_dec,
         r2_base=r2_base, r2_full=r2_full)
print("\n=== CASE #2 SUMMARY ===")
print(f"  n={len(cohort)}  FC reliability={np.nanmean(shr):.3f}")
print(f"  RAW  CCA r1={r_raw:.3f} (p={p_raw:.4f})")
print(f"  DECONF CCA r1={r_dec:.3f} (p={p_dec:.4f})  <- identity-free headline")
print(f"  ridge dR2 over confounds = {r2_full - r2_base:+.3f}")
print("  saved -> /mnt/t9/hbn_eeg_fmri_case2_result.npz")
