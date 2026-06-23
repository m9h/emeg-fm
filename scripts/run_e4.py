"""E4 end-to-end: REVE EEG embeddings ⊗ tier-2 structural embedding → cross-modal spectrum, full and
**age-residualized** (conduction-removed). The real EEG↔sMRI structure–function shared-subspace number.
Run AFTER build_structural_embedding.py.

    python scripts/run_e4.py [--thresh 0.5]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from structural import reve_embeddings, assemble                      # noqa: E402
from cross_modal import cross_modal_spectrum, shared_subspace_summary, permutation_null  # noqa: E402

STRUCT = "/data/derivatives/volume_conduction/structural_emb.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default=STRUCT)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--n-perm", type=int, default=1000, help="permutation-null draws for top-rho")
    ap.add_argument("--out", default="results/e4_cross_modal.json")
    a = ap.parse_args()

    eeg_ids, X_eeg, ages = reve_embeddings()
    if not os.path.exists(a.struct):
        print(f"{a.struct} not found — run scripts/build_structural_embedding.py first.")
        return
    d = np.load(a.struct, allow_pickle=True)
    s_by_sub = {s: d["X"][i] for i, s in enumerate(list(d["ids"]))}
    E, S, y, kept = assemble(eeg_ids, X_eeg, ages, s_by_sub)          # E=EEG, S=structural, y=age

    full = cross_modal_spectrum(E, S)
    resid = cross_modal_spectrum(E, S, covariate=y)                   # remove the age/conduction term
    sf, sr = shared_subspace_summary(full, a.thresh), shared_subspace_summary(resid, a.thresh)
    # permutation null: CCA top-ρ is upward biased at d≈512, so calibrate against shuffled pairing
    nf = permutation_null(E, S, n_perm=a.n_perm)
    nr = permutation_null(E, S, n_perm=a.n_perm, covariate=y)
    print(f"n={len(kept)}  EEG d={E.shape[1]}  structural d={S.shape[1]}")
    print(f"full             ρ[:6]={np.round(full[:6], 2)}  n_strong={sf['n_strong']}  pr={sf['participation_ratio']:.1f}")
    print(f"age-residualized ρ[:6]={np.round(resid[:6], 2)}  n_strong={sr['n_strong']}  pr={sr['participation_ratio']:.1f}")
    print(f"\nperm-null (top-ρ vs shuffled pairing, {a.n_perm} draws):")
    print(f"  full : obs={nf['observed']:.3f}  null_mean={nf['null_mean']:.3f}  null_p95={nf['null_p95']:.3f}  p={nf['p_value']:.4f}")
    print(f"  resid: obs={nr['observed']:.3f}  null_mean={nr['null_mean']:.3f}  null_p95={nr['null_p95']:.3f}  p={nr['p_value']:.4f}")
    print(f"\ntop coupling drops {sf['top']:.2f}→{sr['top']:.2f} when age is removed; "
          f"residualized coupling is {'ABOVE' if nr['observed'] > nr['null_p95'] else 'WITHIN'} the null bias floor "
          f"(p={nr['p_value']:.4f}).")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"n": len(kept), "full": sf, "residualized": sr,
               "null_full": nf, "null_resid": nr,
               "rho_full": [float(v) for v in full[:20]],
               "rho_resid": [float(v) for v in resid[:20]]}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
