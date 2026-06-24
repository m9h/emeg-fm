"""Build the DKT morphometry feature matrix from FastSurfer seg-only stats, align it to the REVE EEG
cohort + ages, and (optionally) run the tier-1 variance partition with morphometry as the anatomy block
— head-to-head against the block-pooled-probseg features on the *same* subjects.

    python scripts/build_morphometry.py [--compare]

Needs FastSurfer seg-only outputs (scripts/run_fastsurfer_seg.sh) under $SEG. With --compare it also loads
the block-pooled structural_emb.npz and runs both anatomy blocks on the common subject set, so the
redundant/EEG-unique numbers are directly comparable (only the anatomy features differ).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emeg_fm"))
from morphometry import parse_volume_stats, assemble_morphometry          # noqa: E402
from structural import reve_embeddings, assemble                           # noqa: E402
from variance_partition import variance_partition                          # noqa: E402

SEG = "/data/derivatives/volume_conduction/fastsurfer_seg"
STRUCT = "/data/derivatives/volume_conduction/structural_emb.npz"
OUT_NPZ = "/data/derivatives/volume_conduction/morphometry.npz"


def _vp_line(tag, E, A, y):
    r = variance_partition(E, A, y)
    print(f"  {tag:16s} n={len(y)} d={A.shape[1]:4d}  R2_eeg={r['r2_eeg']:.3f} R2_anat={r['r2_anat']:.3f} "
          f"R2_joint={r['r2_joint']:.3f}  redundant={r['redundant_fraction']:.3f} "
          f"eeg_unique={r['eeg_unique_fraction']:.3f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="also run block-pooled on the same subjects")
    ap.add_argument("--out", default=OUT_NPZ)
    a = ap.parse_args()

    eeg_ids, X_eeg, ages = reve_embeddings()
    stats_by_sub = {}
    for s in eeg_ids:
        p = f"{SEG}/{s}/stats/aseg+DKT.VINN.stats"
        if os.path.exists(p):
            stats_by_sub[s] = parse_volume_stats(p)
    print(f"morphometry available for {len(stats_by_sub)}/{len(eeg_ids)} EEG subjects")
    if not stats_by_sub:
        print("no morphometry yet — run scripts/run_fastsurfer_seg.sh first."); return
    Xm, regions, morph_ids = assemble_morphometry(stats_by_sub)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, X=Xm, regions=np.array(regions), ids=np.array(morph_ids), ages=ages[[eeg_ids.index(s) for s in morph_ids]])
    print(f"wrote {a.out}  (X {Xm.shape}, {len(regions)} regions)")

    morph_by_sub = {s: Xm[i] for i, s in enumerate(morph_ids)}
    print("\n=== tier-1 with DKT morphometry as anatomy A ===")
    E, A, y, kept = assemble(eeg_ids, X_eeg, ages, morph_by_sub)
    rm = _vp_line("morphometry", E, A, y)

    out = {"n_morph": len(morph_ids), "n_regions": len(regions), "tier1_morphometry": rm}
    if a.compare and os.path.exists(STRUCT):
        d = np.load(STRUCT, allow_pickle=True)
        bp_by_sub = {s: d["X"][i] for i, s in enumerate(list(d["ids"]))}
        common = [s for s in kept if s in bp_by_sub]                       # EEG ∩ morph ∩ block-pooled
        print(f"\n=== head-to-head on the SAME {len(common)} subjects (only anatomy features differ) ===")
        Em, Am, ym, _ = assemble(common, np.array([X_eeg[eeg_ids.index(s)] for s in common]),
                                 np.array([ages[eeg_ids.index(s)] for s in common]), morph_by_sub)
        Eb, Ab, yb, _ = assemble(common, np.array([X_eeg[eeg_ids.index(s)] for s in common]),
                                 np.array([ages[eeg_ids.index(s)] for s in common]), bp_by_sub)
        out["tier1_morphometry_common"] = _vp_line("morphometry", Em, Am, ym)
        out["tier1_blockpooled_common"] = _vp_line("block-pooled", Eb, Ab, yb)
    json.dump(out, open("results/tier1_morphometry.json", "w"), indent=2, default=float)
    print("\nwrote results/tier1_morphometry.json")


if __name__ == "__main__":
    main()
