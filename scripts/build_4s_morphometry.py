"""Per-parcel GM-volume morphometry from QSIPrep MNI GM probseg × the 4S atlas → the fine-p feature
matrix for the structural-spectrum (zeta) test.

No surfaces, no warp: QSIPrep already ships `space-MNI152NLin2009cAsym_label-GM_probseg` and the 4S
atlas ships in the same space, so per-subject per-parcel GM volume is just `sum(GM_probseg within
parcel)` (a VBM-style regional GM measure). Gives X of shape (~2136 subjects, n_parcels) at any 4S
resolution — the n>>p, high-p regime the aseg+DKT p=95 run couldn't reach.

    python scripts/build_4s_morphometry.py --res 456 [--limit N]
"""
import argparse
import glob
import os

import numpy as np
import nibabel as nib

QSIP = "/data/derivatives/volume_conduction/qsiprep_anat"
ATLAS = "/data/derivatives/atlases/AtlasPack"
OUT = "/data/derivatives/volume_conduction"


def _resample_atlas(atlas_data, atlas_aff, ref_img):
    """Nearest-neighbour resample the label atlas onto ref_img's grid (scipy only)."""
    from scipy.ndimage import affine_transform
    M = np.linalg.inv(ref_img.affine) @ atlas_aff
    return affine_transform(atlas_data.astype(np.float32), M[:3, :3], offset=M[:3, 3],
                            output_shape=ref_img.shape, order=0).round().astype(np.int32)


def parcel_gm(gm_img, atlas_data):
    """Total GM partial-volume per parcel id 1..n on the GM grid → (n,) vector."""
    gm = np.asarray(gm_img.dataobj, dtype=np.float32).ravel()
    a = atlas_data.ravel().astype(np.int64)
    n = int(atlas_data.max())
    return np.bincount(a, weights=gm, minlength=n + 1)[1:n + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="456", help="4S parcel count (156,256,...,1056)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    af = f"{ATLAS}/tpl-MNI152NLin2009cAsym_atlas-4S{args.res}Parcels_res-01_dseg.nii.gz"
    at_img = nib.load(af)
    atlas_native = np.asarray(at_img.dataobj)
    print(f"atlas {os.path.basename(af)}  shape {atlas_native.shape}  parcels {int(atlas_native.max())}")

    subs = sorted(glob.glob(f"{QSIP}/sub-*/anat/*space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz"))
    if args.limit:
        subs = subs[:args.limit]
    print(f"{len(subs)} subjects")

    atlas_on_grid, grid_key = None, None
    X, ids = [], []
    for i, p in enumerate(subs):
        sid = os.path.basename(p).split("_")[0]
        g = nib.load(p)
        key = (g.shape, tuple(np.round(g.affine.ravel(), 3)))
        if key != grid_key:                                    # (re)align atlas to this GM grid
            if g.shape == atlas_native.shape and np.allclose(g.affine, at_img.affine, atol=1e-3):
                atlas_on_grid = atlas_native.astype(np.int32)
            else:
                atlas_on_grid = _resample_atlas(atlas_native, at_img.affine, g)
            grid_key = key
        X.append(parcel_gm(g, atlas_on_grid)); ids.append(sid)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(subs)}", flush=True)

    X = np.array(X, float)
    out = f"{OUT}/morphometry_4s{args.res}.npz"
    np.savez(out, X=X, ids=np.array(ids), res=args.res)
    print(f"wrote {out}  X={X.shape}  (n_subjects x n_parcels)")


if __name__ == "__main__":
    main()
