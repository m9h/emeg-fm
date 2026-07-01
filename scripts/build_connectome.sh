#!/usr/bin/env bash
# Build a 4S structural connectome for ONE subject from QSIPrep DWI (MRtrix3 on the DGX/aarch64).
#
# Note: MRtrix3 3.0.4's *Python wrappers* (dwi2response) break under Python 3.12 (the `imp` and
# `distutils` stdlib modules were removed); we run those under a Python 3.11 env, while the C++
# binaries (mrconvert/dwi2fod/tckgen/tck2connectome) use the system install untouched. The 4S atlas
# (MNI152NLin2009cAsym) is warped into the subject's DWI/T1w space with nitransforms (the QSIPrep
# from-MNI-to-T1w ANTs .h5), so no ANTs binary is needed. No ACT (avoids an FSL dependency).
# Per-step idempotent. Env: NSTREAM (default 1M), NTHREADS (default 6), RES (arg2, default 456).
set -euo pipefail
S="$1"; RES="${2:-456}"; NSTREAM="${NSTREAM:-1M}"; NT="${NTHREADS:-6}"
PY311=/home/mhough/miniforge3/envs/py311/bin/python   # runs the mrtrix python wrappers
PY=/home/mhough/miniforge3/bin/python3                 # numpy/nibabel/nitransforms
D=/data/derivatives/volume_conduction/qsiprep_dwi/$S
ATLAS=/data/derivatives/atlases/AtlasPack/tpl-MNI152NLin2009cAsym_atlas-4S${RES}Parcels_res-01_dseg.nii.gz
OUT=/data/derivatives/volume_conduction/connectomes/$S
mkdir -p "$OUT"; cd "$OUT"
[ -f "connectome_${RES}.csv" ] && { echo "[skip $S] connectome_${RES} present"; exit 0; }
dwi=$(find $D -name '*space-T1w_desc-preproc_dwi.nii.gz'|head -1)
bval=$(find $D -name '*space-T1w_desc-preproc_dwi.bval'|head -1)
bvec=$(find $D -name '*space-T1w_desc-preproc_dwi.bvec'|head -1)
mask=$(find $D -name '*space-T1w_desc-brain_mask.nii.gz'|head -1)
T1=$(find $D -name '*desc-preproc_T1w.nii.gz' ! -name '*space-MNI*'|head -1)
XFM=$(find $D -name '*from-MNI152NLin2009cAsym_to-T1w*xfm.h5'|head -1)
[ -n "$dwi" ] && [ -n "$XFM" ] || { echo "[wait $S] DWI/xfm not landed yet"; exit 2; }
# QC pre-flight: a degenerate (mostly-empty) preproc T1 gives a bad MNI->T1w warp -> empty connectome.
t1nz=$($PY -c "import numpy as np,nibabel as nib;print(int((np.asarray(nib.load('$T1').dataobj)>0).sum()))" 2>/dev/null || echo 0)
[ "${t1nz:-0}" -ge 3000000 ] || { echo "[qcfail $S] degenerate T1 (nonzero=$t1nz < 3M) -> skip"; touch "$OUT/.qcfail_t1"; exit 0; }

[ -f dwi.mif ]   || { echo "[$S 1] mrconvert"; mrconvert "$dwi" -fslgrad "$bvec" "$bval" dwi.mif -force -quiet; }
nsh=$(mrinfo dwi.mif -shell_bvalues 2>/dev/null | wc -w)
if [ "$nsh" -ge 3 ]; then
  [ -f wm.txt ]    || { echo "[$S 2] dwi2response dhollander"; $PY311 /usr/bin/dwi2response dhollander dwi.mif wm.txt gm.txt csf.txt -mask "$mask" -force -nthreads $NT -scratch ./d2r_scratch; }
  [ -f wmfod.mif ] || { echo "[$S 3] dwi2fod msmt_csd"; dwi2fod msmt_csd dwi.mif wm.txt wmfod.mif gm.txt gm.mif csf.txt csf.mif -mask "$mask" -force -quiet -nthreads $NT; }
else
  [ -f wm.txt ]    || { echo "[$S 2] dwi2response tournier"; $PY311 /usr/bin/dwi2response tournier dwi.mif wm.txt -mask "$mask" -force -nthreads $NT -scratch ./d2r_scratch; }
  [ -f wmfod.mif ] || { echo "[$S 3] dwi2fod csd"; dwi2fod csd dwi.mif wm.txt wmfod.mif -mask "$mask" -force -quiet -nthreads $NT; }
fi
[ -f atlas_native_${RES}.nii.gz ] || { echo "[$S 4] warp 4S atlas -> native"; $PY - "$ATLAS" "$XFM" "$T1" atlas_native_${RES}.nii.gz <<'PYEOF'
import sys, numpy as np, nibabel as nib
from nitransforms.io.itk import ITKCompositeH5
from nitransforms.nonlinear import DenseFieldTransform
from nitransforms.linear import Affine
from nitransforms.manip import TransformChain
a,x,r,o=sys.argv[1:5]
w,aff=ITKCompositeH5.from_filename(x)
m=TransformChain([DenseFieldTransform(w),Affine(aff.to_ras())]).apply(nib.load(a),reference=nib.load(r),order=0)
m.to_filename(o); print('      atlas_native',m.shape,'max',int(np.asarray(m.dataobj).max()))
PYEOF
}
[ -f tracks.tck ] || { echo "[$S 5] tckgen $NSTREAM"; tckgen wmfod.mif tracks.tck -seed_dynamic wmfod.mif -select "$NSTREAM" -mask "$mask" -force -quiet -nthreads $NT; }
echo "[$S 6] tck2connectome"; tck2connectome tracks.tck atlas_native_${RES}.nii.gz connectome_${RES}.csv -symmetric -zero_diagonal -force -quiet
echo "[$S 7] ESD"; $PY - connectome_${RES}.csv <<'PYEOF'
import sys, numpy as np
C=np.loadtxt(sys.argv[1], delimiter=',')
w=np.sort(np.abs(np.linalg.eigvalsh((C+C.T)/2)))[::-1]
p=len(w); N=min(50,p//2)
sl=np.polyfit(np.log(np.arange(1,N+1)),np.log(w[:N]+1e-9),1)[0]
print(f'      connectome {C.shape}  density {(C>0).mean():.3f}  streamlines {int(C.sum())}  top-mode {w[0]/w.sum():.3f}  top-{N} slope {sl:.2f}')
PYEOF
rm -rf d2r_scratch tracks.tck   # tracks.tck is large; connectome_*.csv is the deliverable
echo "DONE $S"
