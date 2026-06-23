# SimNIBS 4.1.0 install notes (tier-3 forward model)

How SimNIBS is made callable on this box (aarch64 / GB10), for `scripts/tier3_leadfield_prototype.py`.

## Install
SimNIBS 4.1.0 is already installed in a conda env — **no reinstall needed**:

    ENV=/home/mhough/miniforge3/envs/simnibs_test_v11
    $ENV/bin/python            # 3.12; `import simnibs` -> 4.1.0 (with the preload below)
    site-packages/simnibs      # package; CHARM atlas under segmentation/atlases

Two pip deps were missing for CHARM (SAMSEG) and were added into the env:

    $ENV/bin/pip install surfa numba

(The remaining pip warnings — mkl, tbb, PyQt5, pygpc — are non-fatal: perf libs, the GUI, and
uncertainty-quant; none are used by CHARM or the leadfield.)

## The libstdc++ preload (required)
`import simnibs` fails out of the box:

    petsc4py .../libpetsc.so.3.21: undefined symbol _ZTVN10__cxxabiv117__class_type_infoE

The env's `libstdc++.so.6.0.34` *has* the symbol, but the shipped `libpetsc.so` lists only
`RPATH=$ORIGIN/.` and does **not** name libstdc++ as NEEDED, so it never gets loaded. Fix = preload it
(LD_LIBRARY_PATH alone is not enough):

    LD_PRELOAD=$ENV/lib/libstdc++.so.6 LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python -c "import simnibs"
    # -> OK 4.1.0 ; petsc4py + PETSc 3.21.5 also import cleanly

The `bin/` console-scripts (`charm`, `simnibs`, …) do **not** set this up, so they crash. The tier-3
script therefore (a) calls the env's python directly, (b) sets LD_PRELOAD/LD_LIBRARY_PATH via
`_simnibs_env()`, and (c) runs CHARM as a module: `python -m simnibs.cli.charm`.

## Verified callability
- `python -m simnibs.cli.charm --help` — full help prints (CHARM/SAMSEG ready; atlas present).
- `simnibs.sim_struct.TDCSLEADFIELD` — fields subpath, pathfem, eeg_cap, field, anisotropy_type;
  defaults eeg_cap=`EEG10-10_UI_Jurak_2007.csv`, field=E, anisotropy_type=scalar.
- `simnibs.run_simnibs(lf)` — the v4 driver; on a fresh leadfield it reaches the mesh-load step
  (errors only on a missing m2m), confirming the wiring. Standard EEG caps live under
  `site-packages/simnibs/resources/ElectrodeCaps_MNI/`.

## ⚠ Root cause for the two crashes below: SimNIBS ships **x86_64** external binaries
`external/bin/linux/` contains precompiled executables — `mmg3d_O3`, `gmsh`, `meshfix`, `CAT_*` — and
**every one is x86-64** (`file` confirms), while this box is aarch64 (GB10). The aarch64-native parts
(the Python extensions, PETSc, CGAL meshing) work; the standalone binaries raise
`OSError: [Errno 8] Exec format error` the moment they're exec'd. Two of them sit on the head-mesh path
and bit us in sequence (CAT surfaces, then mmg). `gmsh` (interactive viewer, `open_in_gmsh`) and
`meshfix` (only used under `open_sulci_from_surf`/`pial`, which we disable) are **not** on the
segment→mesh→leadfield path, so they need no fix.

## CAT cortical-surface reconstruction crashes → disabled (blocker #1)
The first full run got all the way through SAMSEG (55/55 structures) and the MNI registration, then
**CHARM's cortical-surface step crashed** (it shells out to the x86_64 `CAT_*` binaries):

    [ simnibs ] INFO: Starting surface creation
    subprocess.CalledProcessError: .../run_cat_multiprocessing.py ... returned non-zero exit status 1

`run_cat_multiprocessing.py` is the CAT12-derived central-surface reconstruction (it shells out to the
x86_64 `CAT_*` binaries). **It is not needed for a volume-conduction leadfield:** the central GM
surfaces only (a) refine the segmentation (open sulci / fill GM) and (b) serve source-space modelling;
the TES/EEG FEM mesh is built from the tissue *volume* labels (`label_prep/tissue_labeling_upsampled.nii.gz`),
which SAMSEG produces fine.

**What does NOT work:** disabling surfaces in `charm.ini` via `surf=[]`/`pial=[]`. CHARM still invokes
`run_cat_multiprocessing` — now with empty `--surf`/`--pial` args — and argparse rejects them (exit 2).
(An earlier `charm_nosurf.ini` tried this and failed on every fresh subject.)

**What works = run CHARM in two steps** (see `scripts/tier3_leadfield_prototype.py::run_subject`):

    cd <subdir>
    python -m simnibs.cli.charm <subID> <T1> --forcerun   # SAMSEG + MNI reg write the label image,
                                                           # THEN the surface step crashes — tolerated
                                                           # (rc≠0, but tissue_labeling_upsampled.nii.gz exists)
    python -m simnibs.cli.charm <subID> --mesh             # FEM mesh from the label image (needs aarch64 mmg)

The runner suppresses `check=` on step 1 and proceeds iff the label image landed. Trade-off: no
surface-based refinement, so the GM/CSF boundary is slightly coarser (no opened sulci); first-order head
geometry — the conduction effect we're after — is unaffected. Re-enabling CAT surfaces (or grabbing them
from FreeSurfer via `charm --fs-dir`) is a quality follow-up.

## mmg3d crashes in the mesh step → rebuilt aarch64-native (blocker #2)
With surfaces off, the mesh got through CGAL meshing, relabeling, spike removal, and surface
reconstruction, then died at **"Improving Mesh Quality"**:

    OSError: [Errno 8] Exec format error: .../external/bin/linux/mmg3d_O3

`mmg3d_O3` is the tetrahedral-mesh optimiser. It is effectively mandatory here: the `[mesh]` setting
`optimize = false` ships off *because* SimNIBS "rel[ies] on MMG to optimize the tetrahedral mesh
instead" (`meshing.py:_run_mmg`, called twice in `create_mesh`; flags `-nosurf -nofem -hgrad -1 -rmc
-noinsert` are standard, version-stable mmg CLI). mmg is not on conda-forge, but it's a small C/CMake
project, so we built it native (gcc 13.3 + cmake 4.3 are present):

    git clone --depth 1 https://github.com/MmgTools/mmg.git /tmp/mmg_build
    cmake -S /tmp/mmg_build -B /tmp/mmg_build/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
    make -C /tmp/mmg_build/build -j8                       # -> build/bin/mmg3d_O3  (MMG3D 5.8.0, aarch64)

Install over the x86_64 stub, keeping a backup (all deps resolve via `ldd`; no special preload needed):

    BIN=$ENV/lib/python3.12/site-packages/simnibs/external/bin/linux
    cp -p $BIN/mmg3d_O3 $BIN/mmg3d_O3.x86_64.bak
    cp /tmp/mmg_build/build/bin/mmg3d_O3 $BIN/mmg3d_O3

After this, `charm <id> --mesh` completes the FEM head mesh. (If a future SimNIBS update overwrites the
binary, just rebuild and re-copy.)

## Known gaps (documented follow-ups, not blockers)
- **No `dwi2cond`** in this aarch64 build (not in `simnibs.cli`, no `bin/dwi2cond`) → the prototype
  uses **scalar (isotropic) conductivity**. DWI-anisotropic conductivity (`anisotropy_type='vn'`) is a
  follow-up. The age-varying head *geometry* from CHARM is the first-order conduction effect.
- **EEG cap** is the bundled 10-10 montage. An EGI-128 cap from `emeg_fm/montage.py`
  (GSN-HydroCel-128) via `simnibs.cli.prepare_eeg_montage` is the matching follow-up.

## Cost
CHARM ≈ 1–3 h/subject, leadfield ≈ 0.5–2 h/subject ⇒ ~1 CPU-day for the 5 prototype subjects.
Not run here (long compute). Launch:

    python scripts/tier3_leadfield_prototype.py --dry-run   # confirm inputs
    python scripts/tier3_leadfield_prototype.py             # full run (idempotent: skips existing m2m)
