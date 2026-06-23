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
