# braindecode 1.6.1 — what it adds, and the issues we hit (upstream report)

*Tested 2026-07-02 in NGC PyTorch 26.06 (torch 2.13), braindecode 1.6.1 vs 1.5.2,
against the NeuroTechX-Atlas EEG-FM pipeline. Repro scripts: `/mnt/t9/bd_issue_probe.py`,
`/mnt/t9/bd_pos_probe.py`.*

## What 1.6.1 adds over 1.5.2 (the version our container ships)

**New foundation models** (`braindecode.models.*`, all `from_pretrained`-capable):
- **EEGDINO** — self-distillation FM (S/M/L, pretrained S/M weights) — *Aristimunha*
- **STEEGFormer** — ViT MAE EEG FM, Yang et al. ICLR 2026 — *Mounir*
- **MVPFormer** — multivariate parallel-attention iEEG FM, db4-wavelet encoder — *Aristimunha*
- **InterpolatedEEGPT** — montage-interpolation variant of EEGPT — *Guetschel*
- **TCFormer** — TCN+grouped-query-attention MI decoder (#1065)

**Montage-agnostic infra we actually care about** (alternative to the fragile `Interpolated*` path):
- **`ChannelMerger`** / `use_merger=True` — spatial Fourier merger (`FourierEmb`) for
  montage-agnostic spatial attention (#1076)
- **`pad_channels_collate()`** + **`set_return_ch_pos()`** / cached `ch_pos` — batch
  heterogeneous montages directly (#1066)
- **`sinusoidal_positional_encoding()`**, **`GatedLinearUnit`** (GEGLU/SwiGLU) primitives (#1078)
- `revision=` kwarg for HF datasets pinned to a commit (#1066)

**Registry/breaking:** `Interpolated*` moved to a separate `interpolated_models_dict` (#1093).
**Removed deprecated aliases:** `EEGNetv4`, `SleepStagerEldele2021`, `TSceptionV1`, `BNCI2014001` (#1045).
Various per-model fixes (SSTDPN axis, BatchNorm-at-bs=1, EEGSym conv, FilterBank float64).

**Bottom line for us:** 1.6.1 is a pure feature add (5 FMs + montage infra). It is **safe to
adopt** but **fixes none of the four issues below** — the failure set is byte-identical to
1.5.2. The changelog has *no* entry for the Interpolated-position crash, the BENDR checkpoint
skew, or the missing CBraMod/LUNA weights.

## Issues to report (all reproduce identically on 1.6.1 AND 1.5.2)

### 1. `Interpolated*` raises a cryptic LAPACK/SVD error on any channel lacking a montage position
`InterpolatedBIOT/BENDR/LaBraM.from_pretrained(..., chs_info=info["chs"])` where one or more
channels have no position (e.g. an uppercase name `FZ`/`CZ`/`FP1` that doesn't match MNE
`standard_1005`'s `Fz`/`Cz`/`Fp1`, or a non-EEG channel kept via `on_missing="ignore"`) →
```
numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
** On entry to DLASCL parameter number 4 had an illegal value   (LAPACK, from NaN coords)
```
**Proof it's positions, not montage size:** identical 19-ch set, only the *name casing* differs —
uppercase `["FP1",...,"FZ","CZ","PZ"]` → 14/19 positioned → **FAIL**; MNE-cased
`["Fp1",...,"Fz","Cz","Pz"]` → 19/19 → **PASS**. (Our earlier "small-montage" belief was wrong.)
**Ask:** validate `chs_info` and drop / error clearly on position-less channels before the
lstsq, instead of feeding NaN electrode coords into `numpy.linalg.lstsq`.
(Separately, ≤3 channels hits MNE `get_fitting_dig`: "at least 4 required" — a real limit, but
should surface as a braindecode-level message.)

### 2. `InterpolatedBENDR.from_pretrained` is completely unloadable — checkpoint↔library kwarg skew
```
TypeError: got an unexpected keyword argument 'n_chans_pretrained'
  @ braindecode/models/util.py:83  (track_model_init_kwargs → wrapped)
```
`n_chans_pretrained` appears **nowhere** in the 1.6.1 source, so it's a saved init-kwarg baked
into the published checkpoint `braindecode/braindecode-bendr` that current `BENDR.__init__` no
longer accepts. Fails on every montage. **Ask:** re-save the BENDR checkpoint's init-kwargs, or
have `from_pretrained` filter unknown kwargs against the current signature.

### 3. `CBraMod` / `LUNA` `from_pretrained` → HF 404 (no published weights)
```
RemoteEntryNotFoundError: Entry Not Found …/braindecode/CBraMod/resolve/main/pytorch_model.bin
RemoteEntryNotFoundError: Entry Not Found …/braindecode/LUNA/resolve/main/pytorch_model.bin
```
The model docstrings only show `repo_id="username/my-cbramod-model"` upload examples — no real
`braindecode/CBraMod` / `braindecode/LUNA` weights exist on the Hub. **Ask:** publish the
pretrained weights or document that these ship architecture-only (no `from_pretrained`).

## Our-side fix (independent of upstream)
Normalize channel names to MNE `standard_1005` casing before `set_montage` in the FM extractor
(`emeg_fm`/`atlas_bci_section.embed_eegfm`) → all 19 positions resolve → the `Interpolated*` path
works without patching braindecode. Applied 2026-07-02.
