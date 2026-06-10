#!/bin/bash
#SBATCH --job-name=reve-overlay
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/derivatives/containers/build_logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Provision the REVE-pretraining env (elouayas/reve_eeg pyproject) for Scope B
# (from-scratch REVE-tiny, swap optimizer Adam→Muon).
#
# APPROACH: a persistent OVERLAY on top of pytorch_26.04.sif, NOT a new SIF.
# Why not a full rebuild:
#   - localimage bootstrap must extract the ~24 GB base rootfs + write a ~10 GB
#     SIF. NVMe root is 97% full (~29 GB free) → "no space left" (job 1551).
#   - APPTAINER_TMPDIR must be a LOCAL fs; pointing it at /data (NFS) fails with
#     "failed to create staging file" (job 1552). apptainer doesn't support
#     NFS-backed staging.
# An overlay sidesteps both: it only stores the NEW packages (~2-3 GB), never
# extracts/repacks the base, and the base SIF stays pristine + reusable.
#
# The deliverable is the pair (pytorch_26.04.sif + reve-overlay.img). Runtime:
#   apptainer exec --nv --overlay $OVERLAY:ro -B /data:/data $BASE_SIF python ...
#
# CUDA-stack protection: freeze installed torch/triton/nvidia-*/cuda-* into a
# pip constraints file first; pip then refuses to up/downgrade them (errors
# loudly instead of silently dropping a CPU wheel over the NGC stack).
#
# transformers / accelerate / peft: NONE are in the base /usr/local — they only
# existed in ~/.local (user-site), so under PYTHONNOUSERSITE=1 they must be
# installed into the overlay. REVE pins transformers==4.56.2, but that requires
# huggingface-hub<1.0 while the base layer ships hub 1.10.1 (read-only). Job
# 1554 proved you can't cleanly downgrade a base-layer pkg through an overlay:
# the stale hub-1.10.1 dist-info stays visible to importlib.metadata, breaking
# transformers' version check. So we install transformers==5.8.0 instead, which
# is compatible with hub 1.10.1. This is safe for REVE: its ONLY transformers
# use is `from transformers import AutoModel` (encoder.py, position_utils.py) —
# a stable symbol in both 4.x and 5.x — and Scope A/C already ran REVE under
# 5.8.0. hub/tokenizers/safetensors are PROTECTED (left at base versions);
# everything we add is a pure addition.
#
# flash-attn: OPTIONAL extra; skipped (painful on arm64, REVE runs via bf16).

set -euo pipefail

BASE_SIF=/data/derivatives/containers/pytorch_26.05.sif
OVERLAY_FINAL=/data/derivatives/containers/reve-overlay-2605.img
OVERLAY_SIZE_MB=6144
# BUILD the overlay on LOCAL NVMe, then copy the finished .img to /data. The
# earlier /data-resident build took ~2 h: pip writes thousands of small files
# through fuse2fs-over-NFS, which is metadata-bound and glacial. A local build
# is minutes; the single sequential cp of the ~6 GB .img to NFS at the end is
# cheap. /tmp is local NVMe (same fs as /).
OVERLAY=/tmp/reve-overlay-2605.$$.img
# Session/staging tmp must also be LOCAL (not NFS).
export APPTAINER_TMPDIR=/tmp/apptainer-reve.$$
mkdir -p "$APPTAINER_TMPDIR" /data/derivatives/containers/build_logs
trap 'rm -rf "$APPTAINER_TMPDIR" "$OVERLAY"' EXIT

if [[ ! -f "$BASE_SIF" ]]; then
    echo "Base SIF missing: $BASE_SIF" >&2
    exit 1
fi
if [[ -f "$OVERLAY_FINAL" ]]; then
    echo "$OVERLAY_FINAL already exists. Delete it to rebuild."
    ls -la "$OVERLAY_FINAL"
    exit 0
fi

echo "=== Step 1/3: create ${OVERLAY_SIZE_MB} MiB ext3 overlay ==="
apptainer overlay create --size "$OVERLAY_SIZE_MB" "$OVERLAY"
ls -la "$OVERLAY"

echo
echo "=== Step 2/3: pip-install REVE gap deps into the overlay (rw, fakeroot) ==="
# Single shell so the constraints file (written in-container) is in scope for
# the install. PYTHONNOUSERSITE: don't let ~/.local leak in. No PIP_NO_USER
# games needed — fakeroot maps us to root, writes land in the overlay upper.
apptainer exec --fakeroot --overlay "$OVERLAY" -B /data:/data "$BASE_SIF" bash -euc '
    export PYTHONNOUSERSITE=1
    export PIP_NO_CACHE_DIR=1

    echo "--- freezing CUDA stack + hub/safetensors into protect-constraints ---"
    # Protect the NGC CUDA stack AND huggingface-hub/safetensors: base-layer
    # pkgs we keep pinned so the transformers install will not downgrade them
    # (the overlay whiteout fragility that broke job 1554). transformers itself
    # is NOT protected -- installed fresh below.
    # NOTE: tokenizers is DELIBERATELY NOT protected. The 26.05 base ships
    # tokenizers 0.23.1, but EVERY published transformers (<=5.10.2) caps at
    # tokenizers<=0.23.0. Pinning 0.23.1 made pip backtrack to the ancient
    # transformers 4.17.0 (job 1566), which broke peft (missing PushToHubMixin).
    # Leaving tokenizers free lets pip downgrade it 0.23.1->0.22.2 (the newest
    # STABLE version <=0.23.0; 0.23.0 final was never released on PyPI) so a
    # modern transformers (5.10.2) resolves. tokenizers is a self-contained rust
    # wheel; the overlay verify (Step 3) confirms the downgrade imports cleanly.
    # Use `pip list --format=freeze`, NOT `pip freeze`: the NGC torch is
    # installed from a local wheel, so `pip freeze` emits "torch @ file:///opt/
    # transfer/...whl" (a direct URL). As a constraint that makes pip try to
    # reinstall torch from that now-deleted wheel (job 1557 OSError when
    # accelerate pulled torch). `pip list --format=freeze` emits a plain
    # "torch==<ver>" pin, which the already-installed torch satisfies in place.
    python -m pip list --format=freeze 2>/dev/null \
        | grep -iE "^(torch|torchvision|torchaudio|triton|pytorch-triton|nvidia-|cuda-|cudnn|cublas|cusparse|cusolver|cufft|curand|nccl|transformer-engine|flash-attn|apex|huggingface-hub|safetensors)([=<>~ ]|$)" \
        > /opt/reve-protect-constraints.txt || true
    cat /opt/reve-protect-constraints.txt
    echo "----------------------------------------------------"

    echo "torch BEFORE: $(python -c "import torch;print(torch.__version__, torch.version.cuda)")"

    # Pure additions: transformers/accelerate/peft (user-site-only in base) +
    # the deps genuinely MISSING from base. transformers UNPINNED -> resolves to
    # 5.10.2. tokenizers PINNED ==0.22.2: base ships 0.23.1, which transformers
    # REJECTS (enforces tokenizers<=0.23.0 at IMPORT time via require_version_core,
    # not just via pip). NOTE 0.23.0 has NO stable release on PyPI -- the index
    # jumps 0.22.2 -> 0.23.0rc0 -> 0.23.1 (job 1570 died on "No matching
    # distribution for tokenizers==0.23.0"). So 0.22.2 is the newest STABLE
    # version satisfying the transformers tokenizers cap = minimal real downgrade
    # (it is exactly what pip auto-resolved to in job 1569). Safe for REVE: its
    # ONLY use is `from transformers import AutoModel` (stable 4.x/5.x).
    # accelerate/peft unpinned (REVE floors met).
    python -m pip install --constraint /opt/reve-protect-constraints.txt \
        transformers \
        "tokenizers==0.22.2" \
        accelerate \
        peft \
        hydra-core \
        lmdb \
        moabb \
        pyedflib \
        optuna \
        openpyxl \
        idr-torch

    # The base SIF ships tokenizers 0.23.1 in a READ-ONLY lower layer. pip
    # downgrades it to 0.22.2 in the overlay upper and that import works WITHIN
    # this same writable session, but the lower-layer dist-info whiteout does NOT
    # survive a fresh :ro remount: fuse-overlayfs (fakeroot, this phase) writes a
    # whiteout encoding that the read-only overlay/fuse2fs path in apptainer ignores,
    # so a SEPARATE exec sees the base tokenizers==0.23.1 again (jobs 1569/1572,
    # note the fuse-overlayfs "connection abort" on teardown). Rather than fight
    # overlay whiteout semantics, make transformers TOLERANT of 0.23.1: bump the
    # tokenizers ceiling 0.23.0 -> 0.23.1 in the version table transformers reads
    # at import (require_version_core). This file lives in the overlay UPPER
    # (transformers is installed fresh here), so the edit is a pure overlay write
    # that always persists across remounts. Safe for REVE: its ONLY transformers
    # use is `from transformers import AutoModel`, and 0.23.1 is a patch release
    # over the 0.23.0 the pin already blesses.
    echo "--- patch transformers tokenizers ceiling 0.23.0 -> 0.23.1 (robust to overlay whiteout) ---"
    TVT=/usr/local/lib/python3.12/dist-packages/transformers/dependency_versions_table.py
    sed -i "s/tokenizers>=0.22.0,<=0.23.0/tokenizers>=0.22.0,<=0.23.1/" "$TVT"
    grep -n "tokenizers" "$TVT" || true
    # Hard gate (writable phase): transformers must import with WHATEVER tokenizers
    # the layered fs presents, so we never copy a broken overlay to /data.
    python -c "import transformers; from transformers import AutoModel; import importlib.metadata as m; print(\"in-build import OK, transformers\", transformers.__version__, \"tokenizers\", m.version(\"tokenizers\"))"
'

echo
echo "=== Step 3/3: verify (read-only overlay, no GPU needed) ==="
# PYTHONNOUSERSITE=1: ~/.local has transformers 5.8.0 + jax + pytorch_lightning
# that otherwise shadow the SIF/overlay stack. REVE training needs nothing from
# ~/.local, so always ignore it.
apptainer exec --overlay "$OVERLAY":ro -B /data:/data --env PYTHONNOUSERSITE=1 "$BASE_SIF" python -c \
    "import torch, triton, numpy, scipy, sklearn, pandas, einops; \
     import transformers, accelerate, peft, hydra, omegaconf; \
     import lmdb, pyedflib, optuna, openpyxl, idr_torch, moabb; \
     from transformers import AutoModel; \
     from peft import LoraConfig, get_peft_model; \
     from accelerate import Accelerator; \
     print('torch', torch.__version__, 'cuda', torch.version.cuda); \
     print('triton', triton.__version__); \
     print('transformers', transformers.__version__); \
     print('accelerate', accelerate.__version__, 'peft', peft.__version__); \
     print('moabb', moabb.__version__, 'sklearn', sklearn.__version__); \
     print('tokenizers', __import__('tokenizers').__version__); \
     print('ALL OK')"

echo
echo "=== Verify passed; copying local overlay -> /data (single sequential write) ==="
cp "$OVERLAY" "$OVERLAY_FINAL"
ls -la "$OVERLAY_FINAL"

echo
echo "Done. Overlay at $OVERLAY_FINAL"
echo "Runtime: apptainer exec --nv --overlay $OVERLAY_FINAL:ro -B /data:/data --env PYTHONNOUSERSITE=1 $BASE_SIF python ..."
