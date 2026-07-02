#!/usr/bin/env bash
# Run a LuMamba script inside the Docker NGC PyTorch 26.05 container on /mnt/t9.
#
# LuMamba (PulpBio/LuMamba, Mamba/SSM EEG-FM) needs mamba-ssm + causal-conv1d,
# built from source for the GB10 (sm_121) with --no-build-isolation (their
# setup.py imports torch at build time; PEP-517 isolation hides it). Built once
# into a persistent /mnt/t9 target and reused. Arch = cloned Apache-2.0
# BioFoundation (models.LuMamba); weights from HF PulpBio/LuMamba (not gated).
#
# Usage: scripts/run_lumamba.sh scripts/lumamba_smoke.py [args...]
set -euo pipefail

IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"
T9="${T9:-/mnt/t9}"
LIBS="$T9/lumamba_libs"
mkdir -p "$LIBS"
# BioFoundation first so `import models.LuMamba` resolves; lumamba_libs for the
# built mamba kernels + deps; moabblibs for mne/sklearn; tokfix for hub.
PYPATH="$T9/BioFoundation:$LIBS:/emeg-fm:/emeg-fm/fmscope:$T9/moabblibs:$T9/tokfix"
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"

docker run --rm -i --gpus all --ipc=host \
  -v "$EMEG_FM:/emeg-fm" -v "$T9:$T9" -v "/data:/data:ro" \
  -e PYTHONNOUSERSITE=1 -e PYTHONPATH="$PYPATH" -e LIBS="$LIBS" \
  -e HF_HOME="$T9/hf" -e HF_TOKEN="$HF_TOKEN" -e MNE_DATA="$T9/moabb_data" \
  -e MAX_JOBS=4 -e MAMBA_FORCE_BUILD=TRUE -e CAUSAL_CONV1D_FORCE_BUILD=TRUE \
  -w /emeg-fm "$IMAGE" bash -s -- "$@" <<'INNER'
set -e
if ! python -c "import mamba_ssm, causal_conv1d, rotary_embedding_torch, timm, einops, safetensors" 2>/dev/null; then
  echo "[setup] building mamba kernels + deps into $LIBS (one-time) ..."
  pip install --no-cache-dir --target "$LIBS" --no-build-isolation ninja packaging setuptools wheel >/tmp/lm_setup.log 2>&1 || true
  if ! pip install --no-cache-dir --target "$LIBS" --no-build-isolation causal-conv1d mamba-ssm >>/tmp/lm_setup.log 2>&1; then
    echo "[setup] mamba/causal build FAILED"; tail -25 /tmp/lm_setup.log; exit 3
  fi
  pip install --no-cache-dir --target "$LIBS" rotary-embedding-torch timm einops safetensors huggingface_hub >>/tmp/lm_setup.log 2>&1 || true
  python -c "import mamba_ssm; print('[setup] mamba_ssm', mamba_ssm.__version__, 'OK')"
fi
# BioFoundation's channel_embeddings.py imports torcheeg.datasets.constants.
# SEED_CHANNEL_LIST only to size the DECODER channel vocab (LuMamba.encode never
# uses it), so shim it instead of pulling the heavy torcheeg dep.
python -c "import torcheeg.datasets.constants" 2>/dev/null || python -c "import os;d=os.path.join(os.environ['LIBS'],'torcheeg','datasets');os.makedirs(d,exist_ok=True);open(os.path.join(os.environ['LIBS'],'torcheeg','__init__.py'),'w').close();open(os.path.join(d,'__init__.py'),'w').close();open(os.path.join(d,'constants.py'),'w').write('SEED_CHANNEL_LIST='+repr('FP1 FPZ FP2 AF3 AF4 F7 F5 F3 F1 FZ F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCZ FC2 FC4 FC6 FT8 T7 C5 C3 C1 CZ C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO5 PO3 POZ PO4 PO6 PO8 CB1 O1 OZ O2 CB2'.split())+chr(10))"
python "$@"
INNER
