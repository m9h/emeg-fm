#!/bin/bash
#SBATCH --job-name=alljoined-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/derivatives/alljoined_smoke/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Single-subject Alljoined-1.6M EEG->image retrieval smoke with frozen REVE.
# Stages the gold preprocessed npy + the metadata parquet (which carries the
# per-trial `dropped` flag needed to align epochs to stimuli) + stimuli.zip,
# then runs scripts/extract_alljoined_reve.py inside the PyTorch NGC SIF.
#
# Usage:
#   sbatch scripts/alljoined_smoke.sbatch                 # sub-01, REVE L6
#   SUBJECT=02 LAYER=8 sbatch scripts/alljoined_smoke.sbatch
#
# Pre-req: accept the brain-bzh/reve-base + reve-positions Responsible Use
# Agreement on HF and have HF_TOKEN in ~/.cache/huggingface/token. The
# Alljoined dataset itself is public (no token for the data download).

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/emeg-fm
SIF=/data/derivatives/containers/pytorch_26.05.sif
AJ_DIR=/data/datasets/alljoined_smoke
OUT_DIR=/data/derivatives/alljoined_smoke
HF_CACHE=$OUT_DIR/hf_cache
SKLEARN_LIBS=/data/derivatives/eeg_sae/wwlibs      # aarch64 sklearn 1.8 + joblib
HF_TOKEN_FILE=${HF_HOME:-$HOME/.cache/huggingface}/token

SUBJECT=${SUBJECT:-01}
MODEL=${MODEL:-brain-bzh/reve-base}
LAYER=${LAYER:-6}
PARTITION=${PARTITION:-stim_test}
CLIP_MODEL=${CLIP_MODEL:-openai/clip-vit-base-patch32}
REPO_ID=Alljoined/Alljoined-1.6M

SUB_DIR="$AJ_DIR/preprocessed_eeg/sub-$SUBJECT"
NPY="$SUB_DIR/preprocessed_eeg_test_flat.npy"
META="$SUB_DIR/experiment_metadata_categories.parquet"
STIM_ZIP="$AJ_DIR/stimuli.zip"
# Extract the ~16.7k stimulus jpgs to NODE-LOCAL disk, never to /data (NFS):
# unzipping thousands of small files onto NFS wedges in uninterruptible
# D-state (rpc_wait_bit_killable) — job 1579 hung 4 h there. Local nvme does
# the same extraction in ~3 s.
STIM_DIR="${LOCAL_SCRATCH:-/tmp}/alljoined_stimuli"
OUT_JSON="$OUT_DIR/sub-${SUBJECT}_${PARTITION}_reve-L${LAYER}.json"

mkdir -p "$OUT_DIR/logs" "$SUB_DIR" "$HF_CACHE"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: $SIF not found." >&2; exit 1
fi

# Gated-model token (REVE only). Dataset download below needs no token.
HF_TOKEN=""
if [[ "$MODEL" == brain-bzh/* ]]; then
    if [[ ! -s "$HF_TOKEN_FILE" ]]; then
        echo "ERROR: HF token not at $HF_TOKEN_FILE — needed for REVE gated model." >&2
        exit 1
    fi
    HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

# --- Stage data (idempotent; hf download is cached) ------------------------
# Pass explicit relative paths to `hf download` — NOT --include (the include
# filter is silently ignored when explicit filenames are given).
download() {  # $1 = repo-relative path, $2 = local dest
    if [[ -f "$2" ]]; then echo "[stage] have $(basename "$2")"; return; fi
    echo "[stage] downloading $1"
    hf download "$REPO_ID" "$1" --repo-type dataset \
        --local-dir "$AJ_DIR" >/dev/null
}
download "preprocessed_eeg/sub-$SUBJECT/preprocessed_eeg_test_flat.npy" "$NPY"
download "preprocessed_eeg/sub-$SUBJECT/experiment_metadata_categories.parquet" "$META"
download "stimuli.zip" "$STIM_ZIP"

# Guard on a populated dir, not just existence — a half-extracted dir would
# silently drop stimuli. 16749 jpgs expected; re-extract if short.
N_JPG=$(find "$STIM_DIR" -iname '*.jpg' 2>/dev/null | wc -l)
if [[ "$N_JPG" -lt 16000 ]]; then
    echo "[stage] unzipping stimuli.zip to local $STIM_DIR (had $N_JPG jpgs)"
    rm -rf "$STIM_DIR"; mkdir -p "$STIM_DIR"
    unzip -q -o "$STIM_ZIP" -d "$STIM_DIR"
    echo "[stage] extracted $(find "$STIM_DIR" -iname '*.jpg' | wc -l) jpgs"
else
    echo "[stage] reusing local stimuli ($N_JPG jpgs)"
fi

echo "Job:    alljoined_smoke.sbatch"
echo "Subj:   sub-$SUBJECT  model=$MODEL L$LAYER  partition=$PARTITION"
echo "Out:    $OUT_JSON"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

PY_CMD=( python scripts/extract_alljoined_reve.py
    --eeg-npy "$NPY"
    --stim-parquet "$META"
    --stimuli-dir "$STIM_DIR"
    --partition "$PARTITION"
    --model "$MODEL"
    --layer "$LAYER"
    --clip-model "$CLIP_MODEL"
    --out "$OUT_JSON"
)

# --no-init : NGC pytorch SIF startup scripts hang (license/nvidia-cli probe);
#   safe to bypass since HF_TOKEN/HF_HOME/PYTHONPATH are passed via --env.
# -B /data:/data : apptainer doesn't auto-bind NFS.
# LD path prepend : --nv strips /usr/lib/aarch64-linux-gnu, breaking cuDNN.
# NO PYTHONNOUSERSITE : transformers/tokenizers/safetensors/sklearn live in
#   user-site (~/.local); the SIF ships torch but not transformers. user-site
#   has no torch/triton, so nothing shadows the SIF stack (the usual hazard).
#   SKLEARN_LIBS stays on PYTHONPATH as a known-good aarch64 sklearn fallback.
# No jax shim needed : the emeg_fm package __init__ is import-light and
#   emeg_fm.adapters imports jax only inside torch_to_jax (never on the REVE
#   path), so the submodules import cleanly in this jax-less SIF. PROJECT_ROOT
#   on PYTHONPATH makes `import emeg_fm` resolve.
apptainer exec --nv --no-init \
    --env HF_TOKEN="$HF_TOKEN" \
    --env HF_HOME="$HF_CACHE" \
    --env PYTHONPATH="$PROJECT_ROOT:$SKLEARN_LIBS" \
    --env LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}" \
    -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
    -B /data:/data \
    -B "$STIM_DIR":"$STIM_DIR" \
    "$SIF" bash -c '
        set -euo pipefail
        cd "$1"; shift
        echo "[exec] $*"
        exec "$@"
    ' bash "$PROJECT_ROOT" "${PY_CMD[@]}"

unset HF_TOKEN
echo "[done] smoke finished: $OUT_JSON"
