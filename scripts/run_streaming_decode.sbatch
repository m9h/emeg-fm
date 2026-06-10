#!/bin/bash
#SBATCH --job-name=stream-decode-replay
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/derivatives/alljoined_smoke/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Headless validation of the realtime streaming-decode path: replays a
# pre-epoched Alljoined subject through the SAME Trial interface the live LSL
# acquisition emits (emeg_fm.streaming.FileReplaySource), then runs the
# per-subject ridge calibration + online retrieval (StreamingReveDecoder).
# Proves emeg_fm.{streaming,decoder,stimuli} end-to-end on real EEG before a
# headset session. Same SIF/env as alljoined_smoke.sbatch.
#
# Usage:
#   sbatch scripts/run_streaming_decode.sbatch                  # sub-01, REVE L6
#   SUBJECT=02 CALIB_FRAC=0.8 sbatch scripts/run_streaming_decode.sbatch

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
CALIB_FRAC=${CALIB_FRAC:-0.8}
MAX_IMAGES=${MAX_IMAGES:-}            # blank = use all unique images

SUB_DIR="$AJ_DIR/preprocessed_eeg/sub-$SUBJECT"
NPY="$SUB_DIR/preprocessed_eeg_test_flat.npy"
META="$SUB_DIR/experiment_metadata_categories.parquet"
STIM_ZIP="$AJ_DIR/stimuli.zip"
# Extract stimulus jpgs to NODE-LOCAL disk, never /data (NFS small-file unzip
# wedges in D-state for hours).
STIM_DIR="${LOCAL_SCRATCH:-/tmp}/alljoined_stimuli"
OUT_JSON="$OUT_DIR/sub-${SUBJECT}_${PARTITION}_streamreplay-L${LAYER}.json"

mkdir -p "$OUT_DIR/logs" "$HF_CACHE"

if [[ ! -f "$SIF" ]]; then echo "ERROR: $SIF not found." >&2; exit 1; fi
for f in "$NPY" "$META" "$STIM_ZIP"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f — run alljoined_smoke staging first." >&2; exit 1; }
done

HF_TOKEN=""
if [[ "$MODEL" == brain-bzh/* ]]; then
    [[ -s "$HF_TOKEN_FILE" ]] || { echo "ERROR: HF token not at $HF_TOKEN_FILE." >&2; exit 1; }
    HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

# Guard on a populated dir; re-extract only if short (16749 jpgs expected).
N_JPG=$(find "$STIM_DIR" -iname '*.jpg' 2>/dev/null | wc -l)
if [[ "$N_JPG" -lt 16000 ]]; then
    echo "[stage] unzipping stimuli.zip to local $STIM_DIR (had $N_JPG jpgs)"
    rm -rf "$STIM_DIR"; mkdir -p "$STIM_DIR"
    unzip -q -o "$STIM_ZIP" -d "$STIM_DIR"
    echo "[stage] extracted $(find "$STIM_DIR" -iname '*.jpg' | wc -l) jpgs"
else
    echo "[stage] reusing local stimuli ($N_JPG jpgs)"
fi

echo "Job:    run_streaming_decode.sbatch (--replay)"
echo "Subj:   sub-$SUBJECT  model=$MODEL L$LAYER  calib_frac=$CALIB_FRAC"
echo "Out:    $OUT_JSON"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

PY_CMD=( python scripts/run_streaming_decode.py --replay
    --eeg-npy "$NPY"
    --stim-parquet "$META"
    --stimuli-dir "$STIM_DIR"
    --partition "$PARTITION"
    --model "$MODEL"
    --layer "$LAYER"
    --clip-model "$CLIP_MODEL"
    --calib-frac "$CALIB_FRAC"
    --out "$OUT_JSON"
)
[[ -n "$MAX_IMAGES" ]] && PY_CMD+=( --max-images "$MAX_IMAGES" )

# --no-init: NGC SIF startup probes hang. -B /data: NFS not auto-bound.
# LD path prepend: --nv strips aarch64 libs (breaks cuDNN). No PYTHONNOUSERSITE:
# transformers/sklearn live in user-site; SKLEARN_LIBS is the aarch64 fallback.
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
echo "[done] streaming-decode replay finished: $OUT_JSON"
