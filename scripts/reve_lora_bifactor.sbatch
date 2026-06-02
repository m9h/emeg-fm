#!/bin/bash
#SBATCH --job-name=reve-lora-bifactor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Scope C of the Muon experiment (docs/MUON_EXPERIMENT.md): LoRA fine-tune
# frozen REVE on the HBN psychopathology bifactor probe twice — AdamW vs Muon
# on the LoRA matrices — sharing seed / release / split, so the optimizer is
# the only difference. Then compare held-out Pearson r (does-it-help) and the
# post-hoc Δ(alpha_hill) on the five WeightWatcher-flagged α<2 attention
# matrices (H1). Single GPU, FIFO: adamw then muon.
#
# Usage:
#   sbatch scripts/reve_lora_bifactor.sbatch                       # HBN R1 MINI
#   RELEASE=5 FULL=1 sbatch scripts/reve_lora_bifactor.sbatch      # R5 full
#
# Optional env knobs (defaults aim at a single-GPU smoke run):
#   RELEASE=1 FULL=0 TASK=RestingState LAYER=6
#   RANK=8 ALPHA=16 EPOCHS=10 BATCH=8 WIN_SECONDS=5
#   ADAM_LR=1e-3                # AdamW LoRA LR (also Muon's head/adam-fallback LR)
#   MUON_LR=2e-2                # Muon matrix-update LR — tune separately!
#   VAL_FRAC=0.3 SEED=0 MAX_WINDOWS= (empty=all)
#   TARGETS=externalizing,p_factor,internalizing,attention
#
# Pre-req: REVE is gated — accept the brain-bzh/reve-base Responsible Use
# Agreement on HF and have a token at ~/.cache/huggingface/token. The token is
# read here and passed via --env; it is never written into the repo.

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/eeg-fm-spectral
SIF=/data/derivatives/containers/pytorch_26.04.sif
OUT_DIR=/data/derivatives/eeg_sae
CACHE_DIR=/data/derivatives/eegdash_cache
HF_CACHE=/data/derivatives/eeg_sae/hf_cache
HF_TOKEN_FILE=${HF_HOME:-$HOME/.cache/huggingface}/token

RELEASE=${RELEASE:-1}
TASK=${TASK:-RestingState}
LAYER=${LAYER:-6}
RANK=${RANK:-8}
ALPHA=${ALPHA:-16}
EPOCHS=${EPOCHS:-10}
BATCH=${BATCH:-8}
WIN_SECONDS=${WIN_SECONDS:-5}
ADAM_LR=${ADAM_LR:-1e-3}
MUON_LR=${MUON_LR:-2e-2}
VAL_FRAC=${VAL_FRAC:-0.3}
SEED=${SEED:-0}
MAX_WINDOWS=${MAX_WINDOWS:-}
TARGETS=${TARGETS:-externalizing,p_factor,internalizing,attention}

FULL_FLAG=""
DATASET_TAG="EEG2025R${RELEASE}MINI"
if [[ "${FULL:-0}" == "1" ]]; then
    FULL_FLAG="--full"
    DATASET_TAG="EEG2025R${RELEASE}"
fi

RUN_TAG="reve-base_lora_L${LAYER}_${DATASET_TAG}_${TASK}_r${RANK}"
BASE="$OUT_DIR/lora/${RUN_TAG}"
ADAMW_PREFIX="${BASE}_adamw"
MUON_PREFIX="${BASE}_muon"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/lora" "$CACHE_DIR" "$HF_CACHE"

[[ -f "$SIF" ]] || { echo "ERROR: $SIF not found." >&2; exit 1; }
if [[ ! -s "$HF_TOKEN_FILE" ]]; then
    echo "ERROR: HF token not at $HF_TOKEN_FILE — needed for REVE gated model." >&2
    exit 1
fi
HF_TOKEN=$(cat "$HF_TOKEN_FILE")

echo "Job:    reve_lora_bifactor.sbatch"
echo "Data:   ${DATASET_TAG} / task=$TASK   ${WIN_SECONDS}s @ 200Hz"
echo "LoRA:   block=$LAYER rank=$RANK alpha=$ALPHA epochs=$EPOCHS batch=$BATCH seed=$SEED"
echo "AdamW:  lr=$ADAM_LR  -> ${ADAMW_PREFIX}.json"
echo "Muon:   lr=$MUON_LR (head/adam-fallback lr=$ADAM_LR) -> ${MUON_PREFIX}.json"
echo "Target: $TARGETS"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Args shared by both runs, built in the OUTER shell (the array-in-quoted-bash-c
# quoting bug drops values otherwise — see extract_eeg_fm_acts.sbatch).
COMMON=( python scripts/reve_lora_bifactor.py
    --release "$RELEASE" --task "$TASK" --layer "$LAYER"
    --rank "$RANK" --alpha "$ALPHA" --epochs "$EPOCHS" --batch-size "$BATCH"
    --win-seconds "$WIN_SECONDS" --val-frac "$VAL_FRAC" --seed "$SEED"
    --targets "$TARGETS" --cache-dir "$CACHE_DIR" )
[[ -n "$FULL_FLAG" ]] && COMMON+=( "$FULL_FLAG" )
[[ -n "$MAX_WINDOWS" ]] && COMMON+=( --max-windows "$MAX_WINDOWS" )

ADAMW_CMD=( "${COMMON[@]}" --optimizer adamw --lr "$ADAM_LR"
    --out-prefix "$ADAMW_PREFIX" )
MUON_CMD=( "${COMMON[@]}" --optimizer muon --lr "$MUON_LR" --adam-lr "$ADAM_LR"
    --out-prefix "$MUON_PREFIX" )

# Compare the two JSONs: held-out Pearson r and the H1 spectral deltas.
CMP_PY=$(cat <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1] + ".json"))
m = json.load(open(sys.argv[2] + ".json"))
print("\n=== Scope C: LoRA-REVE bifactor, Muon vs AdamW ===")
print(f"{'':14s}  adamw     muon      Δ(muon-adamw)")
for t in a["pearson_r"]:
    ra, rm = a["pearson_r"][t], m["pearson_r"][t]
    print(f"r/{t:12s}  {ra:+.3f}    {rm:+.3f}    {rm-ra:+.3f}")
print(f"r/{'mean':12s}  {a['pearson_r_mean']:+.3f}    {m['pearson_r_mean']:+.3f}"
      f"    {m['pearson_r_mean']-a['pearson_r_mean']:+.3f}")
print("\n-- H1: alpha_hill on the 5 α<2 LoRA targets (toward 2.0 from below) --")
for name in a["spectral"]:
    da, dm = a["spectral"][name], m["spectral"][name]
    print(f"{name:34s} base={da['base']['alpha_hill']:.2f}  "
          f"adamw→{da['merged']['alpha_hill']:.2f} (Δ{da['delta_alpha_hill']:+.3f})  "
          f"muon→{dm['merged']['alpha_hill']:.2f} (Δ{dm['delta_alpha_hill']:+.3f})")
PYEOF
)

# --no-init: NGC pytorch SIF startup scripts hang (license/nvidia-cli probe);
#   safe to bypass since we pass HF_TOKEN/HF_HOME/PYTHONPATH via --env.
# -B /data:/data: apptainer doesn't auto-bind NFS.
# libcuda bind: same host-driver-over-stale-compat-path fix as the extract
#   sbatch (harmless for REVE; only matters if a Triton kernel JITs).
LIBCUDA_BIND=()
HOST_LIBCUDA=$(readlink -f /usr/lib/aarch64-linux-gnu/libcuda.so.1 2>/dev/null || true)
if [[ -n "$HOST_LIBCUDA" && -e "$HOST_LIBCUDA" ]]; then
    LIBCUDA_BIND=( -B "$HOST_LIBCUDA":/usr/local/cuda/compat/lib/libcuda.so.1 )
fi

run_in_sif () {
    apptainer exec --nv --no-init \
        --env HF_TOKEN="$HF_TOKEN" \
        --env HF_HOME="$HF_CACHE" \
        --env PYTHONPATH="$PROJECT_ROOT" \
        --env PYTHONNOUSERSITE=1 \
        -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
        -B /data:/data \
        -B "$CACHE_DIR":"$CACHE_DIR" \
        -B "$HF_CACHE":"$HF_CACHE" \
        "${LIBCUDA_BIND[@]}" \
        "$SIF" bash -c '
            set -euo pipefail
            cd "$1"; shift
            echo "[exec] $*"
            exec "$@"
        ' bash "$PROJECT_ROOT" "$@"
}

echo "[lora] === AdamW run ==="
run_in_sif "${ADAMW_CMD[@]}"
echo "[lora] === Muon run ==="
run_in_sif "${MUON_CMD[@]}"
echo "[lora] === comparison ==="
run_in_sif python -c "$CMP_PY" "$ADAMW_PREFIX" "$MUON_PREFIX"

unset HF_TOKEN
echo "[done] Scope C complete:"
echo "  adamw: ${ADAMW_PREFIX}.json"
echo "  muon:  ${MUON_PREFIX}.json"
