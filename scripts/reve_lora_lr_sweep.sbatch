#!/bin/bash
#SBATCH --job-name=reve-lora-lr-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Fair best-of-best for Scope C (docs/MUON_EXPERIMENT.md): Muon's update is
# RMS-matched, so comparing it at AdamW's LR is a rigged test — the same caveat
# the Scope A sweep (muon_lr_sweep.sbatch) addresses, now applied to the
# LoRA-REVE bifactor probe. This sweeps the LoRA-matrix LR independently for
# each optimizer on a cheap (few-epoch) budget over the SAME subjects / split /
# seed, auto-picks the best LR per optimizer (highest held-out mean Pearson r),
# then runs a full-budget AdamW-vs-Muon head-to-head at the two winners and
# prints both the downstream r table and the H1 Δ(alpha_hill) on the 5 α<2
# matrices. Single GPU, FIFO.
#
# Usage:
#   RELEASE=1 FULL=1 sbatch scripts/reve_lora_lr_sweep.sbatch
#
# Optional env knobs:
#   RELEASE=1 FULL=0 TASK=RestingState LAYER=6 RANK=8 ALPHA=16 BATCH=8
#   WIN_SECONDS=5 VAL_FRAC=0.3 SEED=0
#   ADAM_LRS="3e-4 1e-3 3e-3"            # AdamW LoRA-LR grid
#   MUON_LRS="5e-3 1e-2 2e-2 5e-2"       # Muon matrix-LR grid (needs larger LR)
#   ADAM_FALLBACK_LR=1e-3                # AdamW LR for Muon's head/1-D group (fixed)
#   SWEEP_EPOCHS=5                       # cheap grid budget (same data, fewer epochs)
#   FINAL_EPOCHS=10                      # full confirm budget at the winners
#
# Pre-req: REVE is gated — accept the brain-bzh/reve-base Responsible Use
# Agreement on HF and have a token at ~/.cache/huggingface/token. The token is
# read here and passed via --env; it is never written into the repo.

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/emeg-fm
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
BATCH=${BATCH:-8}
WIN_SECONDS=${WIN_SECONDS:-5}
VAL_FRAC=${VAL_FRAC:-0.3}
SEED=${SEED:-0}
TARGETS=${TARGETS:-externalizing,p_factor,internalizing,attention}
ADAM_LRS=${ADAM_LRS:-"3e-4 1e-3 3e-3"}
MUON_LRS=${MUON_LRS:-"5e-3 1e-2 2e-2 5e-2"}
ADAM_FALLBACK_LR=${ADAM_FALLBACK_LR:-1e-3}
SWEEP_EPOCHS=${SWEEP_EPOCHS:-5}
FINAL_EPOCHS=${FINAL_EPOCHS:-10}

FULL_FLAG=""
DATASET_TAG="EEG2025R${RELEASE}MINI"
if [[ "${FULL:-0}" == "1" ]]; then
    FULL_FLAG="--full"
    DATASET_TAG="EEG2025R${RELEASE}"
fi

RUN_TAG="reve-base_lora_L${LAYER}_${DATASET_TAG}_${TASK}_r${RANK}"
SWEEP_DIR="$OUT_DIR/lora/${RUN_TAG}_lrsweep"
BEST_ENV="$SWEEP_DIR/best.env"

mkdir -p "$OUT_DIR/logs" "$SWEEP_DIR" "$CACHE_DIR" "$HF_CACHE"

[[ -f "$SIF" ]] || { echo "ERROR: $SIF not found." >&2; exit 1; }
if [[ ! -s "$HF_TOKEN_FILE" ]]; then
    echo "ERROR: HF token not at $HF_TOKEN_FILE — needed for REVE gated model." >&2
    exit 1
fi
HF_TOKEN=$(cat "$HF_TOKEN_FILE")

echo "Job:    reve_lora_lr_sweep.sbatch"
echo "Data:   ${DATASET_TAG} / task=$TASK   ${WIN_SECONDS}s @ 200Hz"
echo "LoRA:   block=$LAYER rank=$RANK alpha=$ALPHA batch=$BATCH seed=$SEED"
echo "Sweep:  epochs=$SWEEP_EPOCHS  (full data, same split — only LR varies)"
echo "  adam LRs: $ADAM_LRS"
echo "  muon LRs: $MUON_LRS  (adam-fallback lr=$ADAM_FALLBACK_LR)"
echo "Final:  epochs=$FINAL_EPOCHS"
echo "Target: $TARGETS"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# --no-init: NGC pytorch SIF startup scripts hang (license/nvidia-cli probe);
#   safe to bypass since we pass HF_TOKEN/HF_HOME/PYTHONPATH via --env.
# -B /data:/data: apptainer doesn't auto-bind NFS.
# libcuda bind: host-driver-over-stale-compat-path fix (harmless for REVE).
# No PYTHONNOUSERSITE: eegdash/braindecode/mne live in ~/.local and the SIF
#   lacks them; torch/triton/torchaudio are NOT in user-site so nothing shadows
#   the SIF CUDA stack (same rationale as reve_lora_bifactor.sbatch).
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

# Shared args (everything but optimizer / lr / epochs / out-prefix). Built in
# the OUTER shell to dodge the array-in-quoted-bash-c quoting bug.
common_cmd () {  # $1=epochs $2=out_prefix
    local epochs="$1" prefix="$2"
    local -a cmd=( python scripts/reve_lora_bifactor.py
        --release "$RELEASE" --task "$TASK" --layer "$LAYER"
        --rank "$RANK" --alpha "$ALPHA" --epochs "$epochs" --batch-size "$BATCH"
        --win-seconds "$WIN_SECONDS" --val-frac "$VAL_FRAC" --seed "$SEED"
        --targets "$TARGETS" --cache-dir "$CACHE_DIR" --out-prefix "$prefix" )
    [[ -n "$FULL_FLAG" ]] && cmd+=( "$FULL_FLAG" )
    printf '%s\0' "${cmd[@]}"
}

run_adam () {  # $1=lr $2=epochs $3=prefix
    mapfile -d '' -t _c < <(common_cmd "$2" "$3")
    run_in_sif "${_c[@]}" --optimizer adamw --lr "$1"
}
run_muon () {  # $1=lr $2=epochs $3=prefix
    mapfile -d '' -t _c < <(common_cmd "$2" "$3")
    run_in_sif "${_c[@]}" --optimizer muon --lr "$1" --adam-lr "$ADAM_FALLBACK_LR"
}

# ---------------------------------------------------------------------------
# Phase 1 — independent LR grids (cheap epoch budget, identical split)
# ---------------------------------------------------------------------------
for lr in $ADAM_LRS; do
    echo "[sweep] === adam lr=$lr ==="
    run_adam "$lr" "$SWEEP_EPOCHS" "$SWEEP_DIR/adam_lr${lr}"
done
for lr in $MUON_LRS; do
    echo "[sweep] === muon lr=$lr ==="
    run_muon "$lr" "$SWEEP_EPOCHS" "$SWEEP_DIR/muon_lr${lr}"
done

# ---------------------------------------------------------------------------
# Phase 2 — rank each optimizer's grid by held-out mean Pearson r
# ---------------------------------------------------------------------------
PICK_PY=$(cat <<'PYEOF'
import glob, json, os, sys
sweep_dir = sys.argv[1]
runs = {}
for jf in glob.glob(os.path.join(sweep_dir, "*.json")):
    d = json.load(open(jf))
    c = d["config"]
    opt = c["optimizer"]
    lr = c["muon_lr"] if opt == "muon" else c["adamw_lr"]
    runs.setdefault(opt, []).append(dict(lr=lr, r=d["pearson_r_mean"]))

best = {}
for opt, rs in runs.items():
    rs.sort(key=lambda r: r["lr"])
    pick = max(rs, key=lambda r: r["r"])
    best[opt] = pick
    print(f"\n=== {opt} LR grid ===")
    print(f"{'lr':>8s}  {'mean_r':>7s}")
    for r in rs:
        mark = "  <-- best" if r is pick else ""
        print(f"{r['lr']:>8.2e}  {r['r']:+7.3f}{mark}")

with open(os.path.join(sweep_dir, "best.env"), "w") as f:
    if "adamw" in best:
        f.write(f"BEST_ADAM_LR={best['adamw']['lr']:.6g}\n")
    if "muon" in best:
        f.write(f"BEST_MUON_LR={best['muon']['lr']:.6g}\n")
print("\n[pick] wrote best.env")
PYEOF
)

echo "[sweep] === ranking ==="
run_in_sif python -c "$PICK_PY" "$SWEEP_DIR"

# shellcheck disable=SC1090
source "$BEST_ENV"
BEST_ADAM_LR=${BEST_ADAM_LR:-$ADAM_FALLBACK_LR}
BEST_MUON_LR=${BEST_MUON_LR:-2e-2}
echo "[sweep] best-of-best: ADAM_LR=$BEST_ADAM_LR  MUON_LR=$BEST_MUON_LR"

# ---------------------------------------------------------------------------
# Phase 3 — full-budget head-to-head at the winning LRs
# ---------------------------------------------------------------------------
FINAL_ADAM="$SWEEP_DIR/final_adamw"
FINAL_MUON="$SWEEP_DIR/final_muon"
echo "[final] === adamw lr=$BEST_ADAM_LR (full) ==="
run_adam "$BEST_ADAM_LR" "$FINAL_EPOCHS" "$FINAL_ADAM"
echo "[final] === muon lr=$BEST_MUON_LR (full) ==="
run_muon "$BEST_MUON_LR" "$FINAL_EPOCHS" "$FINAL_MUON"

CMP_PY=$(cat <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1] + ".json"))
m = json.load(open(sys.argv[2] + ".json"))
print("\n=== Best-of-best Scope C: LoRA-REVE bifactor, Muon vs AdamW ===")
print(f"adam lr={a['config']['adamw_lr']:.1e}   "
      f"muon lr={m['config']['muon_lr']:.1e}")
print(f"\n{'':14s}  adamw     muon      Δ(muon-adamw)")
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
echo "[final] === comparison ==="
run_in_sif python -c "$CMP_PY" "$FINAL_ADAM" "$FINAL_MUON"

echo "[done] Scope C LR sweep complete. best.env: $BEST_ENV"
echo "  finals: ${FINAL_ADAM}.json  ${FINAL_MUON}.json"
