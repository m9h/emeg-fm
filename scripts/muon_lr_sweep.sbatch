#!/bin/bash
#SBATCH --job-name=muon-lr-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Fair best-of-best for Scope A (docs/MUON_EXPERIMENT.md): Muon's update is
# RMS-matched, so comparing it at AdamW's optimal LR is a rigged test. This
# sweeps ADAM_LR and MUON_LR *independently* on a matched (subsampled, fewer
# epochs) budget — every grid run shares seed / d_dict / k / max-tokens /
# epochs, so the only difference within an optimizer's grid is its LR — then
# auto-picks the best LR per optimizer (most live features at matched EV) and
# runs a final FULL-scale AdamW-vs-Muon head-to-head at the two winners.
#
# Usage:
#   ACTS=/data/derivatives/eeg_sae/acts/brain-bzh_reve-base_L-1_EEG2025R1_RestingState.npz \
#     sbatch scripts/muon_lr_sweep.sbatch
#
# Optional env knobs:
#   D_DICT=4096 K=32 AUX_K=32 SEED=0 BATCH=4096
#   ADAM_LRS="3e-4 1e-3 3e-3"            # AdamW LR grid
#   MUON_LRS="5e-3 1e-2 2e-2 5e-2 1e-1"  # Muon matrix-LR grid (needs larger LR)
#   ADAM_FALLBACK_LR=1e-3                # AdamW LR for Muon's 1-D bias group (fixed)
#   SWEEP_EPOCHS=12 SWEEP_MAX_TOKENS=3000000   # cheap grid budget
#   FINAL_EPOCHS=20 FINAL_MAX_TOKENS=          # full confirm budget (empty=all)

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/eeg-fm-spectral
SIF=/data/derivatives/containers/jax-arm64-cu13.sif
OUT_DIR=/data/derivatives/eeg_sae

ACTS=${ACTS:?"set ACTS=/path/to/activations.npz"}
K=${K:-32}
D_DICT=${D_DICT:-4096}
BATCH=${BATCH:-4096}
AUX_K=${AUX_K:-32}
AUX_COEF=${AUX_COEF:-0.03125}
AUX_KILL=${AUX_KILL:-200}
SEED=${SEED:-0}
ADAM_LRS=${ADAM_LRS:-"3e-4 1e-3 3e-3"}
MUON_LRS=${MUON_LRS:-"5e-3 1e-2 2e-2 5e-2 1e-1"}
ADAM_FALLBACK_LR=${ADAM_FALLBACK_LR:-1e-3}
SWEEP_EPOCHS=${SWEEP_EPOCHS:-12}
SWEEP_MAX_TOKENS=${SWEEP_MAX_TOKENS:-3000000}
FINAL_EPOCHS=${FINAL_EPOCHS:-20}
FINAL_MAX_TOKENS=${FINAL_MAX_TOKENS:-}

ACTS_TAG=$(basename "$ACTS" .npz)
SWEEP_DIR="$OUT_DIR/sae/${ACTS_TAG}_k${K}_dd${D_DICT}_lrsweep"
BEST_ENV="$SWEEP_DIR/best.env"

mkdir -p "$OUT_DIR/logs" "$SWEEP_DIR"

[[ -f "$SIF"  ]] || { echo "ERROR: $SIF not found (build_jax_arm64_sif.sbatch)." >&2; exit 1; }
[[ -f "$ACTS" ]] || { echo "ERROR: activations not found: $ACTS" >&2; exit 1; }

echo "Job:    muon_lr_sweep.sbatch"
echo "Acts:   $ACTS"
echo "Cfg:    k=$K d_dict=$D_DICT seed=$SEED aux_k=$AUX_K"
echo "Sweep:  epochs=$SWEEP_EPOCHS max_tokens=$SWEEP_MAX_TOKENS"
echo "  adam LRs: $ADAM_LRS"
echo "  muon LRs: $MUON_LRS  (adam-fallback lr=$ADAM_FALLBACK_LR)"
echo "Final:  epochs=$FINAL_EPOCHS max_tokens=${FINAL_MAX_TOKENS:-ALL}"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# --- Each training run gets a fresh JAX process (no XLA cache cross-talk).
run_in_sif () {
    apptainer exec --nv \
        --env PYTHONPATH="$PROJECT_ROOT" \
        --env PYTHONNOUSERSITE=1 \
        -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
        -B /data:/data \
        "$SIF" bash -c '
            set -euo pipefail
            export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
            cd "$1"; shift
            echo "[exec] $*"
            exec "$@"
        ' bash "$PROJECT_ROOT" "$@"
}

# Build a train_sae command. $1=optimizer $2=lr $3=epochs $4=max_tokens(""=all)
# $5=out_prefix.  Common knobs shared across the whole sweep.
train_cmd () {
    local opt="$1" lr="$2" epochs="$3" maxtok="$4" prefix="$5"
    local -a cmd=( python scripts/train_sae.py
        --activations "$ACTS" --k "$K" --d-dict "$D_DICT"
        --epochs "$epochs" --batch-size "$BATCH" --seed "$SEED"
        --aux-k "$AUX_K" --aux-coef "$AUX_COEF" --aux-steps-to-kill "$AUX_KILL"
        --optimizer "$opt" --lr "$lr" --out-prefix "$prefix" )
    [[ "$opt" == "muon" ]] && cmd+=( --adam-lr "$ADAM_FALLBACK_LR" )
    [[ -n "$maxtok" ]] && cmd+=( --max-tokens "$maxtok" )
    printf '%s\0' "${cmd[@]}"
}

run_train () {  # $1=optimizer $2=lr $3=epochs $4=max_tokens $5=prefix
    mapfile -d '' -t _cmd < <(train_cmd "$@")
    run_in_sif "${_cmd[@]}"
}

# ---------------------------------------------------------------------------
# Phase 1 — independent LR grids
# ---------------------------------------------------------------------------
for lr in $ADAM_LRS; do
    echo "[sweep] === adam lr=$lr ==="
    run_train adam "$lr" "$SWEEP_EPOCHS" "$SWEEP_MAX_TOKENS" \
        "$SWEEP_DIR/adam_lr${lr}"
done
for lr in $MUON_LRS; do
    echo "[sweep] === muon lr=$lr ==="
    run_train muon "$lr" "$SWEEP_EPOCHS" "$SWEEP_MAX_TOKENS" \
        "$SWEEP_DIR/muon_lr${lr}"
done

# ---------------------------------------------------------------------------
# Phase 2 — rank each optimizer's grid, pick the best LR
# ---------------------------------------------------------------------------
# "Best" = most live features (lowest dead_fraction) among runs whose EV is
# within 0.02 of that optimizer's best EV (so we don't reward a high-LR run
# that revived features by trashing reconstruction). Writes BEST_*_LR to an
# env file the shell then sources.
PICK_PY=$(cat <<'PYEOF'
import glob, json, os, sys
sweep_dir = sys.argv[1]
runs = {}
for jf in glob.glob(os.path.join(sweep_dir, "*.json")):
    d = json.load(open(jf))
    c, h = d["config"], d["final_health"]
    opt = c["optimizer"]
    runs.setdefault(opt, []).append(
        dict(lr=c["lr"], ev=h["explained_var"], dead=h["dead_fraction"],
             live=(1.0 - h["dead_fraction"]) * c["d_dict"]))

best = {}
for opt, rs in runs.items():
    rs.sort(key=lambda r: r["lr"])
    ev_max = max(r["ev"] for r in rs)
    eligible = [r for r in rs if r["ev"] >= ev_max - 0.02]
    pick = min(eligible, key=lambda r: (r["dead"], -r["ev"]))
    best[opt] = pick
    print(f"\n=== {opt} LR grid ===")
    print(f"{'lr':>8s}  {'EV':>6s}  {'dead':>7s}  {'live':>7s}")
    for r in rs:
        mark = "  <-- best" if r is pick else ""
        print(f"{r['lr']:>8.2e}  {r['ev']:.3f}  {r['dead']:6.1%}  "
              f"{r['live']:7.0f}{mark}")

with open(os.path.join(sweep_dir, "best.env"), "w") as f:
    if "adam" in best:
        f.write(f"BEST_ADAM_LR={best['adam']['lr']:.6g}\n")
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
# Phase 3 — full-scale head-to-head at the winning LRs
# ---------------------------------------------------------------------------
FINAL_ADAM="$SWEEP_DIR/final_adam"
FINAL_MUON="$SWEEP_DIR/final_muon"
echo "[final] === adam lr=$BEST_ADAM_LR (full) ==="
run_train adam "$BEST_ADAM_LR" "$FINAL_EPOCHS" "$FINAL_MAX_TOKENS" "$FINAL_ADAM"
echo "[final] === muon lr=$BEST_MUON_LR (full) ==="
run_train muon "$BEST_MUON_LR" "$FINAL_EPOCHS" "$FINAL_MAX_TOKENS" "$FINAL_MUON"

CMP_PY=$(cat <<'PYEOF'
import json, sys
adam = json.load(open(sys.argv[1] + ".json"))
muon = json.load(open(sys.argv[2] + ".json"))
def row(tag, d):
    h, s = d["final_health"], d["spectral"]["dec_weight"]
    live = (1.0 - h["dead_fraction"]) * d["config"]["d_dict"]
    return (f"{tag:5s}  lr={d['config']['lr']:.1e}  EV={h['explained_var']:.3f}  "
            f"dead={h['dead_fraction']:.1%}  live={live:7.0f}  l0={h['l0_actual']:.1f}  "
            f"PR={s['participation_ratio']:7.1f}  alpha_hill={s['alpha_hill']:.2f}  "
            f"stable_rank={s['stable_rank']:.1f}")
print("\n=== Best-of-best Muon vs AdamW SAE (dec_weight, full scale) ===")
print(row("adam", adam)); print(row("muon", muon))
da, dm = adam["spectral"]["dec_weight"], muon["spectral"]["dec_weight"]
print(f"\nDelta(PR)  muon-adam = {dm['participation_ratio']-da['participation_ratio']:+.1f}")
print(f"Delta(alpha_hill)     = {dm['alpha_hill']-da['alpha_hill']:+.2f}  (RG ideal ~2.0)")
print(f"Delta(dead_frac)      = {muon['final_health']['dead_fraction']-adam['final_health']['dead_fraction']:+.1%}")
PYEOF
)
echo "[final] === comparison ==="
run_in_sif python -c "$CMP_PY" "$FINAL_ADAM" "$FINAL_MUON"

echo "[done] lr sweep complete. best.env: $BEST_ENV"
echo "  finals: ${FINAL_ADAM}.json  ${FINAL_MUON}.json"
