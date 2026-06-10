#!/bin/bash
#SBATCH --job-name=muon-sae-bakeoff
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Scope A of the Muon experiment (docs/MUON_EXPERIMENT.md): train two TopK SAEs
# on the SAME activations — one with AdamW, one with Muon — and compare their
# dictionary health (live-feature yield, EV) and HTSR spectral summary
# (participation ratio, alpha_hill). Both runs share seed / d_dict / k / epochs,
# so the optimizer is the only difference. Single GPU, FIFO: adam then muon.
#
# Usage:
#   ACTS=/data/derivatives/eeg_sae/acts/brain-bzh_reve-base_L-1_EEG2025R1_RestingState.npz \
#     sbatch scripts/muon_sae_bakeoff.sbatch
#
# Optional env knobs (defaults match the README REVE right-sizing run):
#   D_DICT=4096 K=32 EPOCHS=20 BATCH=4096
#   ADAM_LR=1e-3                # AdamW LR (also Muon's adam-fallback LR)
#   MUON_LR=2e-2                # Muon matrix-update LR — tune separately!
#   AUX_K=32                    # Gao 2024 aux_k; 0 to disable

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/emeg-fm
SIF=/data/derivatives/containers/jax-arm64-cu13.sif
OUT_DIR=/data/derivatives/eeg_sae

ACTS=${ACTS:?"set ACTS=/path/to/activations.npz"}
K=${K:-32}
D_DICT=${D_DICT:-4096}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-4096}
ADAM_LR=${ADAM_LR:-1e-3}
MUON_LR=${MUON_LR:-2e-2}
AUX_K=${AUX_K:-32}
AUX_COEF=${AUX_COEF:-0.03125}
AUX_KILL=${AUX_KILL:-200}
MAX_TOKENS=${MAX_TOKENS:-}
SEED=${SEED:-0}

ACTS_TAG=$(basename "$ACTS" .npz)
BASE="$OUT_DIR/sae/${ACTS_TAG}_k${K}_dd${D_DICT}_bakeoff"
ADAM_PREFIX="${BASE}_adam"
MUON_PREFIX="${BASE}_muon"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/sae"

[[ -f "$SIF"  ]] || { echo "ERROR: $SIF not found (build_jax_arm64_sif.sbatch)." >&2; exit 1; }
[[ -f "$ACTS" ]] || { echo "ERROR: activations not found: $ACTS" >&2; exit 1; }

echo "Job:    muon_sae_bakeoff.sbatch"
echo "Acts:   $ACTS"
echo "Cfg:    k=$K d_dict=$D_DICT epochs=$EPOCHS batch=$BATCH seed=$SEED aux_k=$AUX_K"
echo "Adam:   lr=$ADAM_LR  -> ${ADAM_PREFIX}.{npz,json}"
echo "Muon:   lr=$MUON_LR (adam-fallback lr=$ADAM_LR) -> ${MUON_PREFIX}.{npz,json}"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Common args shared by both runs. Built in the OUTER shell to dodge the
# array-in-quoted-bash-c quoting bug (drops `--d-dict 4096` to `--d-dict`).
COMMON=( --activations "$ACTS" --k "$K" --d-dict "$D_DICT"
    --epochs "$EPOCHS" --batch-size "$BATCH" --seed "$SEED"
    --aux-k "$AUX_K" --aux-coef "$AUX_COEF" --aux-steps-to-kill "$AUX_KILL" )
[[ -n "$MAX_TOKENS" ]] && COMMON+=( --max-tokens "$MAX_TOKENS" )

ADAM_CMD=( python scripts/train_sae.py "${COMMON[@]}"
    --optimizer adam --lr "$ADAM_LR" --out-prefix "$ADAM_PREFIX" )
MUON_CMD=( python scripts/train_sae.py "${COMMON[@]}"
    --optimizer muon --lr "$MUON_LR" --adam-lr "$ADAM_LR" --out-prefix "$MUON_PREFIX" )

# Compare the two JSONs at the end. Heredoc'd into a temp script so the
# apptainer bash -c payload stays simple.
CMP_PY=$(cat <<'PYEOF'
import json, sys
adam = json.load(open(sys.argv[1] + ".json"))
muon = json.load(open(sys.argv[2] + ".json"))
def row(tag, d):
    h, s = d["final_health"], d["spectral"]["dec_weight"]
    live = (1.0 - h["dead_fraction"]) * d["config"]["d_dict"]
    return (f"{tag:5s}  EV={h['explained_var']:.3f}  dead={h['dead_fraction']:.1%}  "
            f"live={live:7.0f}  l0={h['l0_actual']:.1f}  "
            f"PR={s['participation_ratio']:7.1f}  alpha_hill={s['alpha_hill']:.2f}  "
            f"stable_rank={s['stable_rank']:.1f}")
print("\n=== Muon vs AdamW SAE bakeoff (dec_weight) ===")
print(row("adam", adam))
print(row("muon", muon))
da, dm = adam["spectral"]["dec_weight"], muon["spectral"]["dec_weight"]
print(f"\nDelta(PR)  muon-adam = {dm['participation_ratio']-da['participation_ratio']:+.1f}")
print(f"Delta(alpha_hill)     = {dm['alpha_hill']-da['alpha_hill']:+.2f}  (RG ideal ~2.0)")
print(f"Delta(dead_frac)      = {muon['final_health']['dead_fraction']-adam['final_health']['dead_fraction']:+.1%}")
PYEOF
)

# Each step is its own apptainer exec — simplest correct quoting, and each
# training run gets a fresh JAX process (no XLA cache cross-talk between the
# adam and muon runs). Same env contract as train_sae.sbatch.
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

echo "[bakeoff] === AdamW run ==="
run_in_sif "${ADAM_CMD[@]}"
echo "[bakeoff] === Muon run ==="
run_in_sif "${MUON_CMD[@]}"
echo "[bakeoff] === comparison ==="
run_in_sif python -c "$CMP_PY" "$ADAM_PREFIX" "$MUON_PREFIX"

echo "[done] bakeoff complete:"
echo "  adam: ${ADAM_PREFIX}.json"
echo "  muon: ${MUON_PREFIX}.json"
