#!/bin/bash
#SBATCH --job-name=reve-tiny-open-dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Download the TINY open subset of brain-bzh/reve-dataset and lay it out in the
# directory structure REVE's data_loading.py expects, so training with
# data.subset=tiny_open finds it.
#
# The HF "open" dataset is the redistributable half of REVE's corpus (~4.85 TB).
# We pull only a handful of the smallest recordings that have n_chans > 6 (the
# train GroupedSampler drops <=6-channel groups) -> ~2-3 GB, enough to exercise
# the full loader path and run an optimizer sweep without the multi-TB pull.
#
# HF repo layout            ->   REVE loader expects (rooted at data.path=$ROOT)
#   data/recording_-_eeg_-_*.npy     ->  $ROOT/recordings/   (memmap)
#   positions/recording_-_positions_-_*.npy -> $ROOT/positions/  (sibling of recordings)
#   stats/recording_-_stats_-_*.npy  ->  $ROOT/stats/        (sibling of recordings)
#   df_big.csv                       ->  $ROOT/csv_recordings/df_big.csv
#   df_corrected.csv                 ->  $ROOT/csv_recordings/df_corrected.csv
#   df_stats.csv                     ->  $ROOT/csv_recordings/df_stats_tmp.csv  (RENAMED)
# Dirs/CSVs are wired with symlinks (no copy); the full CSVs are small and the
# subset filter keeps only the tiny indices, so listing all 358 rows is fine.

set -euo pipefail

REPO=brain-bzh/reve-dataset
ROOT=/data/derivatives/reve_dataset/tiny_open
STAGING=$ROOT/_staging

# Index split — must match TINY_OPEN_TRAIN/VAL in src/utils/data_loading.py
# (and data.tiny.{train,val} in config_train.yaml). n_chans: 147->8, 140->18,
# 145->24, 144->24, 139->24.
INDICES=(147 140 145 144 139)

mkdir -p "$STAGING" /data/derivatives/eeg_sae/logs

# Build the include list for the LFS .npy files: eeg/positions/stats per index.
INCLUDE=()
for i in "${INDICES[@]}"; do
    INCLUDE+=("data/recording_-_eeg_-_${i}.npy")
    INCLUDE+=("positions/recording_-_positions_-_${i}.npy")
    INCLUDE+=("stats/recording_-_stats_-_${i}.npy")
done

echo "=== hf download $REPO (${#INDICES[@]} recordings) -> $STAGING ==="
# Public dataset (gated: False) -> no token needed. Absolute path so it resolves
# regardless of the compute node's PATH.
HF=${HF_CLI:-$HOME/.local/bin/hf}
[[ -x "$HF" ]] || HF=$(command -v hf)
"$HF" download "$REPO" --repo-type dataset --local-dir "$STAGING" \
    --include "${INCLUDE[@]}"

# Fetch the 3 metadata CSVs by curl, NOT via hf --include: mixing bare-filename
# includes with glob includes made hf silently skip df_big.csv (job 1575). These
# are tiny public files, so a direct resolve is reliable and fast.
echo "=== fetch metadata CSVs (curl) ==="
RESOLVE="https://huggingface.co/datasets/${REPO}/resolve/main"
for c in df_big.csv df_corrected.csv df_stats.csv; do
    curl -fsSL "$RESOLVE/$c" -o "$STAGING/$c"
    echo "  fetched $c ($(wc -l < "$STAGING/$c") lines)"
done

echo
echo "=== build REVE loader layout at $ROOT ==="
mkdir -p "$ROOT/csv_recordings"
ln -sfn "$STAGING/data"      "$ROOT/recordings"
ln -sfn "$STAGING/positions" "$ROOT/positions"
ln -sfn "$STAGING/stats"     "$ROOT/stats"
ln -sfn "$STAGING/df_big.csv"       "$ROOT/csv_recordings/df_big.csv"
ln -sfn "$STAGING/df_corrected.csv" "$ROOT/csv_recordings/df_corrected.csv"
# df_stats.csv -> df_stats_tmp.csv (the name the loader reads)
ln -sfn "$STAGING/df_stats.csv"     "$ROOT/csv_recordings/df_stats_tmp.csv"

echo
echo "=== verify presence ==="
for i in "${INDICES[@]}"; do
    for kind in "recordings/recording_-_eeg_-_${i}.npy" \
                "positions/recording_-_positions_-_${i}.npy" \
                "stats/recording_-_stats_-_${i}.npy"; do
        if [[ -f "$ROOT/$kind" ]]; then
            printf '  OK  %s (%s)\n' "$kind" "$(du -h "$ROOT/$kind" | cut -f1)"
        else
            echo "  MISSING $ROOT/$kind" >&2; exit 1
        fi
    done
done
for c in df_big.csv df_corrected.csv df_stats_tmp.csv; do
    [[ -f "$ROOT/csv_recordings/$c" ]] && echo "  OK  csv_recordings/$c" || { echo "  MISSING csv $c" >&2; exit 1; }
done

echo
echo "=== total on disk ==="
du -sh "$STAGING"
echo
echo "Done. Point training at: data.path=$ROOT data.subset=tiny_open"
