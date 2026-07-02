#!/usr/bin/env bash
# Turnkey MDD-Dx challenge run from the full V3.1 BDF dataset.
#
# Usage:
#   export TDBRAIN_ZIP_PW='<zip password from the DUA email>'
#   scripts/run_tdbrain_mdd_dx.sh /path/to/TDBRAIN_Dataset_V3_1_Encr.zip
#
# Extracts the full dataset to NODE-LOCAL /mnt/t9 (never /data NFS: thousands of
# small BDFs wedge NFS in D-state), auto-detects the tree root (the dir holding
# sub-*), then trains one head per template column on Discovery (183 controls) and
# predicts the blinded MDD-Dx replication set. Discovery + Replication are BOTH
# V3.1 BDF, so --norm none is correct (no cross-format spectral shift; recording-
# zscore was only needed to bridge the legacy-BrainVision Discovery -> BDF gap).
set -euo pipefail

ZIP="${1:?usage: run_tdbrain_mdd_dx.sh <TDBRAIN_Dataset_V3_1_Encr.zip>}"
PW="${TDBRAIN_ZIP_PW:?set TDBRAIN_ZIP_PW to the zip password (single-quote it)}"
PY=/home/mhough/dev/neurojax/.venv-models/bin/python
REPO=/home/mhough/dev/emeg-fm
V31=/data/datasets/tdbrain_v3.1
DEST=/mnt/t9/tdbrain_v3.1_full
NORM="${TDBRAIN_NORM:-none}"   # both trees are BDF -> none; use recording-zscore for legacy BrainVision Discovery

echo ">> extracting Discovery BDF to $DEST (node-local) ..."
mkdir -p "$DEST"
unzip -q -o -P "$PW" "$ZIP" -d "$DEST"

# auto-detect the tree root: the directory that directly contains sub-* dirs
DISC_BIDS="$(dirname "$(find "$DEST" -maxdepth 4 -type d -name 'sub-*' | head -1)")"
echo ">> Discovery BIDS root: $DISC_BIDS"
echo ">>   $(find "$DISC_BIDS" -maxdepth 1 -type d -name 'sub-*' | wc -l) subjects, $(find "$DISC_BIDS" -name '*.bdf' | wc -l) BDFs"

"$PY" "$REPO/scripts/tdbrain_challenge.py" \
  --challenge mdd_dx --norm "$NORM" \
  --discovery-bids   "$DISC_BIDS" \
  --discovery-labels "$V31/TDBRAIN_participants_V3.xlsx" \
  --replication-bids /mnt/t9/tdbrain_v3.1_repl/TDBRAIN_MDD_Dx_ReplicationV3_1 \
  --template         "$V31/TDBRAIN_MDD_Dx_Prediction_replicationV3.xlsx" \
  --out              "$V31/mdd_dx_submission.xlsx"

echo ">> wrote $V31/mdd_dx_submission.xlsx"
