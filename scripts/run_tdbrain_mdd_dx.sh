#!/usr/bin/env bash
# Turnkey MDD-Dx challenge run once the V3.1 Discovery BDF is downloaded.
#
# Usage:
#   export TDBRAIN_ZIP_PW='<zip password from the DUA email>'
#   scripts/run_tdbrain_mdd_dx.sh /path/to/TDBRAIN_DiscoveryV3_1_Encr.zip
#
# Extracts the Discovery set to NODE-LOCAL /mnt/t9 (never /data NFS: thousands of
# small BDFs wedge NFS in D-state), auto-detects the tree root (the dir holding
# sub-*), then trains one head per template column on Discovery and predicts the
# blinded MDD-Dx replication set. Discovery + Replication are both V3.1 BDF here,
# so there is no cross-format spectral shift (unlike the legacy BrainVision tree).
set -euo pipefail

ZIP="${1:?usage: run_tdbrain_mdd_dx.sh <TDBRAIN_DiscoveryV3_1_Encr.zip>}"
PW="${TDBRAIN_ZIP_PW:?set TDBRAIN_ZIP_PW to the zip password (single-quote it)}"
PY=/home/mhough/dev/neurojax/.venv-models/bin/python
REPO=/home/mhough/dev/emeg-fm
V31=/data/datasets/tdbrain_v3.1
DEST=/mnt/t9/tdbrain_v3.1_disc

echo ">> extracting Discovery BDF to $DEST (node-local) ..."
mkdir -p "$DEST"
unzip -q -o -P "$PW" "$ZIP" -d "$DEST"

# auto-detect the tree root: the directory that directly contains sub-* dirs
DISC_BIDS="$(dirname "$(find "$DEST" -maxdepth 4 -type d -name 'sub-*' | head -1)")"
echo ">> Discovery BIDS root: $DISC_BIDS"
echo ">>   $(find "$DISC_BIDS" -maxdepth 1 -type d -name 'sub-*' | wc -l) subjects, $(find "$DISC_BIDS" -name '*.bdf' | wc -l) BDFs"

"$PY" "$REPO/scripts/tdbrain_challenge.py" \
  --challenge mdd_dx \
  --discovery-bids   "$DISC_BIDS" \
  --discovery-labels "$V31/TDBRAIN_participants_V3.xlsx" \
  --replication-bids /mnt/t9/tdbrain_v3.1_repl/TDBRAIN_MDD_Dx_ReplicationV3_1 \
  --template         "$V31/TDBRAIN_MDD_Dx_Prediction_replicationV3.xlsx" \
  --out              "$V31/mdd_dx_submission.xlsx"

echo ">> wrote $V31/mdd_dx_submission.xlsx"
