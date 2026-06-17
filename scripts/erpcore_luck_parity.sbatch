#!/usr/bin/env bash
#SBATCH --job-name=erpcore-luck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# Queue the ERP CORE Luck-parity decoding comparison through slurm so it waits
# its turn for the single GB10. REVE extraction runs on GPU inside the Docker
# NGC 26.05 container; the pseudo-ERP SVM grid is CPU-bound (hence 12 cores).
#
# Usage:
#   sbatch scripts/erpcore_luck_parity.sbatch                       # all 7 components
#   sbatch scripts/erpcore_luck_parity.sbatch --components N170 P3 --max-subjects 10
set -euo pipefail
cd "$HOME/dev/emeg-fm"
exec scripts/run_erpcore_luck_parity.sh "$@"
