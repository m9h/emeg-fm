#!/usr/bin/env bash
#SBATCH --job-name=lumamba
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
# No --time: gx10-dgx-spark single-user / infinite walltime.
# Queue a LuMamba script (smoke or brain-age) on the single GB10.
# Usage: sbatch scripts/lumamba.sbatch scripts/lumamba_smoke.py [args...]
set -euo pipefail
cd "$HOME/dev/emeg-fm"
exec bash scripts/run_lumamba.sh "$@"
