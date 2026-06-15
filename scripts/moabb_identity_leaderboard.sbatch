#!/usr/bin/env bash
#SBATCH --job-name=fmscope-lb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# Queue the FMScope identity-free MOABB leaderboard sweep through slurm so it
# waits its turn for the single GB10 rather than contending with other GPU work.
# The actual compute runs in the Docker NGC 26.05 container via the launcher;
# requesting the GPU here serializes this job against other slurm GPU jobs.
#
# Usage:
#   sbatch scripts/moabb_identity_leaderboard.sbatch                       # full registry
#   sbatch scripts/moabb_identity_leaderboard.sbatch --datasets Beetl2021-A Beetl2021-B PhysionetMotorImagery
set -euo pipefail
cd "$HOME/dev/emeg-fm"
exec scripts/run_fmscope_leaderboard.sh "$@"
