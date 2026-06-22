#!/usr/bin/env bash
# Reproducible build (Python/Sweave pattern): tangle the audit CSVs into macros + tables,
# then weave the LaTeX. One command regenerates the manuscript from results/moabb_fmscope/
# *.csv -- nothing in paper.tex is hand-transcribed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TECTONIC="${TECTONIC:-/home/mhough/miniforge3/envs/texlive/bin/tectonic}"
command -v "$TECTONIC" >/dev/null 2>&1 || TECTONIC=tectonic
python "$HERE/generate.py"                                     # tangle
( cd "$HERE" && "$TECTONIC" --keep-logs --reruns 3 paper.tex ) # weave
echo "built paper/paper.pdf"
