#!/usr/bin/env bash
# Figures 1 and 3 to 9 of the manuscript from the stored raw outputs (results/figures/*.pdf).
# Figure 2 is a TikZ drawing in the manuscript and has no data. No optimization is rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg
mkdir -p results/figures
python3 analysis/figure_problem.py       # Fig. 1, schematic (no data)
python3 analysis/figures_seaborn.py      # Figs. 3, 4 and 7 to 9
python3 analysis/figures_additional.py   # Figs. 5 and 6
echo "figures written to results/figures/"
