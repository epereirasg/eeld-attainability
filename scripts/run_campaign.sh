#!/usr/bin/env bash
# Full experimental campaign, as run for the stored results (two worker processes at a time).
# Measured wall times on two cores are given per block; the total is about 7 hours of wall time
# (about 13 CPU-hours) plus 3 hours for the operational extension.
#
# Outputs go to results/raw/<experiment>/ by default. To rerun without touching the stored results,
# set OUT to a scratch directory, e.g.  OUT=results/rerun scripts/run_campaign.sh
# Runners refuse to resume into a directory that holds a different configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
# BLAS threads change the end points of multistart solves on the valve-point landscape. The stored runs
# used two OpenBLAS threads (the default on the two-core machine), except X7 (sets40), which used one.
# Keep these values to reproduce the stored results; set THREADS=1 for runs that do not depend on the core count.
THREADS=${THREADS:-2}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MPLBACKEND=Agg
OUT=${OUT:-results/raw}
LOG=${LOG:-results/logs}
mkdir -p "$LOG"/{attainability,certificate,robustness,methods_dfo,operational}
E=experiments

# RQ1-RQ3, RQ7: attainability campaign (EELD10 about 2 h per worker)
python3 $E/attainability/run.py --case 10 --hours 1,3,5,7,9,11,13,15,17,19,21,23 --tag e10a --out $OUT/attainability > $LOG/attainability/log_e10a.txt 2>&1 &
python3 $E/attainability/run.py --case 10 --hours 2,4,6,8,10,12,14,16,18,20,22,24 --tag e10b --out $OUT/attainability > $LOG/attainability/log_e10b.txt 2>&1 &
wait
# 40-unit campaign: emission without the factor 1e-2, four starts, eight bisection steps, SCIP checks (one thread)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $E/additional/forty_main.py --hours 1,3,5 --tag e40a --out $OUT/attainability > $LOG/attainability/log_e40a.txt 2>&1 &
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $E/additional/forty_main.py --hours 2,4,6 --tag e40b --out $OUT/attainability > $LOG/attainability/log_e40b.txt 2>&1 &
wait

# RQ4: solver certificate (structural checks on all cells, differential evolution on a stratified sample).
# --grid-order fixes the sampled cells; the stored samples used e10a,e10b and e40b,e40a.
python3 $E/certificate/run.py --case 10 --frac 0.05 --part 0 --parts 1 --grid-order e10a,e10b --campaign $OUT/attainability --out $OUT/certificate > $LOG/certificate/log_c10.txt 2>&1 &
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $E/certificate/run.py --case 40 --frac 0.05 --maxiter 250 --pop 15 --part 0 --parts 2 --grid-order e40b,e40a --campaign $OUT/attainability --out $OUT/certificate > $LOG/certificate/log_c40a.txt 2>&1 &
wait
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $E/certificate/run.py --case 40 --frac 0.05 --maxiter 250 --pop 15 --part 1 --parts 2 --grid-order e40b,e40a --campaign $OUT/attainability --out $OUT/certificate > $LOG/certificate/log_c40b.txt 2>&1

# RQ5-RQ6: method comparison (needs the earlier eeld-cf code base, see scripts/fetch_legacy_repo.sh)
scripts/fetch_legacy_repo.sh
python3 $E/robustness/run.py --hours 1,10,16,22 --tag A --out $OUT/robustness > $LOG/robustness/log_A.txt 2>&1 &
python3 $E/robustness/run.py --hours 4,13,19 --tag B --out $OUT/robustness > $LOG/robustness/log_B.txt 2>&1 &
wait
python3 $E/robustness/run.py --hours 7 --tag C_hour7 --out $OUT/robustness > $LOG/robustness/log_C_hour7.txt 2>&1
python3 $E/methods_dfo/run.py --hours 1,4,7,10,13,16,19,22 --tag all --out $OUT/methods_dfo > $LOG/methods_dfo/log_all.txt 2>&1

# Operational extension X1-X4, X7 (about 3 h on two cores)
python3 $E/operational/run.py --exp ops --out $OUT/operational > $LOG/operational/ops_a.log 2>&1
python3 $E/operational/run.py --exp window --hours 2-23 --out $OUT/operational > $LOG/operational/window_a.log 2>&1 &
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $E/operational/run.py --exp sets40 --attain-dir $OUT/attainability --out $OUT/operational > $LOG/operational/sets40_a.log 2>&1 &
wait
python3 $E/operational/run.py --exp seeds --hours 1-12 --tag a --out $OUT/operational > $LOG/operational/seeds_a.log 2>&1 &
python3 $E/operational/run.py --exp seeds --hours 13-24 --tag b --out $OUT/operational > $LOG/operational/seeds_b.log 2>&1 &
wait
python3 $E/auxiliary/daily_optima.py --out $OUT/daily_optima.json      # Appendix B, emission convention
echo "campaign finished"
