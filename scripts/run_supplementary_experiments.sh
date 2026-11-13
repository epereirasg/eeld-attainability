#!/usr/bin/env bash
# Supplementary experiments: window dissection (RQ10), oracle-limited comparison (RQ5c), stylized 40-unit
# operational replication with its seeds and certificate (RQ4, RQ9), exhaustive 40-unit pairs (RQ11),
# objective scales (Table 8) and the anchor checks. About 9 CPU-hours in total.
#
# Outputs go to results/raw/supplementary/ by default. To rerun without touching the stored results, set OUT and
# LOG, e.g.  OUT=results/rerun/supplementary LOG=results/rerun/logs/supplementary scripts/run_supplementary_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg
# Thread counts match the stored runs each experiment builds on: two threads for the campaign and the
# method-comparison instances, one thread for the 40-unit operational sets (X7). See README.md.
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"; T2="env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2"
OUT=${OUT:-results/raw/supplementary}; LOG=${LOG:-results/logs/supplementary}; mkdir -p "$OUT" "$LOG"
E=experiments/supplementary
scripts/fetch_legacy_repo.sh                                                                  # needed by oracle.py
$T2 python3 $E/window_lock.py --out $OUT > $LOG/window_lock.txt 2>&1 &                        # RQ10, about 20 min
$T2 python3 $E/oracle.py --out $OUT > $LOG/oracle.txt 2>&1 &                                  # RQ5c
wait
$T1 python3 $E/oracle.py --out $OUT/oracle_t1 > $LOG/oracle_t1.txt 2>&1                      # RQ5c, one-thread sensitivity
$T1 python3 $E/ops40.py --out $OUT > $LOG/ops40.txt 2>&1 &                                    # RQ9, one request per level
$T1 python3 $E/pairs40.py --out $OUT > $LOG/pairs40.txt 2>&1 &                                # RQ11, one request per level
wait
$T1 python3 $E/pairs40.py --per-level 4 --part 0 --parts 2 --out $OUT/pairs40_full > $LOG/pairs40_full_p0.txt 2>&1 &   # RQ11, all requests
$T1 python3 $E/pairs40.py --per-level 4 --part 1 --parts 2 --out $OUT/pairs40_full > $LOG/pairs40_full_p1.txt 2>&1 &
wait
# Full stylized 40-unit operational replication, seed replication and DE certificate (about 3 CPU-hours)
$T1 python3 $E/ops40.py --full --part 0 --parts 2 --out $OUT > $LOG/ops40full_p0.txt 2>&1 &
$T1 python3 $E/ops40.py --full --part 1 --parts 2 --out $OUT > $LOG/ops40full_p1.txt 2>&1 &
wait
V=$OUT/ops40full_p0_vectors.jsonl,$OUT/ops40full_p1_vectors.jsonl
$T1 python3 $E/ops40_seeds.py --vectors $V --out $OUT > $LOG/ops40_seeds.txt 2>&1           # one process, as stored
for k in 0 1; do                                                                              # one certificate run per half
  $T1 python3 $E/certificate_ops40.py --results $OUT/ops40full_p$k.csv --vectors $OUT/ops40full_p${k}_vectors.jsonl \
      --name certificate_ops40_p$k --out $OUT > $LOG/certificate_ops40_p$k.txt 2>&1 &
done
wait
$T2 python3 $E/scale_cf.py --part 0 --parts 2 --out $OUT > $LOG/scale_cf_p0.txt 2>&1 &       # Table 8, objective scales
$T2 python3 $E/scale_cf.py --part 1 --parts 2 --out $OUT > $LOG/scale_cf_p1.txt 2>&1 &
wait
echo "supplementary experiments finished"
