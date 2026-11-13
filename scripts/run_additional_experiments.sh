#!/usr/bin/env bash
# Additional experiments, in the order of the stored runs (README.md, "Additional experiments").
# They read the stored campaigns through their stored vectors and write to results/raw/additional/ unless stated.
# Every runner resumes after an interruption. Two lanes run in parallel on two cores.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export EELD_LEGACY_REPO=${EELD_LEGACY_REPO:-external/eeld-cf}
L=results/logs/additional; mkdir -p $L
E=experiments/additional; H=1,3,5,7,9,11,13,15,17,19,21,23

# SCIP checks of the 10-unit anchors, incumbents and targets, of every cell and of every counterfactual
lane_scip() {
  python3 $E/scip_anchors.py > $L/scip_anchors.txt 2>&1
  python3 $E/scip_incumbents_inverse.py > $L/scip_incumbents_inverse.txt 2>&1
  python3 $E/scip_all_cells.py > $L/scip_cells10.txt 2>&1
  python3 $E/scip_retry.py --time-limit 300 > $L/scip_retry.txt 2>&1
  python3 experiments/certificate/adopt_scip.py > $L/adopt_scip.txt 2>&1     # stores SCIP values better by > 1e-4
  python3 $E/scip_counterfactuals.py > $L/scip_counterfactuals.txt 2>&1
  python3 $E/scip_direct.py --time-limit 600 > $L/scip_direct.txt 2>&1          # formulation (22) where the bounds do not decide
  python3 $E/adopt_direct.py > $L/adopt_direct.txt 2>&1                         # stores the two smaller global minima
  python3 $E/scip_ops_anchors.py > $L/scip_ops_anchors.txt 2>&1
  python3 $E/scip_ops_cells.py --time-limit 120 > $L/scip_ops_cells.txt 2>&1
  python3 $E/scip_ops_cells.py --grid ramp --time-limit 20 > $L/scip_ramp_grid.txt 2>&1
  python3 $E/scip_network.py --network dc > $L/scip_network_dc.txt 2>&1
  python3 $E/scip_network.py --network copper > $L/scip_network_copper.txt 2>&1
}

# counterfactual studies on the 10-unit campaign, the model variants and the network case
lane_cf() {
  python3 $E/compliance_cf.py > $L/compliance.txt 2>&1
  python3 $E/compliance_cf.py --eps 0.0283,0.0707 > $L/compliance_match.txt 2>&1   # eps_E = sqrt(2) eps for every request
  python3 $E/compliance_cf.py --calibrated 0.05,0.02 > $L/compliance_calibrated.txt 2>&1   # Proposition 1 tolerance per request
  python3 $E/unit_sets.py > $L/unit_sets.txt 2>&1
  python3 $E/allowance_grid.py > $L/allowance_grid.txt 2>&1
  python3 $E/cap_reach.py > $L/cap_reach.txt 2>&1                               # reach of the dense grid against the cap
  python3 $E/rq3_seeds.py > $L/rq3_seeds.txt 2>&1
  python3 $E/direct_cf.py > $L/direct_cf.txt 2>&1
  python3 $E/explanation_box.py > $L/explanation_box.txt 2>&1
  python3 $E/ops_uniform_relief.py > $L/ops_uniform_relief.txt 2>&1
  for v in convex reserve leastcost perturbed; do
    python3 $E/variant_campaign.py --variant $v --hours $H --certify > $L/variant_$v.txt 2>&1
  done
  python3 $E/variant_campaign.py --variant leastcost --hours 2,4,6,8,10,12,14,16,18,20,22,24 --tag _even --certify > $L/variant_leastcost_even.txt 2>&1
  python3 $E/onesided.py > $L/onesided_p0.txt 2>&1                              # one-sided distance of Proposition 2
  python3 $E/audit_efficiency.py > $L/audit_efficiency.txt 2>&1                  # efficiency loss at the audit allowance
  python3 $E/network30.py --network dc > $L/network_dc.txt 2>&1
  python3 $E/network30.py --network copper > $L/network_copper.txt 2>&1
  python3 $E/network_split.py > $L/network_split.txt 2>&1
  python3 $E/cf_with_methods.py > $L/cf_with_methods.txt 2>&1
  python3 $E/ablation_methods.py > $L/ablation_methods.txt 2>&1
  python3 $E/methods_extended.py --part 0 --parts 2 > $L/methods_extended_p0.txt 2>&1
  python3 $E/methods_extended.py --part 1 --parts 2 > $L/methods_extended_p1.txt 2>&1
  python3 $E/methods_extended.py --variant convex --part 0 --parts 2 > $L/methods_convex_p0.txt 2>&1
  python3 $E/methods_extended.py --variant convex --part 1 --parts 2 > $L/methods_convex_p1.txt 2>&1
}

lane_scip & lane_cf & wait

# follow-ups of the 40-unit campaign of scripts/run_campaign.sh (experiments/additional/forty_main.py)
python3 $E/compliance_cf.py --case 40 > $L/compliance40.txt 2>&1 &
python3 $E/scip_forty.py --part 0 --parts 2 --time-limit 600 > $L/scip_forty_p0.txt 2>&1 &
wait
python3 $E/scip_forty.py --part 1 --parts 2 --time-limit 600 > $L/scip_forty_p1.txt 2>&1
python3 $E/scip_forty_region.py --part 0 --parts 2 > $L/scip_forty_region_p0.txt 2>&1 &   # study-region cells, 120 s each
python3 $E/scip_forty_region.py --part 1 --parts 2 > $L/scip_forty_region_p1.txt 2>&1 &
wait

# printed emission convention of the data source (variant of Appendix B, results/raw/additional/forty_printed/)
python3 $E/forty_main.py --escale 0.01 --hours 1,3,5 --tag e40a > $L/forty_printed_a.txt 2>&1 &
python3 $E/forty_main.py --escale 0.01 --hours 2,4,6 --tag e40b > $L/forty_printed_b.txt 2>&1 &
wait
echo ALLDONE
