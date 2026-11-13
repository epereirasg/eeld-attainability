# Evaluation criteria of the campaign

This document states the criteria behind the outcomes of Table 14 of the article. The article groups the identifiers below into six objectives, O1 = RQ1, O2 = RQ2, O3 = RQ3 and RQ7, O4 = RQ4, O5 = RQ5 and RQ6, O6 = RQ8 to RQ11.

## Notation shared by all questions

- A request is the tuple (hour h, incumbent P_inc, target F_tar). The incumbent is efficient for a weight w. The target asks for an emission cut c.
- The allowance is (R_max, rho). The bounded set A_h(R_max, rho) contains every dispatch that meets power balance, generator limits, the total redispatch budget R_max and the unit band rho.
- The target distance is d(P) = ||(F(P) - F_tar) / s||_2, with the target-linked scale s = |F_tar - F(P_inc)| (floored at 1e-6 |F(P_inc)|). The incumbent distance is e0 = d(P_inc).
- Attainability is a property of the request and the allowance. It is G*(R_max, rho) = min over A_h of d(P) / e0. The physical attainability is G*_phys = G*(infinity, 1), taken over the full feasible set.
- Attainment is a property of a method m. It is G_m = d(P_m) / e0 for the dispatch P_m that the method returns.
- The allowance gap is G*_allow = G*(R_max, rho) - G*_phys. The attainment shortfall of a method is Delta_m = G_m - G*(R_max, rho).
- The emission reach is eta_E = min(1, (E_inc - E_min) / (E_inc - E_tar)). It needs only the minimum-emission dispatch and does not depend on the target cost or the scale.

## Mathematical guarantees versus empirical evidence

| Statement | Status | Consequence for the campaign |
|---|---|---|
| G* is non-increasing in R_max and in rho | Guaranteed, because the bounded sets are nested | Any violation in solver output signals a solver failure, never a finding |
| 0 <= G* <= 1 and G*_allow >= 0 | Guaranteed, because the incumbent is feasible | Values outside the range are solver failures |
| Bisection returns the minimal rho* or R* | Guaranteed only for exact G* | Its validity rests on the solver certificate (RQ4) |
| SLSQP multistart returns the global G* | Not guaranteed for the nonconvex valve-point problem | Certified empirically against differential evolution |
| A method's ranking generalizes beyond the tested instances | Not guaranteed | Claimed only within the tested design, with robustness configurations |

## RQ1 Physical attainability across the load profile

| Item | Specification |
|---|---|
| Definition | eta_E per request and the front emission width W_h = (E(P_cmin) - E_min) / E(P_cmin) per hour |
| Primary metric | Mean eta_E per hour and cut |
| Claim | Physical shortfall concentrates in hours with a narrow front |
| Hypotheses | Not tested statistically. The benchmark is deterministic, so the values are exact up to solver tolerance |
| Unit and randomness | Hour-level quantities. Randomness enters only through solver starts |
| Reference | Minimum-emission and minimum-cost dispatches certified by 100-start SLSQP and differential evolution |
| Acceptance | Spearman correlation between W_h and mean eta_E across the 24 hours of at least 0.6. Otherwise the claim is reported as unsupported |

## RQ2 Size and structure of the allowance gap

| Item | Specification |
|---|---|
| Definition | G*_allow on the grid R_max in {10, 25, 50, 100, 250} MW and rho in {0.02, 0.05, 0.10, 0.20} |
| Primary metric | Per-request mean of G*_allow over the realistic region R_max in {25, 50, 100}, rho in {0.05, 0.10, 0.20} |
| Claims | (a) The gap is material. (b) The budget saturates. (c) The budget and the band bind jointly |
| Hypotheses | Descriptive over a full factorial design. Bootstrap intervals over requests describe heterogeneity, not a population |
| Unit and randomness | Request (hour, weight, cut), deduplicated when two weights give the same incumbent |
| Reference | Certified G* (RQ4) |
| Acceptance | (a) At least 50% of requests have a mean G*_allow of at least 0.10, with the lower 95% bootstrap bound above 40%. (b) The mean change in G* from 100 to 250 MW is at most 0.02 for rho at most 0.10. (c) Relaxing the budget alone and relaxing the band alone each remove less than 70% of the gap in most requests |

## RQ3 Allowance counterfactuals

| Item | Specification |
|---|---|
| Definition | rho*(eps) = min rho such that G*(infinity, rho) <= G*_phys + eps. R*(eps) is defined the same way with rho = 1. S*(eps) is the smallest set of units whose bands, once relaxed, meet the same condition. The default is eps = 0.05 |
| Primary metrics | rho*, R* and the size of S* |
| Claims | The counterfactuals are reproducible. On EELD10, few units explain the band gap |
| Hypotheses | Not tested statistically. Reproducibility is checked against tolerances |
| Unit and randomness | Request with a positive allowance gap. Randomness comes from solver seeds |
| Reference | One-at-a-time screening is replaced by an exact or relaxed minimum-cardinality formulation, validated against exhaustive search on EELD10 |
| Acceptance | Differences across seeds are within the bisection resolution (0.004 for rho, 8 MW for R). The size of S* is at most 2 in at least 50% of EELD10 requests. On EELD40 the sparsity claim is made only if the same rule holds |

## RQ4 Solver reliability and the attainability certificate

| Item | Specification |
|---|---|
| Definition | Agreement between the campaign solver and an independent global method on a stratified sample of cells |
| Primary metrics | Maximum of (G_campaign - G_DE), the count of monotonicity violations larger than 0.01, and the fraction of cells reproduced across two seeds |
| Claim | The reported G* values are optima for practical purposes |
| Hypotheses | Not a statistical claim. It is a certificate with explicit tolerances |
| Unit and randomness | Grid cell. Randomness comes from solver starts and DE seeds |
| Reference | Differential evolution with polishing, plus a second multistart seed |
| Acceptance | DE never improves by more than 0.01 on a stratified sample of at least 5% of cells. There are zero monotonicity violations. Cells that fail are re-solved, and if they still fail they are flagged and excluded from claims |

## RQ5 Attainment of methods relative to the attainable reference and to each other

| Item | Specification |
|---|---|
| Definition | Delta_m for the repaired procedures (POLY, ELLP, CGM), the EELD-adapted counterfactual baselines (GS-R, FACE-R, NICE-R, DAACE-R, DiCE-R) and general derivative-free optimizers (COBYQA, CMA-ES) |
| Primary metric | Paired difference in G_m between the best repaired procedure and the best baseline of each family |
| Claims | Repaired procedures attain more than counterfactual baselines at efficient incumbents. General derivative-free optimizers attain more than repaired procedures |
| Hypotheses | H0 says the median paired difference is zero. H1 is two-sided. Wilcoxon signed-rank tests with Holm correction across configurations, Hodges-Lehmann estimates and matched-pairs rank-biserial correlation |
| Unit and randomness | Instance (request and allowance). Method seeds are replicated |
| Reference | The certified G*, so that every method is also reported as a shortfall from what is attainable |
| Acceptance | A comparison is claimed in general only if it holds (Holm-adjusted p < 0.05 and |HL| >= 0.05) in at least 7 of the 9 robustness configurations. Otherwise it is stated together with the conditions under which it holds |

## RQ6 Feasibility versus objective performance

| Item | Specification |
|---|---|
| Definition | Independent audit of power balance (1e-4 MW), generator limits, budget and unit bands (1e-6) for every returned dispatch |
| Primary metrics | Feasibility rate, and the rate of outputs worse than the incumbent (G_m > 1) |
| Claim | Feasibility is reported separately and never mixed into attainment |
| Rule | An infeasible output counts as failed. Attainment is reported both raw and with an incumbent fallback (min(G_m, 1)) |
| Acceptance | All methods are reported under both rules. A ranking that changes between the rules is reported as conditional |

## RQ7 Plausibility versus the size of the redispatch

| Item | Specification |
|---|---|
| Definition | Plausibility here means physical feasibility plus efficiency, measured by the cost excess over the front at equal emission. Redispatch size means MW moved (L1), ErrX and the number of units moved by more than 0.5 MW |
| Primary metric | The attainment-intervention frontier, the minimum G* reachable for each MW budget |
| Claim | Bounded recommendations reach most of a reachable request with a fraction of the redispatch that a full re-solve uses |
| Hypotheses | Descriptive. Reported with exact values per request |
| Acceptance | The claim names a fraction only if the median over requests supports it |
