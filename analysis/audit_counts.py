"""Numbers that the manuscript states about the extended method audit and about the 40-unit anchors at 10,500 MW.

The audit of Section VI-E rebuilds incumbents and targets with the multistart of the method comparison
(experiments/additional/methods_extended.py), so its instance count at the cuts of 2.5% and 5% differs from the count
of distinct requests of the main campaign. This script records both counts and the attainable gap of the reachable
audit instances under the broad allowance. It also reports the efficiency loss of the bounded optimum at the audit
allowance (experiments/additional/audit_efficiency.py) and compares the 40-unit anchors at 10,500 MW under both emission
conventions with the optima published by Bhattacharjee et al., Int. J. Electr. Power Energy Syst. 59, 2014. Writes results/macros/numbers_audit_counts.tex and
results/processed/numbers_audit_counts.json.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from common import MAC, PROC, load_case
from eeldattain import paths
from eeldattain.model import System

N = {}
d = pd.concat([pd.read_csv(f) for f in sorted((paths.RAW / 'additional' / 'methods_extended').glob('raw_p*.csv'))])
inst = d.drop_duplicates('inst_hash')
reach = inst[inst.reachable.astype(bool)]
N['AuditN'] = str(len(inst))
N['AuditReachN'] = str(len(reach))
N['AuditReachZeroN'] = str(int((reach.G_ref <= 0.01).sum()))       # attainable gap at most the bound tolerance
N['AuditReachAboveEpsN'] = str(int((reach.G_ref > 0.05).sum()))
R, _ = load_case(10)
u = R[R.unique & R.cut.isin([0.025, 0.05])]
N['MainLowCutN'] = str(len(u))
N['MainLowCutReachN'] = str(int(u.reachable.sum()))

# efficiency loss of the bounded optimum at the audit allowance (250 MW, 0.20), main-campaign requests with the audit cuts
e = pd.read_csv(paths.RAW / 'additional' / 'audit_efficiency.csv')
N['AuditEffN'] = str(len(e))
N['AuditEffLossMed'] = f'{e.eff_loss_pct.median():.2f}'
N['AuditEffLossQc'] = f'{e.eff_loss_pct.quantile(0.9):.2f}'
N['AuditEffLossMax'] = f'{e.eff_loss_pct.max():.2f}'

# 40-unit anchors at 10,500 MW against the published optima (minimum emission 176,682 ton/h at 129,995 $/h,
# minimum cost 121,412.54 $/h)
PD, PUB_CMIN, PUB_EMIN, PUB_CEMIN = 10500.0, 121412.5355, 176682.2541, 129995.2695
coeffs = str(paths.COEFFS['40'])
for tag, k in (('Scaled', 0.01), ('Std', 1.0)):
    s = System(coeffs, escale=k); best = None
    for seed in range(4):                       # the emission is convex, the starts only guard the solver
        rng = np.random.default_rng(seed); x0 = s.lo + rng.random(s.n) * s.span; x0 *= PD / x0.sum()
        r = minimize(s.E, x0, jac=s.gE, method='SLSQP', bounds=list(zip(s.lo, s.hi)),
                     constraints=[{'type': 'eq', 'fun': lambda P: P.sum() - PD, 'jac': lambda P: np.ones(len(P))}],
                     options={'ftol': 1e-14, 'maxiter': 2000})
        if best is None or r.fun < best.fun:
            best = r
    N[f'AnchorEmin{tag}'] = f'{best.fun:,.0f}'.replace(',', '{,}')
    if k == 1.0:                                # the constant terms alpha_i shift the emission without moving the optimum
        N['AnchorAlphaSum'] = f'{s.al.sum():,.0f}'.replace(',', '{,}')
        N['AnchorEminNoConstDiff'] = f'{100 * abs(best.fun - s.al.sum() - PUB_EMIN) / PUB_EMIN:.1f}'
        N['AnchorCostDiff'] = f'{100 * abs(s.C(best.x) - PUB_CEMIN) / PUB_CEMIN:.3f}'
    N[f'AnchorCost{tag}'] = f'{s.C(best.x):,.0f}'.replace(',', '{,}')
R40, _ = load_case(40)
cmin = R40[R40.PD == PD].C_min.iloc[0]
N['AnchorCminExcess'] = f'{100 * (cmin / PUB_CMIN - 1):.1f}'

with open(f'{MAC}/numbers_audit_counts.tex', 'w') as f:
    for k in sorted(N):
        f.write(f'\\newcommand{{\\{k}}}{{{N[k]}}}\n')
json.dump(N, open(f'{PROC}/numbers_audit_counts.json', 'w'), indent=1)
print(N)
