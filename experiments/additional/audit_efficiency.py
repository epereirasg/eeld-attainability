"""Efficiency loss of the bounded optimum at the audit allowance (250 MW, 0.20) on the stored 10-unit requests.

For every distinct request with a cut of 2.5% or 5%, the cuts of the extended audit, the script solves the gap
problem at the audit allowance and then the cheapest dispatch of the physical set with the same emission. The
efficiency loss is the excess cost of the bounded optimum over that dispatch, in percent. It bounds from above the
cost by which another admissible dispatch could dominate the bounded optimum.
Output: results/raw/additional/audit_efficiency.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from eeldattain.model import eps_point

R_AUDIT, RHO_AUDIT, STARTS = 250.0, 0.20, 8


def main():
    s = System(str(COEFFS['10']))
    R, vk = load_requests('10')
    R = R[R.cut.isin([0.025, 0.05])].reset_index(drop=True)
    rows = []
    for r in R.itertuples():
        v = vk[r.key]; req = make_request(s, v)
        rng = np.random.default_rng(700000 + 1000 * int(r.hour) + int(100 * r.w) + int(1000 * r.cut))
        G, P = req.gap(R_AUDIT, RHO_AUDIT, rng, k=4)
        x0s = [P, req.Pinc, np.array(v['Pfr'])] + [s.rand_feasible(req.PD, rng) for _ in range(STARTS)]
        Peff = eps_point(s, req.PD, s.E(P), x0s)
        loss = 100 * (s.C(P) / s.C(Peff) - 1) if Peff is not None else np.nan
        rows.append(dict(hour=r.hour, w=r.w, cut=r.cut, reachable=bool(r.reachable), G=G,
                         mw=float(np.abs(P - req.Pinc).sum()), eff_loss_pct=loss))
        print(rows[-1], flush=True)
    RAW_ADD.mkdir(parents=True, exist_ok=True)
    D = pd.DataFrame(rows); D.to_csv(RAW_ADD / f'audit_efficiency.csv', index=False)
    print(D.eff_loss_pct.describe())


if __name__ == '__main__':
    main()
