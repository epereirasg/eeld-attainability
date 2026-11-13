"""Store the SCIP value wherever branch and bound finds a better dispatch than the multistart campaign (10-unit system).

For every cell of results/raw/additional/scip/cells10_p*.csv whose stored value exceeds the SCIP primal value by more
than 1e-4, the script re-solves the cell with SCIP (returning the dispatch), polishes that dispatch with the campaign
solver (SLSQP started from it, plus the usual starts) and stores the better of the two values in the campaign files
(grid_e10*.csv for grid and frontier cells, requests_e10*.csv for the physical, reference, budget-free, band-free and
alternative cells, including the rows of repeated incumbents). The SCIP cell file then records the new stored value.
Every change is logged in results/raw/certificate/scip_adopted.csv, so the certificate is a property of the final data.
"""
import glob
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'additional'))
from common import *
from scip_certificate import scip_gap
from eeldattain.paths import RAW_CERT

COL = {'physical': 'G_phys', 'reference': 'G_ref', 'budget_free': 'G_Rfree', 'band_free': 'G_rhofree', 'alternative': 'G_alt'}
TOL = 1e-4


def main():
    s = System(str(COEFFS['10'])); _, vk = load_requests(distinct=False)
    fcell = sorted(glob.glob(str(RAW_ADD / 'scip' / 'cells10_p*.csv')))
    C = pd.concat([pd.read_csv(f).assign(_f=f) for f in fcell], ignore_index=True)
    key6 = ['hour', 'w', 'cut', 'kind', 'R', 'rho']
    fr = RAW_ADD / 'scip' / 'retry.csv'
    if fr.exists():                    # longer re-solves of cells that stopped at the 10 s limit
        T = pd.read_csv(fr)
        M = C.merge(T[key6 + ['G_lb', 'G_ub']], on=key6, how='left', suffixes=('', '_rt'))
        C['G_ub_best'] = np.fmin(M.G_ub.values, M.G_ub_rt.values)
    else:
        C['G_ub_best'] = C.G_ub
    todo = C[C.G_stored - C.G_ub_best > TOL]
    print('cells to adopt', len(todo), flush=True)
    RQ = {f: pd.read_csv(f, dtype={'S_units': str}) for f in sorted(glob.glob(str(RAW_ATTAIN / 'requests_e10*.csv')))}
    GR = {f: pd.read_csv(f) for f in sorted(glob.glob(str(RAW_ATTAIN / 'grid_e10*.csv')))}
    log = []
    for i, c in todo.iterrows():
        key = (int(c.hour), float(c.w), float(c.cut)); v = vk[key]
        R = BIG if not np.isfinite(c.R) else float(c.R)
        r = scip_gap(s, v, R, float(c.rho), time_limit=60, return_P=True)
        req = make_request(s, v)
        if r['P'] is not None:
            req.pool.insert(0, np.asarray(r['P']))
        Gp, _ = req.gap(R, float(c.rho), np.random.default_rng(1000 + i), k=4)
        G_new = float(min(Gp, r['G_ub'], c.G_stored))      # never above the campaign value
        # the same incumbent appears under every weight that produced it (duplicate rows)
        same = [k for k, vv in vk.items() if k[0] == key[0] and k[2] == key[2] and np.allclose(vv['Pinc'], v['Pinc'], atol=1e-6)]
        for f, D in RQ.items():
            if c.kind in COL:
                for k in same:
                    m = (D.hour == k[0]) & (D.w == k[1]) & (D.cut == k[2])
                    D.loc[m, COL[c.kind]] = G_new
        for f, D in GR.items():
            if c.kind in ('grid', 'budget_frontier'):
                for k in same:
                    m = (D.hour == k[0]) & (D.w == k[1]) & (D.cut == k[2]) & (D.kind == c.kind) & np.isclose(D.R, R) & np.isclose(D.rho, c.rho)
                    D.loc[m, 'G'] = G_new
            if c.kind == 'physical':
                for k in same:
                    m = (D.hour == k[0]) & (D.w == k[1]) & (D.cut == k[2])
                    D.loc[m, 'G_phys'] = G_new
        C.loc[i, 'G_stored'] = G_new
        log.append(dict(hour=key[0], w=key[1], cut=key[2], kind=c.kind, R=c.R, rho=c.rho, G_campaign=c.G_stored,
                        G_scip_ub=r['G_ub'], G_polished=Gp, G_stored_new=G_new, G_lb=c.G_lb, scip_status=r['status']))
        print(key, c.kind, c.R, c.rho, round(c.G_stored, 5), '->', round(G_new, 5), flush=True)
    for f, D in RQ.items(): D.to_csv(f, index=False)
    for f, D in GR.items(): D.to_csv(f, index=False)
    for f, D in C.groupby('_f'): D.drop(columns=['_f', 'G_ub_best']).to_csv(f, index=False)
    RAW_CERT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(RAW_CERT / 'scip_adopted.csv', index=False)
    print('DONE', len(log), flush=True)


if __name__ == '__main__':
    main()
