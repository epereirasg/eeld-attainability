"""SCIP on the direct formulation (22) for the band and budget counterfactuals whose minimality the bisection bounds leave open.

For a stored 10-unit counterfactual, the bound propagation of analysis/additional.py certifies minimality when the
lower bound of G* one bisection step below exceeds the upper bound of G*_phys plus 0.05. Where the bounds cannot decide,
this script solves the direct problem globally with the most permissive threshold, d(P) <= d(P_inc) (U_phys + 0.05):
    R  : min sum_i |P_i - P_inc,i|          over the physical set,
    rho: min t, |P_i - P_inc,i| <= t Delta_i over the physical set, without a budget.
A global lower bound L above the value one step below (R* - 1000/256 MW or rho* - 0.95/256) proves that no allowance at
that step reaches the threshold, so the stored value is minimal to one step. Output: results/raw/additional/scip/direct.csv
"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from pyscipopt import Model, quicksum, sin, exp
sys.path.insert(0, str(Path(__file__).parent))
from common import *

STEP = {'R': 1000 / 256, 'rho': 0.95 / 256}


def open_cases():
    C = pd.read_csv(RAW_ADD / 'scip' / 'counterfactuals.csv')
    D = pd.concat([pd.read_csv(f) for f in glob.glob(str(RAW_ADD / 'scip' / 'cells10_p*.csv'))])
    P = D[D.kind == 'physical'].copy()
    P['G_ub'] = np.fmin(P.G_ub, P.G_stored)          # the stored value is attained by a feasible dispatch
    P = P[['hour', 'w', 'cut', 'G_lb', 'G_ub']].rename(columns={'G_lb': 'Lp', 'G_ub': 'Up'})
    M = C.merge(P, on=['hour', 'w', 'cut'])
    ins = M.lb_below - M.Up > EPS
    first = (M.cf == 'rho') & (M.value - STEP['rho'] <= 0.05 + 1e-9)
    return M[~ins & ~first].reset_index(drop=True)


def scip_direct(sysm, v, cf, thr, time_limit):
    Pinc = np.array(v['Pinc']); PD = float(v['PD']); Ft = np.array(v['Ft']); sc = np.array(v['scale'])
    n = sysm.n; d0 = float(np.linalg.norm((sysm.F(Pinc) - Ft) / sc))
    m = Model(); m.hideOutput()
    m.setParam('timing/clocktype', 1); m.setParam('limits/time', time_limit); m.setParam('limits/gap', 1e-6)
    P = [m.addVar(lb=sysm.lo[i], ub=sysm.hi[i], name=f'P{i}') for i in range(n)]
    m.addCons(quicksum(P) == PD)
    u = [m.addVar(lb=0, ub=sysm.span[i], name=f'u{i}') for i in range(n)]
    for i in range(n):
        m.addCons(u[i] >= P[i] - Pinc[i]); m.addCons(u[i] >= Pinc[i] - P[i])
    if cf == 'rho':
        t = m.addVar(lb=0, ub=1, name='t')
        for i in range(n):
            m.addCons(u[i] <= t * float(sysm.span[i]))
        m.setObjective(t, 'minimize')
    else:
        m.setObjective(quicksum(u), 'minimize')
    vp = []
    for i in range(n):
        if sysm.e[i] == 0:
            continue
        wa = m.addVar(lb=sysm.f[i] * (sysm.lo[i] - sysm.hi[i]), ub=0.0, name=f'w{i}')
        m.addCons(wa == sysm.f[i] * (sysm.lo[i] - P[i]))
        z = m.addVar(lb=-1, ub=1, name=f'z{i}'); m.addCons(z == sin(wa))
        y = m.addVar(lb=0, ub=1, name=f'y{i}'); m.addCons(y == abs(z))
        vp.append(sysm.e[i] * y)
    Cexpr = quicksum(sysm.a[i] * P[i] * P[i] + sysm.b[i] * P[i] + sysm.c[i] for i in range(n)) + quicksum(vp)
    Eexpr = quicksum(sysm.k * (sysm.al[i] + sysm.be[i] * P[i] + sysm.ga[i] * P[i] * P[i]) for i in range(n)) + \
        quicksum(sysm.ze[i] * exp(sysm.la[i] * P[i]) for i in range(n))
    xv = m.addVar(lb=None, name='x'); yv = m.addVar(lb=None, name='y')
    m.addCons(xv == (Cexpr - Ft[0]) / sc[0]); m.addCons(yv == (Eexpr - Ft[1]) / sc[1])
    m.addCons(xv * xv + yv * yv <= (thr * d0) ** 2)
    t0 = time.perf_counter(); m.optimize(); el = time.perf_counter() - t0
    lb = m.getDualbound(); ub = m.getPrimalbound() if m.getNSols() > 0 else np.inf
    return dict(L=lb, U=ub, status=m.getStatus(), time_s=el)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--time-limit', type=float, default=600.0)
    a = ap.parse_args()
    s = System(str(COEFFS['10'])); _, vk = load_requests('10')
    M = open_cases(); out = RAW_ADD / 'scip' / 'direct.csv'
    rows = pd.read_csv(out).to_dict('records') if out.exists() else []
    done = {(r['hour'], r['w'], r['cut'], r['cf']) for r in rows}
    for r in M.itertuples():
        if (r.hour, r.w, r.cut, r.cf) in done:
            continue
        thr = r.Up + EPS
        res = scip_direct(s, vk[(int(r.hour), float(r.w), float(r.cut))], r.cf, thr, a.time_limit)
        below = r.value - STEP[r.cf]
        rows.append(dict(hour=r.hour, w=r.w, cut=r.cut, cf=r.cf, value=r.value, below=below, thr=thr, **res,
                         minimal=bool(res['L'] > below + 1e-9)))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(r.hour, r.w, r.cut, r.cf, round(r.value, 4), round(below, 4), round(res['L'], 4), round(res['U'], 4),
              res['status'], round(res['time_s']), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
