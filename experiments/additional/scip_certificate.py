"""Global certification of stored 10-unit cells with SCIP (spatial branch and bound).

For a cell (request, R, rho) it solves the bounded problem of Eq. (gap) exactly as stated, with the
valve-point term |e_i sin(f_i (Pmin_i - P_i))| modelled through SCIP's sine and absolute-value expressions,
and returns a certified lower bound and the best primal value of G*(R, rho). A cell is certified when the
stored campaign value lies within 0.01 of the global lower bound (the tolerance of the empirical certificate).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from pyscipopt import Model, quicksum, sin, exp
sys.path.insert(0, str(Path(__file__).parent))
from common import *


def scip_gap(sysm, v, R, rho, time_limit=600, gap_tol=1e-4, convex=False, lines=None, return_P=False):
    """Global bounds on G*(R, rho) for one stored request. ``lines`` = (Hg, f0, rate) adds DC line limits.

    The sine argument of the valve-point term is an explicit bounded variable and the objective uses scaled
    residuals. With the argument written as an expression and unscaled cost and emission variables, SCIP 10
    returned lower bounds above the values of feasible SLSQP dispatches in some cells.
    """
    Pinc = np.array(v['Pinc']); PD = float(v['PD']); Ft = np.array(v['Ft']); sc = np.array(v['scale'])
    n = sysm.n
    r = np.broadcast_to(np.asarray(rho, float), (n,))
    lo = np.maximum(sysm.lo, Pinc - r * sysm.span); hi = np.minimum(sysm.hi, Pinc + r * sysm.span)
    m = Model(); m.hideOutput()
    m.setParam('timing/clocktype', 1)          # CPU time, so that a busy machine does not shorten the limit
    m.setParam('limits/time', time_limit); m.setParam('limits/gap', gap_tol)
    d0 = float(np.linalg.norm((sysm.F(Pinc) - Ft) / sc))
    m.setParam('limits/absgap', 1e-4 * d0 ** 2)     # stops when the bound on G* is resolved to about 1e-4
    P = [m.addVar(lb=lo[i], ub=hi[i], name=f'P{i}') for i in range(n)]
    m.addCons(quicksum(P) == PD)
    if R < BIG:
        u = [m.addVar(lb=0, ub=hi[i] - lo[i], name=f'u{i}') for i in range(n)]
        for i in range(n):
            m.addCons(u[i] >= P[i] - Pinc[i]); m.addCons(u[i] >= Pinc[i] - P[i])
        m.addCons(quicksum(u) <= R)
    if lines is not None:
        Hg, f0, rate = lines
        for l in range(len(rate)):
            if np.all(np.abs(Hg[l]) < 1e-12):
                continue                                  # flow fixed by the loads, not by the dispatch
            fl = quicksum(float(Hg[l, i]) * P[i] for i in range(n) if Hg[l, i] != 0) + float(f0[l])
            m.addCons(fl <= float(rate[l])); m.addCons(fl >= -float(rate[l]))
    vp = []
    for i in range(n):
        if convex or sysm.e[i] == 0:
            continue
        # explicit bounded sine argument (see the docstring)
        wa = m.addVar(lb=sysm.f[i] * (sysm.lo[i] - hi[i]), ub=sysm.f[i] * (sysm.lo[i] - lo[i]), name=f'w{i}')
        m.addCons(wa == sysm.f[i] * (sysm.lo[i] - P[i]))
        z = m.addVar(lb=-1, ub=1, name=f'z{i}')
        m.addCons(z == sin(wa))
        y = m.addVar(lb=0, ub=1, name=f'y{i}')
        m.addCons(y == abs(z))
        vp.append(sysm.e[i] * y)
    Cexpr = quicksum(sysm.a[i] * P[i] * P[i] + sysm.b[i] * P[i] + sysm.c[i] for i in range(n)) + quicksum(vp)
    Eexpr = quicksum(sysm.k * (sysm.al[i] + sysm.be[i] * P[i] + sysm.ga[i] * P[i] * P[i]) for i in range(n)) + \
        quicksum(sysm.ze[i] * exp(sysm.la[i] * P[i]) for i in range(n))
    # normalized objective residuals as variables
    xv = m.addVar(lb=None, name='x'); yv = m.addVar(lb=None, name='y')
    m.addCons(xv == (Cexpr - Ft[0]) / sc[0]); m.addCons(yv == (Eexpr - Ft[1]) / sc[1])
    t = m.addVar(lb=0, name='t')
    m.addCons(t >= xv * xv + yv * yv)
    m.setObjective(t, 'minimize')
    t0 = time.perf_counter(); m.optimize(); el = time.perf_counter() - t0
    lb = max(m.getDualbound(), 0.0)
    ub = m.getPrimalbound() if m.getNSols() > 0 else np.inf
    Pbest = np.array([m.getVal(x) for x in P]) if m.getNSols() > 0 else None
    # the exact primal value of the SCIP point, re-evaluated with the model code
    ub_exact = float(np.linalg.norm((sysm.F(Pbest) - Ft) / sc)) if Pbest is not None else np.inf
    out = dict(G_lb=min(1.0, np.sqrt(lb) / d0), G_ub=min(1.0, ub_exact / d0), status=m.getStatus(),
               time_s=el, nodes=m.getNNodes())
    if return_P:
        out['P'] = Pbest
    return out


def sample_cells(seed=20261028):
    """Stratified sample: one request per hour (weight and cut rotate), three cells per request
    (physical set, reference allowance, one other grid cell drawn at random)."""
    Rq, vk = load_requests()
    G = pd.concat([pd.read_csv(RAW_ATTAIN / f'grid_e10{t}.csv') for t in 'ab'])
    G = G[(G.kind == 'grid') & G.duplicate_of_w.isna()]
    rng = np.random.default_rng(seed)
    combos = [(w, c) for w in (0.25, 0.5, 0.75) for c in (0.025, 0.05, 0.10)]
    cells = []
    for h in range(1, 25):
        sub = Rq[Rq.hour == h]
        for j in range(len(combos)):
            w, c = combos[(h + j) % len(combos)]
            m = sub[(sub.w == w) & (sub.cut == c)]
            if len(m): break
        row = m.iloc[0]
        g = G[(G.hour == h) & (G.w == row.w) & (G.cut == row.cut) & ~((G.R == 50) & (G.rho == 0.05))]
        gr = g.iloc[rng.integers(len(g))]
        cells += [(row.key, BIG, 1.0, row.G_phys, 'physical'), (row.key, 50.0, 0.05, row.G_ref, 'reference'),
                  (row.key, float(gr.R), float(gr.rho), gr.G, 'grid')]
    return cells, vk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--time-limit', type=float, default=600)
    a = ap.parse_args()
    out = RAW_ADD / 'scip'; out.mkdir(parents=True, exist_ok=True)
    s = System(str(COEFFS['10']))
    cells, vk = sample_cells()
    rows = []
    for j, (key, R, rho, Gst, kind) in enumerate(cells):
        if j % a.parts != a.part:
            continue
        res = scip_gap(s, vk[key], R, rho, time_limit=a.time_limit)
        rows.append(dict(hour=key[0], w=key[1], cut=key[2], R=R if R < BIG else np.inf, rho=rho, kind=kind,
                         G_stored=Gst, **res))
        print(j, key, kind, R, rho, round(Gst, 4), round(res['G_lb'], 4), round(res['G_ub'], 4), res['status'],
              round(res['time_s'], 1), flush=True)
        pd.DataFrame(rows).to_csv(out / f'scip_p{a.part}.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
