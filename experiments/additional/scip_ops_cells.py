"""SCIP on the stored cells of the loss model (operational extension, 10-unit system).

The loss model replaces the linear balance by sum(P) - P^T B P = PD, a nonconvex quadratic equality, and keeps
the valve-point cost. For every distinct request the script solves the physical cell and the two reference
cells, G*(50 MW, b(15 min)) with ramp bands and G*(50 MW, 0.05) with uniform bands, and compares the global
lower bound with the stored value. With --grid ramp it solves every cell of the ramp-band grid instead
(budgets 25 to 250 MW and unrestricted, response times 10 to 60 min), with a 20 s CPU limit and a gap
tolerance that resolves G* to about 0.005, enough to decide the certificate at 0.01.
Output: results/raw/additional/scip/ops_cells.csv, or scip/ops_ramp_grid.csv with --grid ramp
"""
import argparse, os, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from pyscipopt import Model, quicksum, sin, exp
sys.path.insert(0, str(Path(__file__).parent))
from common import RAW_ADD
from eeldattain.paths import RAW_OPS
from eeldops.experiments import OPS, system


def scip_ops(s, v, R, band_mw, time_limit, absgap=1e-4):
    Pinc = np.array(v['Pinc']); PD = float(v['PD']); Ft = np.array(v['Ft']); sc = np.array(v['scale'])
    n = s.n
    lo = np.maximum(s.lo, Pinc - band_mw); hi = np.minimum(s.hi, Pinc + band_mw)
    m = Model(); m.hideOutput(); m.setParam('limits/time', time_limit); m.setParam('timing/clocktype', 1)
    d0 = float(np.linalg.norm((s.F(Pinc) - Ft) / sc)); m.setParam('limits/absgap', absgap * d0 ** 2)
    P = [m.addVar(lb=lo[i], ub=hi[i], name=f'P{i}') for i in range(n)]
    m.addCons(quicksum(P) - quicksum(float(s.B[i, j]) * P[i] * P[j] for i in range(n) for j in range(n)) == PD)
    if R < 1e8:
        u = [m.addVar(lb=0, ub=hi[i] - lo[i]) for i in range(n)]
        for i in range(n):
            m.addCons(u[i] >= P[i] - Pinc[i]); m.addCons(u[i] >= Pinc[i] - P[i])
        m.addCons(quicksum(u) <= R)
    vp = []
    for i in range(n):
        if s.e[i] == 0:
            continue
        wa = m.addVar(lb=s.f[i] * (s.lo[i] - hi[i]), ub=s.f[i] * (s.lo[i] - lo[i])); m.addCons(wa == s.f[i] * (s.lo[i] - P[i]))
        z = m.addVar(lb=-1, ub=1); m.addCons(z == sin(wa))           # bounded argument, as in scip_certificate.py
        y = m.addVar(lb=0, ub=1); m.addCons(y == abs(z)); vp.append(s.e[i] * y)
    Cx = quicksum(s.a[i] * P[i] * P[i] + s.b[i] * P[i] + s.c[i] for i in range(n)) + quicksum(vp)
    Ex = quicksum(s.al[i] + s.be[i] * P[i] + s.ga[i] * P[i] * P[i] for i in range(n)) + quicksum(s.ze[i] * exp(s.la[i] * P[i]) for i in range(n))
    xv = m.addVar(lb=None); yv = m.addVar(lb=None)
    m.addCons(xv == (Cx - Ft[0]) / sc[0]); m.addCons(yv == (Ex - Ft[1]) / sc[1])
    t = m.addVar(lb=0); m.addCons(t >= xv * xv + yv * yv)
    m.setObjective(t, 'minimize')
    t0 = time.perf_counter(); m.optimize(); el = time.perf_counter() - t0
    lb = max(m.getDualbound(), 0.0)
    Pb = np.array([m.getVal(x) for x in P]) if m.getNSols() > 0 else None
    ub = float(np.linalg.norm((s.F(Pb) - Ft) / sc)) if Pb is not None else np.inf
    res_bal = abs(s.residual(Pb, PD)) if Pb is not None else np.nan
    return dict(G_lb=min(1.0, np.sqrt(lb) / d0), G_ub=min(1.0, ub / d0), status=m.getStatus(), time_s=el,
                nodes=m.getNNodes(), balance_residual=res_bal)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--time-limit', type=float, default=120)
    ap.add_argument('--grid', default='', choices=['', 'ramp'])
    a = ap.parse_args()
    if a.grid == 'ramp':
        return ramp_grid(a.time_limit)
    s, _, _ = system()
    R = pd.read_csv(RAW_OPS / 'ops_requests_a.csv', dtype={'S_units': str})
    if os.environ.get('EELD_HOURS'):          # rerun of some hours, merged afterwards
        R = R[R.hour.isin([int(x) for x in os.environ['EELD_HOURS'].split(',')])]; R = R[R.duplicate_of_w.isna()]
    V = {(int(v['hour']), float(v['w']), float(v['cut'])): v for v in map(json.loads, open(RAW_OPS / 'ops_vectors_a.jsonl'))}
    out = RAW_ADD / 'scip' / 'ops_cells.csv'
    rows = pd.read_csv(out).to_dict('records') if out.exists() else []
    done = {(r['hour'], r['w'], r['cut'], r['kind']) for r in rows}
    for r in R.itertuples():
        v = V[(int(r.hour), float(r.w), float(r.cut))]
        for kind, Rb, band, Gst in (('physical', 1e9, s.span, r.G_phys),
                                    ('reference_ramp', OPS['R0'], s.band_ramp(OPS['tau0']), r.G_ref_ramp),
                                    ('reference_uniform', OPS['R0'], s.band_uniform(OPS['rho0']), r.G_ref_uniform)):
            if (r.hour, r.w, r.cut, kind) in done:
                continue
            res = scip_ops(s, v, Rb, band, a.time_limit)
            rows.append(dict(hour=r.hour, w=r.w, cut=r.cut, kind=kind, G_stored=Gst, **res))
            print(r.hour, r.w, r.cut, kind, round(Gst, 4), round(res['G_lb'], 4), round(res['G_ub'], 4), res['status'], round(res['time_s']), flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    print('DONE', flush=True)


def ramp_grid(time_limit):
    s, _, _ = system()
    R = pd.read_csv(RAW_OPS / 'ops_requests_a.csv', dtype={'S_units': str}); R = R[R.duplicate_of_w.isna()]
    G = pd.read_csv(RAW_OPS / 'ops_grid_a.csv'); G = G[G.band == 'ramp'].merge(R[['hour', 'w', 'cut']], on=['hour', 'w', 'cut'])
    V = {(int(v['hour']), float(v['w']), float(v['cut'])): v for v in map(json.loads, open(RAW_OPS / 'ops_vectors_a.jsonl'))}
    out = RAW_ADD / 'scip' / 'ops_ramp_grid.csv'
    rows = pd.read_csv(out).to_dict('records') if out.exists() else []
    done = {(r['hour'], r['w'], r['cut'], r['R'], r['tau']) for r in rows}
    for k, g in enumerate(G.itertuples()):
        if (g.hour, g.w, g.cut, g.R, g.level) in done:
            continue
        v = V[(int(g.hour), float(g.w), float(g.cut))]
        res = scip_ops(s, v, float(g.R), s.band_ramp(float(g.level)), time_limit, absgap=1e-3)   # resolves G* to about 0.005
        rows.append(dict(hour=g.hour, w=g.w, cut=g.cut, R=g.R, tau=g.level, G_stored=g.G, **res))
        if k % 20 == 0:
            print(k, len(G), g.hour, g.w, g.cut, g.R, g.level, round(g.G, 4), round(res['G_lb'], 4), res['status'], flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
