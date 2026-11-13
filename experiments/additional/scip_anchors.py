"""Global certification of the 10-unit anchors, incumbents and targets with SCIP.

For every hour it solves min C and min E over Omega_h (the anchors that set the front width W_h), and for every
distinct request
    min C(P)  s.t.  P in Omega_h, E(P) <= E(P_inc)    (the incumbent is efficient if no cheaper dispatch exists)
    min C(P)  s.t.  P in Omega_h, E(P) <= E_tar        (the target dispatch solves Eq. (5))
with the valve-point formulation of scip_certificate.py (explicit bounded sine argument). It reports SCIP's
global lower bound, its best dispatch and the stored value, so that the relative excess of the stored dispatch
over the certified lower bound can be read directly.
Output: results/raw/additional/scip/anchors.csv, incumbents_targets_p<part>.csv
"""
import argparse
import glob
import json
import os
import time
import numpy as np, pandas as pd
from pyscipopt import Model, quicksum, sin, exp
from eeldattain.model import System
from eeldattain.paths import COEFFS, RAW_ATTAIN, RAW


def build(s, PD, objective, Ecap=None, time_limit=120, relgap=1e-6, Ccap=None):
    m = Model(); m.hideOutput()
    m.setParam('limits/time', time_limit); m.setParam('limits/gap', relgap)
    n = s.n
    P = [m.addVar(lb=s.lo[i], ub=s.hi[i], name=f'P{i}') for i in range(n)]
    m.addCons(quicksum(P) == PD)
    vp = []
    for i in range(n):
        if s.e[i] == 0:
            continue
        wa = m.addVar(lb=s.f[i] * (s.lo[i] - s.hi[i]), ub=0.0, name=f'w{i}')
        m.addCons(wa == s.f[i] * (s.lo[i] - P[i]))
        z = m.addVar(lb=-1, ub=1, name=f'z{i}'); m.addCons(z == sin(wa))
        y = m.addVar(lb=0, ub=1, name=f'y{i}'); m.addCons(y == abs(z))
        vp.append(s.e[i] * y)
    Cexpr = quicksum(s.a[i] * P[i] * P[i] + s.b[i] * P[i] + s.c[i] for i in range(n)) + quicksum(vp)
    Eexpr = quicksum(s.k * (s.al[i] + s.be[i] * P[i] + s.ga[i] * P[i] * P[i]) for i in range(n)) + \
        quicksum(s.ze[i] * exp(s.la[i] * P[i]) for i in range(n))
    cv = m.addVar(lb=None, name='C'); ev = m.addVar(lb=None, name='E')
    m.addCons(cv == Cexpr); m.addCons(ev == Eexpr)
    if Ecap is not None:
        m.addCons(ev <= Ecap)
    if Ccap is not None:
        m.addCons(cv <= Ccap)
    m.setObjective(cv if objective == 'C' else ev, 'minimize')
    t0 = time.perf_counter(); m.optimize(); el = time.perf_counter() - t0
    Pb = np.array([m.getVal(x) for x in P]) if m.getNSols() > 0 else None
    return dict(lb=m.getDualbound(), ub=m.getPrimalbound() if Pb is not None else np.inf, status=m.getStatus(), time_s=el,
                C_best=s.C(Pb) if Pb is not None else np.nan, E_best=s.E(Pb) if Pb is not None else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--out', default=str(RAW / 'additional' / 'scip'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = System(str(COEFFS['10']))
    req = pd.concat([pd.read_csv(f, dtype={'S_units': str}) for f in sorted(glob.glob(f'{RAW_ATTAIN}/requests_e10*.csv'))])
    req = req[req.duplicate_of_w.isna()].sort_values(['hour', 'w', 'cut']).reset_index(drop=True)
    V = {(v['hour'], v['w'], v['cut']): v for f in sorted(glob.glob(f'{RAW_ATTAIN}/vectors_e10*.jsonl')) for v in map(json.loads, open(f))}
    rows = []
    if a.part == 0:                                   # anchors, every hour
        for h, X in req.groupby('hour'):
            PD = float(X.PD.iloc[0])
            rc = build(s, PD, 'C'); re_ = build(s, PD, 'E')
            rows.append(dict(hour=int(h), PD=PD, Cmin_lb=rc['lb'], Cmin_scip=rc['C_best'], E_at_Cmin_scip=rc['E_best'],
                             Cmin_stored=float(X.C_min.iloc[0]), st_C=rc['status'],
                             Emin_lb=re_['lb'], Emin_scip=re_['E_best'], Emin_stored=float(X.E_min.iloc[0]), st_E=re_['status'],
                             t=rc['time_s'] + re_['time_s']))
            pd.DataFrame(rows).to_csv(f'{a.out}/anchors.csv', index=False)
            print('anchors', h, round(rc['time_s'] + re_['time_s'], 1), flush=True)
    rows, seen = [], {}
    for i, r in req.iterrows():
        if i % a.nparts != a.part:
            continue
        v = V[(r.hour, r.w, r.cut)]; PD = float(v['PD'])
        Pinc, Pfr = np.array(v['Pinc']), np.array(v['Pfr'])
        k = (int(r.hour), float(r.w))
        if k not in seen:
            seen[k] = build(s, PD, 'C', Ecap=s.E(Pinc) + 1e-9)
        ri = seen[k]
        row = dict(hour=int(r.hour), w=float(r.w), cut=float(r.cut), reachable=bool(r.reachable), C_inc=s.C(Pinc),
                   inc_lb=ri['lb'], inc_best=ri['C_best'], st_inc=ri['status'])
        if r.reachable:
            rt = build(s, PD, 'C', Ecap=float(r.E_tar))
            row.update(C_tar=s.C(Pfr), tar_lb=rt['lb'], tar_best=rt['C_best'], tar_E_best=rt['E_best'], st_tar=rt['status'])
        rows.append(row)
        pd.DataFrame(rows).to_csv(f'{a.out}/incumbents_targets_p{a.part}.csv', index=False)
        print('request', i, int(r.hour), r.w, r.cut, flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
