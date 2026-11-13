"""Attainability campaign under alternative modelling choices (10-unit system).

Variants (option --variant):
  convex      valve-point term removed (e_i = 0), efficient incumbents for w in {0.25, 0.5, 0.75}
  leastcost   valve-point model, incumbent = least-cost dispatch P^C (the outcome of economic dispatch)
  perturbed   valve-point model, incumbent = efficient w = 0.5 dispatch moved by a balanced random
              perturbation of up to 5% of each unit's range (a dominated dispatch in service)
  reserve     valve-point model with a 10-min spinning-reserve requirement of 3% of demand,
              sum_i min(Pmax_i - P_i, r_up_i / 6) >= 0.03 D_h, imposed on the incumbent, on Omega_h and on
              every bounded set (unit commitment fixed, all units online)

Each request runs the pipeline of experiments/attainability/run.py (gap grid, reference gaps, rho*, R*)
with an exhaustive unit-set search over all units up to size 2 (class 3 means three or more), and records wall-clock times per quantity.
"""
import argparse
import itertools
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import eeldattain.model as M
from eeldattain.model import System, Request, anchors, efficient_incumbent, eps_point, balance
from eeldattain.paths import COEFFS, DEMAND
from common import RAW_ADD, BIG, Timer, emin_bounded, eta_bounded
from eeldops.data_loader import load_eeld10

R_GRID = (10, 25, 50, 100, 250); RHO_GRID = (0.02, 0.05, 0.10, 0.20)
REF = (50.0, 0.05); EPS = 0.05


def bisect(fun, lo, hi, thr, steps=8):
    if fun(hi) > thr:
        return np.nan
    for _ in range(steps):
        m = 0.5 * (lo + hi)
        if fun(m) <= thr: hi = m
        else: lo = m
    return hi


def install_reserve(sysm, ramp_up, share):
    """Wrap eeldattain.model.solve so that every solve also meets the spinning-reserve constraint."""
    base_solve = M.solve
    cap = ramp_up / 6.0
    state = {'PD': None}
    def res_fun(P):
        return np.minimum(sysm.hi - P, cap).sum() - share * state['PD']
    def res_jac(P):
        return np.where(sysm.hi - P < cap, -1.0, 0.0)
    def solve(s, PD, fun, jac, x0s, lo=None, hi=None, cons=(), tol=1e-6):
        state['PD'] = PD
        return base_solve(s, PD, fun, jac, x0s, lo=lo, hi=hi,
                          cons=(*cons, {'type': 'ineq', 'fun': res_fun, 'jac': res_jac}), tol=tol)
    M.solve = solve
    sysm.reserve_cap, sysm.reserve_share, sysm.reserve_state = cap, share, state   # read by eeldattain.globalopt
    return res_fun, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['convex', 'leastcost', 'perturbed', 'reserve'])
    ap.add_argument('--hours', default=','.join(str(h) for h in range(1, 25)))
    ap.add_argument('--tag', default='')
    ap.add_argument('--weights', default='0.25,0.5,0.75', help='incumbent weights (convex and reserve variants)')
    ap.add_argument('--certify', action='store_true', help='SCIP check of anchors, incumbents and targets (globalopt)')
    ap.add_argument('--check-only', action='store_true', help='only report which SCIP checks change a dispatch')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else RAW_ADD / 'variants'; out.mkdir(parents=True, exist_ok=True)
    tag = a.variant + a.tag
    from eeldattain import globalopt
    CERT = []
    s = System(str(COEFFS['10']))
    if a.variant == 'convex':
        s.e = np.zeros_like(s.e)
    res = None
    if a.variant == 'reserve':
        _, B, ramp, _ = load_eeld10()
        res = install_reserve(s, ramp.UR_MW_per_h.to_numpy(float), 0.03)
    dem = pd.read_csv(DEMAND['10'])
    if a.variant in ('convex', 'reserve'):
        incs = [('w', float(w)) for w in a.weights.split(',')]
    elif a.variant == 'leastcost':
        incs = [('Pc', 1.0)]
    else:
        incs = [('pert', 0.5)]
    cuts = (0.025, 0.05, 0.10)
    rows, grid, done = [], [], set()
    fr, fg = out / f'requests_{tag}.csv', out / f'grid_{tag}.csv'
    if fr.exists() and fg.exists():             # resume: keep the hours that were written completely
        R0, G0 = pd.read_csv(fr), pd.read_csv(fg)
        cnt = R0.groupby('hour').size()
        done = {int(h) for h, n in cnt.items() if n >= len(incs) * len(cuts)}   # complete hours only
        rows = R0[R0.hour.isin(done)].to_dict('records'); grid = G0[G0.hour.isin(done)].to_dict('records')
    for h in [int(x) for x in a.hours.split(',')]:
        if h in done:
            continue
        PD = float(dem.loc[dem.hour == h, 'PD'].iloc[0])
        rng = np.random.default_rng(20261028 + 1000 * h + len(a.variant))
        if res is not None:
            res[1]['PD'] = PD
        T = Timer()
        A = anchors(s, PD, rng, K=100)
        if a.certify:
            A = globalopt.certify_anchors(s, PD, A, log=CERT)
        t_anchor = T.lap()
        Emin = A['ideal'][1]; W = (A['nadir'][1] - Emin) / A['nadir'][1]
        seen = {}
        for kind, w in incs:
            T.lap()
            if kind == 'w':
                Pinc = efficient_incumbent(s, PD, w, A)
                if a.certify:
                    Pinc = globalopt.improve(s, PD, 'W', Pinc, w=w, ideal=A['ideal'], nadir=A['nadir'], log=CERT)
            elif kind == 'Pc':
                Pinc = A['Pc'].copy()
            else:
                P0 = efficient_incumbent(s, PD, 0.5, A)
                if a.certify:
                    P0 = globalopt.improve(s, PD, 'W', P0, w=0.5, ideal=A['ideal'], nadir=A['nadir'], log=CERT)
                prng = np.random.default_rng(777 + h)
                Pinc = None
                for _ in range(200):
                    q = balance(P0 + prng.uniform(-0.05, 0.05, s.n) * s.span, PD, s.lo, s.hi)
                    if q is not None:
                        Pinc = q; break
            t_inc = T.lap()
            key = tuple(np.round(Pinc, 4)); dup = seen.get(key); seen.setdefault(key, w)
            Finc = s.F(Pinc)
            for cut in cuts:
                T.lap(); tt = {}
                Et = (1 - cut) * Finc[1]; reach = bool(Et >= Emin - 1e-9)
                if reach:
                    Pfr = eps_point(s, PD, Et, [Pinc, A['Pe']] + A['front'])
                    if a.certify:
                        Pfr = globalopt.improve(s, PD, 'C', Pfr, Ecap=Et, log=CERT)
                    if a.check_only:
                        continue
                    if Pfr is None:
                        Pfr = A['Pe']; reach = False
                if reach:
                    Ft = s.F(Pfr)
                else:
                    Pfr = A['Pe']; Ft = np.array([s.C(A['Pe']), Et])
                if a.check_only:
                    continue
                sc = np.maximum(np.abs(Ft - Finc), 1e-6 * np.abs(Finc))
                req = Request(s, PD, Pinc, Ft, sc, Pfr, A['Pc'], A['Pe'], pool_cap=8)
                tt['t_target'] = T.lap()
                Gp, _ = req.gap(BIG, 1.0, rng); tt['t_phys'] = T.lap()
                eta = min(1.0, max(0.0, (Finc[1] - Emin) / (Finc[1] - Et)))
                base = dict(variant=a.variant, hour=h, PD=PD, inc=kind, w=w, cut=cut, duplicate_of_w=dup,
                            reachable=reach, E_inc=Finc[1], C_inc=Finc[0], E_min=Emin, C_min=A['ideal'][0],
                            front_width=W, eta_E=eta, C_tar=Ft[0], E_tar=Et, G_phys=Gp,
                            mw_full=float(np.abs(Pfr - Pinc).sum()) if reach else np.nan)
                for R in R_GRID:
                    for rho in RHO_GRID:
                        T.lap(); G, P = req.gap(R, rho, rng); tg = T.lap()
                        Eb, _ = emin_bounded(req, R, rho) if a.variant != 'reserve' else (np.nan, None)
                        grid.append(dict(base, R=R, rho=rho, G=G, mw=float(np.abs(P - Pinc).sum()),
                                         E_opt=s.E(P), Emin_b=Eb,
                                         eta_b=eta_bounded(req, Eb, Et) if Eb == Eb else np.nan, t_gap=tg))
                T.lap()
                G0, P0r = req.gap(*REF, rng); GRf, PRf = req.gap(BIG, REF[1], rng); Grf, _ = req.gap(REF[0], 1.0, rng)
                tt['t_ref3'] = T.lap()
                r = dict(base, G_ref=G0, G_Rfree=GRf, G_rhofree=Grf, ref_mw=float(np.abs(P0r - Pinc).sum()),
                         ref_units_moved=int(np.sum(np.abs(P0r - Pinc) > 0.5)), t_anchor=t_anchor, t_inc=t_inc, **tt)
                thr = Gp + EPS
                if G0 > thr:
                    T.lap()
                    r['rho_star'] = bisect(lambda x: req.gap(BIG, x, rng)[0], REF[1], 1.0, thr); r['t_rho_star'] = T.lap()
                    r['R_star'] = bisect(lambda x: req.gap(x, 1.0, rng)[0], 0.0, 1000.0, thr); r['t_R_star'] = T.lap()
                    if GRf > thr:
                        found = None; nclose = 0
                        for k in (1, 2):
                            for S in itertools.combinations(range(s.n), k):
                                rv = np.full(s.n, REF[1]); rv[list(S)] = 1.0
                                if req.gap(BIG, rv, rng)[0] <= thr:
                                    nclose += 1
                                    if found is None: found = S
                            if found is not None:
                                break
                        r.update(S_size=len(found) if found else 3, S_units='-'.join(str(j + 1) for j in found) if found else '',
                                 S_n_minimal=nclose, t_S_star=T.lap())
                    else:
                        r.update(S_size=0)
                r['t_request'] = sum(v for k_, v in r.items() if k_.startswith('t_') and k_ not in ('t_anchor', 't_inc')) + \
                    sum(g['t_gap'] for g in grid[-len(R_GRID) * len(RHO_GRID):])
                rows.append(r)
                print(tag, h, kind, w, cut, round(r['t_request'], 1), flush=True)
        if a.certify:
            ad = [c for c in CERT if c['adopted']]
            print('certify', tag, h, len(CERT), 'adopted', [(c['kind'], c['w'], round(c['v_multistart'] - c['v_final'], 6)) for c in ad], flush=True)
            pd.DataFrame(CERT).to_csv(out / f'certify_{tag}.csv', index=False)
        if a.check_only:
            continue
        if True:
            pd.DataFrame(rows).to_csv(out / f'requests_{tag}.csv', index=False)
            pd.DataFrame(grid).to_csv(out / f'grid_{tag}.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
