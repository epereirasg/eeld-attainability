"""Dense allowance grid, bounded emission reach, joint counterfactual,
distance and budget-norm sensitivity, and tolerance sensitivity of rho* and R*.

For every distinct stored 10-unit request it re-solves G*(R, rho) on a dense (R, rho) grid and records,
per cell, the emission and cost of the bounded optimum and the minimum emission over the bounded set
(a convex problem). It also computes the gap under the Chebyshev distance and under an L2 budget, and
the band and budget counterfactuals for eps in {0.02, 0.05, 0.10}. Wall-clock times are recorded per call.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from common import *

R_GRID = (10, 25, 50, 100, 250, BIG)
RHO_GRID = (0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
REALISTIC = [(R, rho) for R in (25, 50, 100) for rho in (0.05, 0.10, 0.20)]
EPS_SET = (0.02, 0.10)        # eps = 0.05 is stored in the main campaign


def bisect(fun, lo, hi, thr, steps=8):
    if fun(hi) > thr:
        return np.nan
    for _ in range(steps):
        m = 0.5 * (lo + hi)
        if fun(m) <= thr: hi = m
        else: lo = m
    return hi


def gap_variant(req, R, rho, norm='inf', budget='l1', rng=None, k=4):
    """G* under the Chebyshev distance (norm='inf') or with an L2 budget (budget='l2', radius R/sqrt(2)).
    Epigraph form for the Chebyshev case: variables (P, t), minimize t s.t. |r_j(P)| <= t."""
    s = req.s; lo, hi = req.band(rho); n = s.n
    sc = req.sc; Ft = req.Ft
    def r(P): return (s.F(P) - Ft) / sc
    if norm == 'inf':
        d0 = float(np.max(np.abs(r(req.Pinc))))
    else:
        d0 = req.e0
    cons = [{'type': 'eq', 'fun': lambda x: x[:n].sum() - req.PD, 'jac': lambda x: np.r_[np.ones(n), 0.0]}]
    if R < BIG:
        if budget == 'l1':
            cons.append({'type': 'ineq', 'fun': lambda x: R - np.abs(x[:n] - req.Pinc).sum(),
                         'jac': lambda x: np.r_[-np.sign(x[:n] - req.Pinc), 0.0]})
        else:
            R2 = R / np.sqrt(2)
            cons.append({'type': 'ineq', 'fun': lambda x: R2 ** 2 - np.sum((x[:n] - req.Pinc) ** 2),
                         'jac': lambda x: np.r_[-2 * (x[:n] - req.Pinc), 0.0]})
    if norm == 'inf':
        gC = lambda P: s.gC(P) / sc[0]; gE = lambda P: s.gE(P) / sc[1]
        cons += [{'type': 'ineq', 'fun': lambda x: x[n] - r(x[:n])[0], 'jac': lambda x: np.r_[-gC(x[:n]), 1.0]},
                 {'type': 'ineq', 'fun': lambda x: x[n] + r(x[:n])[0], 'jac': lambda x: np.r_[gC(x[:n]), 1.0]},
                 {'type': 'ineq', 'fun': lambda x: x[n] - r(x[:n])[1], 'jac': lambda x: np.r_[-gE(x[:n]), 1.0]},
                 {'type': 'ineq', 'fun': lambda x: x[n] + r(x[:n])[1], 'jac': lambda x: np.r_[gE(x[:n]), 1.0]}]
        fun = lambda x: x[n]; jac = lambda x: np.r_[np.zeros(n), 1.0]
        val = lambda P: float(np.max(np.abs(r(P))))
    else:
        fun = lambda x: req.d(x[:n]); jac = lambda x: np.r_[req.grad(x[:n]), 0.0]
        val = req.d
    starts = [req.Pinc.copy()]
    for _ in range(4 * k):
        if len(starts) > k: break
        P = balance(rng.uniform(lo, hi), req.PD, lo, hi)
        if P is not None: starts.append(P)
    for e in req.pool:
        P = balance(e, req.PD, lo, hi)
        if P is not None: starts.append(P)
    best = d0
    for P0 in starts:
        if R < BIG:   # shrink into the budget
            if budget == 'l1':
                dd = np.abs(P0 - req.Pinc).sum(); lim = R
            else:
                dd = np.linalg.norm(P0 - req.Pinc); lim = R / np.sqrt(2)
            if dd > lim: P0 = req.Pinc + (lim / dd) * (P0 - req.Pinc)
        x0 = np.r_[np.clip(P0, lo, hi), val(P0)]
        res = minimize(fun, x0, jac=jac, method='SLSQP', bounds=list(zip(lo, hi)) + [(0, None)],
                       constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
        P = res.x[:n]
        ok = abs(P.sum() - req.PD) < 1e-5 and np.all(P >= lo - 1e-7) and np.all(P <= hi + 1e-7)
        if R < BIG:
            ok = ok and (np.abs(P - req.Pinc).sum() <= R + 1e-6 if budget == 'l1'
                         else np.linalg.norm(P - req.Pinc) <= R / np.sqrt(2) + 1e-6)
        if ok:
            best = min(best, val(P))
    return min(best, d0) / d0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    a = ap.parse_args()
    out = RAW_ADD / 'allowance_grid'; out.mkdir(parents=True, exist_ok=True)
    s = System(str(COEFFS['10']))
    Rq, vk = load_requests()
    rows_g, rows_r, done = [], [], set()
    fg, fr = out / f'grid_p{a.part}.csv', out / f'requests_p{a.part}.csv'
    if fg.exists() and fr.exists():          # resume after an interruption
        rows_r = pd.read_csv(fr).to_dict('records')
        done = {(int(r['hour']), float(r['w']), float(r['cut'])) for r in rows_r}
        rows_g = [g for g in pd.read_csv(fg).to_dict('records') if (int(g['hour']), float(g['w']), float(g['cut'])) in done]
    for i, row in Rq.iterrows():
        if i % a.parts != a.part or row.key in done:
            continue
        v = vk[row.key]
        rng = np.random.default_rng(20261028 + i)
        req = make_request(s, v, pool_cap=8)
        Et = float(v['Ft'][1]); T = Timer()
        Gp, Pp = req.gap(BIG, 1.0, rng); t_phys = T.lap()
        base = dict(hour=row.hour, w=row.w, cut=row.cut, reachable=row.reachable, G_phys_stored=row.G_phys,
                    G_phys=Gp, eta_E=row.eta_E, E_inc=req.Finc[1], C_inc=req.Finc[0], E_tar=Et, C_tar=v['Ft'][0])
        for R in R_GRID:
            for rho in RHO_GRID:
                T.lap()
                G, P = req.gap(R, rho, rng); tg = T.lap()
                Eb, _ = emin_bounded(req, R, rho); te = T.lap()
                rows_g.append(dict(base, R=R if R < BIG else np.inf, rho=rho, G=G, E_opt=s.E(P), C_opt=s.C(P),
                                   mw=float(np.abs(P - req.Pinc).sum()), Emin_b=Eb,
                                   eta_b=eta_bounded(req, Eb, Et),
                                   eta_opt=float((req.Finc[1] - s.E(P)) / (req.Finc[1] - Et)),
                                   t_gap=tg, t_emin=te))
        # distance and budget-norm variants at the reference and realistic cells
        rr = dict(base, t_phys=t_phys)
        T.lap()
        rr['Ginf_phys'] = gap_variant(req, BIG, 1.0, 'inf', rng=rng)
        rr['Ginf_ref'] = gap_variant(req, 50, 0.05, 'inf', rng=rng)
        for R, rho in REALISTIC:
            rr[f'Ginf_{R}_{rho}'] = gap_variant(req, R, rho, 'inf', rng=rng)
            rr[f'Gl2_{R}_{rho}'] = gap_variant(req, R, rho, '2', 'l2', rng=rng)
        rr['t_variants'] = T.lap()
        G0 = req.gap(50, 0.05, rng)[0]; GRf = req.gap(BIG, 0.05, rng)[0]
        rr.update(G_ref=G0, G_Rfree=GRf)
        for eps in EPS_SET:
            thr = Gp + eps
            need = G0 > thr
            rr[f'need_{eps}'] = need
            T.lap()
            if need:
                rr[f'rho_star_{eps}'] = bisect(lambda x: req.gap(BIG, x, rng)[0], 0.05, 1.0, thr)
                rr[f't_rho_{eps}'] = T.lap()
                rr[f'R_star_{eps}'] = bisect(lambda x: req.gap(x, 1.0, rng)[0], 0.0, 1000.0, thr)
                rr[f't_R_{eps}'] = T.lap()
        rows_r.append(rr)
        print(i, row.hour, row.w, row.cut, round(time.time()), flush=True)
        pd.DataFrame(rows_g).to_csv(out / f'grid_p{a.part}.csv', index=False)
        pd.DataFrame(rows_r).to_csv(out / f'requests_p{a.part}.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
