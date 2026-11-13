"""One-sided distance d+ of Proposition 2 on the stored 10-unit requests.

d+(P) = ||max(0, F(P) - (C_tar, E_cap)) / s+||, s+ = (C_tar - C(P_inc), E(P_inc) - E_cap), so d+(P_inc) = 1, and
G+(R, rho) = min over A_h(R, rho) of d+. For each cell the script first solves the convex least-emission problem over
A_h(R, rho). When that dispatch costs at most C_tar, Proposition 2 gives G+ = 1 - eta_E(R, rho) exactly (flag
cond = True). Otherwise it minimizes d+ by multistart SLSQP (cond = False). Cells: the physical set, the reference
allowance and the study region. Band and budget counterfactuals of d+ by bisection (eight steps, as in the main
campaign) for eps in {0.05, 0.01}. Output: results/raw/additional/onesided/{cells,cf}_p<part>.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *


def gplus(req, Ct, Ecap, sp, R, rho):
    s = req.s
    Eb, PE = emin_bounded(req, R, rho)
    lb = max(0.0, (Eb - Ecap) / sp[1])
    if s.C(PE) <= Ct * (1 + 1e-7):
        return lb, True, lb
    lo, hi = req.band(rho); n = len(lo)
    f = lambda x: x[n] ** 2 + x[n + 1] ** 2
    gf = lambda x: np.r_[np.zeros(n), 2 * x[n], 2 * x[n + 1]]
    cons = [{'type': 'eq', 'fun': lambda x: x[:n].sum() - req.PD, 'jac': lambda x: np.r_[np.ones(n), 0, 0]},
            {'type': 'ineq', 'fun': lambda x: x[n] - (s.C(x[:n]) - Ct) / sp[0], 'jac': lambda x: np.r_[-s.gC(x[:n]) / sp[0], 1, 0]},
            {'type': 'ineq', 'fun': lambda x: x[n + 1] - (s.E(x[:n]) - Ecap) / sp[1], 'jac': lambda x: np.r_[-s.gE(x[:n]) / sp[1], 0, 1]}]
    if R < BIG:
        cons.append({'type': 'ineq', 'fun': lambda x: R - np.abs(x[:n] - req.Pinc).sum(),
                     'jac': lambda x: np.r_[-np.sign(x[:n] - req.Pinc), 0, 0]})
    dplus = lambda P: float(np.hypot(max(0.0, (s.C(P) - Ct) / sp[0]), max(0.0, (s.E(P) - Ecap) / sp[1])))
    best = min(1.0, dplus(PE))
    starts = [PE, req.Pinc.copy()]
    for p in req.pool:
        q = balance(p, req.PD, lo, hi)
        if q is not None:
            starts.append(req._shrink(q, R))
    rng = np.random.default_rng(0)
    for _ in range(4):
        q = balance(rng.uniform(lo, hi), req.PD, lo, hi)
        if q is not None:
            starts.append(req._shrink(q, R))
    for P0 in starts:
        x0 = np.r_[np.clip(P0, lo, hi), max(0.0, (s.C(P0) - Ct) / sp[0]), max(0.0, (s.E(P0) - Ecap) / sp[1])]
        bnds = list(zip(lo, hi)) + [(0, None), (0, None)]
        r = minimize(f, x0, jac=gf, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-12})
        P = r.x[:n]
        if abs(P.sum() - req.PD) < 1e-5 and np.all(P >= lo - 1e-7) and np.all(P <= hi + 1e-7) and \
                (R >= BIG or np.abs(P - req.Pinc).sum() <= R + 1e-6):
            best = min(best, dplus(P))
    return max(best, lb), False, lb


def bisect(test, lo, hi, steps=8):
    if test(lo):
        return lo
    if not test(hi):
        return np.nan
    for _ in range(steps):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if test(m) else (m, hi)
    return hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    a = ap.parse_args()
    s = System(str(COEFFS['10']))
    Rq, vk = load_requests('10')
    out = RAW_ADD / 'onesided'; out.mkdir(parents=True, exist_ok=True)
    fc, ff = out / f'cells_p{a.part}.csv', out / f'cf_p{a.part}.csv'
    C = pd.read_csv(fc).to_dict('records') if fc.exists() else []
    F = pd.read_csv(ff).to_dict('records') if ff.exists() else []
    done = {(r['hour'], r['w'], r['cut']) for r in F}
    for i, row in Rq.iterrows():
        if i % a.parts != a.part or (row.hour, row.w, row.cut) in done:
            continue
        t0 = time.perf_counter()
        req = make_request(s, vk[row.key]); Ct = float(row.C_tar); Ecap = (1 - row.cut) * req.Finc[1]
        sp = np.array([Ct - req.Finc[0], req.Finc[1] - Ecap])
        cells = [('physical', BIG, 1.0), ('reference', 50.0, 0.05)] + [('grid', R, r) for R in (25.0, 50.0, 100.0, 250.0) for r in (0.05, 0.1, 0.2)]
        val = {}
        for kind, R, rho in cells:
            g, cond, lb = gplus(req, Ct, Ecap, sp, R, rho)
            val[(R, rho)] = g
            C.append(dict(hour=row.hour, w=row.w, cut=row.cut, reachable=row.reachable, kind=kind, R=R, rho=rho, Gp=g, cond=cond, lb=lb))
        gph = val[(BIG, 1.0)]; rec = dict(hour=row.hour, w=row.w, cut=row.cut, reachable=row.reachable, Gp_phys=gph, Gp_ref=val[(50.0, 0.05)])
        for eps in (0.05, 0.01):
            tag = f'{eps:g}'
            need = val[(50.0, 0.05)] > gph + eps
            rec[f'need_{tag}'] = need
            ncond = [0, 0]
            def test_rho(r):
                g, c, _ = gplus(req, Ct, Ecap, sp, BIG, r); ncond[0] += 1; ncond[1] += int(c); return g <= gph + eps
            def test_R(R):
                g, c, _ = gplus(req, Ct, Ecap, sp, R, 1.0); ncond[0] += 1; ncond[1] += int(c); return g <= gph + eps
            rec[f'rho_{tag}'] = bisect(test_rho, 0.05, 1.0) if need else np.nan
            rec[f'R_{tag}'] = bisect(test_R, 0.0, 1000.0) if need else np.nan
            rec[f'evals_{tag}'] = ncond[0]; rec[f'cond_{tag}'] = ncond[1]
        rec['time_s'] = time.perf_counter() - t0
        F.append(rec)
        pd.DataFrame(C).to_csv(fc, index=False); pd.DataFrame(F).to_csv(ff, index=False)
        print(row.hour, row.w, row.cut, round(gph, 3), round(rec['Gp_ref'], 3), rec['rho_0.05'], rec['R_0.05'], round(rec['time_s'], 1), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
