"""Band and budget counterfactuals by direct optimization instead of bisection.

For every distinct 10-unit request with stored rho* and R* (Section IV-D), the minimal allowance is computed directly:
  R*_direct   = min ||P - Pinc||_1          s.t. P in Omega_h, d(P) <= d(Pinc) (G*_phys + eps)
  rho*_direct = min rho                     s.t. P in Omega_h, |P_i - Pinc_i| <= rho span_i, d(P) <= d(Pinc) (G*_phys + eps)
with the stored G*_phys, by multistart SLSQP (incumbent, target point, stored reference optimum, four random points).
The distance constraint is nonconvex, so the direct problem has the same multimodality as the gap itself and no
certificate. Output: results/raw/additional/direct_cf.csv with values and wall-clock times.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from common import *

OUT = RAW_ADD / 'direct_cf.csv'


def starts(s, v, rng, k=4):
    PD = float(v['PD']); x = [np.array(v['Pinc']), np.array(v['Pfr']), np.array(v['P_ref'])]
    x += [s.rand_feasible(PD, rng) for _ in range(k)]
    return x


def direct(s, v, thr, kind, rng):
    Pinc = np.array(v['Pinc']); PD = float(v['PD']); n = s.n
    req = make_request(s, v); lim = thr * req.e0
    best = None
    for x0 in starts(s, v, rng):
        if kind == 'R':          # variables P and u >= |P - Pinc|
            z0 = np.concatenate([x0, np.abs(x0 - Pinc)])
            f = lambda z: z[n:].sum(); g = lambda z: np.concatenate([np.zeros(n), np.ones(n)])
            cons = [{'type': 'eq', 'fun': lambda z: z[:n].sum() - PD},
                    {'type': 'ineq', 'fun': lambda z: z[n:] - (z[:n] - Pinc)},
                    {'type': 'ineq', 'fun': lambda z: z[n:] + (z[:n] - Pinc)},
                    {'type': 'ineq', 'fun': lambda z: lim - req.d(z[:n]), 'jac': lambda z: np.concatenate([-req.grad(z[:n]), np.zeros(n)])}]
            bnds = list(zip(s.lo, s.hi)) + [(0, None)] * n
        else:                    # variables P and rho
            z0 = np.concatenate([x0, [np.max(np.abs(x0 - Pinc) / s.span)]])
            f = lambda z: z[n]; g = lambda z: np.concatenate([np.zeros(n), [1.0]])
            cons = [{'type': 'eq', 'fun': lambda z: z[:n].sum() - PD},
                    {'type': 'ineq', 'fun': lambda z: z[n] * s.span - (z[:n] - Pinc)},
                    {'type': 'ineq', 'fun': lambda z: z[n] * s.span + (z[:n] - Pinc)},
                    {'type': 'ineq', 'fun': lambda z: lim - req.d(z[:n]), 'jac': lambda z: np.concatenate([-req.grad(z[:n]), [0.0]])}]
            bnds = list(zip(s.lo, s.hi)) + [(0, 1)]
        r = minimize(f, z0, jac=g, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10})
        z = r.x; P = z[:n]
        ok = abs(P.sum() - PD) < 1e-5 and np.all(P >= s.lo - 1e-7) and np.all(P <= s.hi + 1e-7) and req.d(P) <= lim * (1 + 1e-6) + 1e-9
        if ok:
            val = float(np.abs(P - Pinc).sum()) if kind == 'R' else float(np.max(np.abs(P - Pinc) / s.span))
            if best is None or val < best: best = val
    return best


def main():
    s = System(str(COEFFS['10'])); Rq, vk = load_requests()
    rows = pd.read_csv(OUT).to_dict('records') if OUT.exists() else []
    done = {(r['hour'], r['w'], r['cut']) for r in rows}
    rng = np.random.default_rng(20261102)
    for r in Rq.itertuples():
        if not (np.isfinite(r.rho_star) and np.isfinite(r.R_star)) or (r.hour, r.w, r.cut) in done:
            continue
        thr = r.G_phys + EPS; v = vk[r.key]
        t0 = time.perf_counter(); Rd = direct(s, v, thr, 'R', rng); tR = time.perf_counter() - t0
        t0 = time.perf_counter(); rd = direct(s, v, thr, 'rho', rng); tr = time.perf_counter() - t0
        rows.append(dict(hour=r.hour, w=r.w, cut=r.cut, R_star=r.R_star, R_direct=Rd, t_R=tR, rho_star=r.rho_star, rho_direct=rd, t_rho=tr))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(r.key, r.R_star, Rd, round(tR, 1), r.rho_star, rd, round(tr, 1), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
