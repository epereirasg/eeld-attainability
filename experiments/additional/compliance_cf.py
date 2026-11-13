"""Compliance counterfactuals on the bounded emission reach eta_E(R, rho) of the 10-unit system.

The counterfactuals rho*, R*, tau* and S* of the main campaign close the normalized distance d to the cheapest
compliant dispatch. The compliance counterfactuals ask the cap question alone. For a tolerance eps_E,
    rho*_E = min{rho in [rho_0, 1] | eta_E(inf, rho) >= (1 - eps_E) eta_E}
    R*_E   = min{R >= 0           | eta_E(R, 1)     >= (1 - eps_E) eta_E}
    tau*_E = min{tau >= tau_0      | eta_E(inf, b(tau)) >= (1 - eps_E) eta_E}
    S*_E   = a smallest set S with eta_E(inf, rho_S) >= (1 - eps_E) eta_E (all sets of at most three of the ten units)
where eta_E is the physical reach, measured against the cap E^tar = (1 - c) E(P_inc), and b(tau) the ramp-derived band of the lossless model. For a reachable
request eta_E = 1, and eps_E = 0 asks that the bounded set contain a dispatch that meets the cap. A request needs a
compliance counterfactual when eta_E(R_0, rho_0) < (1 - eps_E) eta_E. The tolerance is relative to what the
system allows, so for an unreachable request it asks for all but eps_E of the physically attainable reduction.

eta_E(R, rho) is a convex program (convex emission over a polytope). We solve it with SLSQP on a lifted smooth
form, P and u with u_i >= |P_i - P_inc,i| and sum(u) <= R, from the incumbent and the projected minimum-emission
dispatch. Bisection is then exact up to its step. For the main tolerance we certify each value. The dispatch found
at the value is a feasible point with eta_E above the threshold, so the value suffices, and a SCIP lower bound on
min E at one step below shows that the smaller allowance does not.

With --calibrated the band and budget values use, for each request and gap tolerance eps, the compliance tolerance
eps_E = eps d(P_inc) / eta_E. Without the floor of the scale, every dispatch with G <= G*_phys + eps then meets the
compliance condition (Proposition 1 (inclusion) of the manuscript), so rho*_E <= rho* and R*_E <= R* up to one bisection step.
For a reachable request d(P_inc) = sqrt(2) and eta_E = 1, which gives eps_E = sqrt(2) eps. Output with --calibrated:
compliance_calibrated_p<part>.csv, with the gap tolerance in eps_gap.
It also records, for the counterfactuals of the main campaign, the share of the requested cut that the allowance
rho* or R* delivers, eta_E(inf, rho*) and eta_E(R*, 1).
Output: results/raw/additional/compliance/compliance_p<part>.csv (one row per distinct request and tolerance)
"""
import argparse
import itertools
import sys
import time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from eeldops.data_loader import load_eeld10

EPS_E = (0.01, 0.0, 0.05)        # main tolerance first
RHO0, R0, TAU0 = 0.05, 50.0, 15.0
STEPS, R_HI, TAU_HI = 8, 1000.0, 300.0
NUM_TOL = 1e-6                   # numerical slack of the threshold on eta (share of the requested cut)


def emin_lifted(s, Pinc, PD, lo, hi, R, x0s):
    """min E(P) s.t. sum P = PD, lo <= P <= hi, ||P - Pinc||_1 <= R, lifted to smooth constraints."""
    n = s.n
    best = None
    for x0 in x0s:
        P0 = np.clip(x0, lo, hi)
        if R >= BIG:
            r = minimize(s.E, P0, jac=s.gE, method='SLSQP', bounds=list(zip(lo, hi)),
                         constraints=[{'type': 'eq', 'fun': lambda P: P.sum() - PD, 'jac': lambda P: np.ones(n)}],
                         options={'maxiter': 1000, 'ftol': 1e-14})
            P = r.x
        else:
            def f(x): return s.E(x[:n])
            def g(x): return np.concatenate([s.gE(x[:n]), np.zeros(n)])
            I = np.eye(n)
            cons = [{'type': 'eq', 'fun': lambda x: x[:n].sum() - PD, 'jac': lambda x: np.concatenate([np.ones(n), np.zeros(n)])},
                    {'type': 'ineq', 'fun': lambda x: x[n:] - (x[:n] - Pinc), 'jac': lambda x: np.hstack([-I, I])},
                    {'type': 'ineq', 'fun': lambda x: x[n:] + (x[:n] - Pinc), 'jac': lambda x: np.hstack([I, I])},
                    {'type': 'ineq', 'fun': lambda x: R - x[n:].sum(), 'jac': lambda x: np.concatenate([np.zeros(n), -np.ones(n)])}]
            x0l = np.concatenate([P0, np.abs(P0 - Pinc)])
            r = minimize(f, x0l, jac=g, method='SLSQP', bounds=list(zip(lo, hi)) + [(0, sp) for sp in (hi - lo + 1e-9)],
                         constraints=cons, options={'maxiter': 1000, 'ftol': 1e-14})
            P = r.x[:n]
        ok = (abs(P.sum() - PD) < 1e-5 and np.all(P >= lo - 1e-7) and np.all(P <= hi + 1e-7)
              and (R >= BIG or np.abs(P - Pinc).sum() <= R + 1e-5))
        if ok and (best is None or s.E(P) < best[0]):
            best = (s.E(P), P.copy())
    if best is None:
        return s.E(Pinc), Pinc.copy()
    return best


def scip_emin_lb(s, Pinc, PD, lo, hi, R, time_limit=60):
    """SCIP lower bound on min E over the bounded set (convex, so SCIP closes the gap quickly)."""
    from pyscipopt import Model, quicksum, exp
    m = Model(); m.hideOutput(); m.setParam('limits/time', time_limit); m.setParam('limits/gap', 1e-9)
    n = s.n
    P = [m.addVar(lb=lo[i], ub=hi[i]) for i in range(n)]
    m.addCons(quicksum(P) == PD)
    if R < BIG:
        u = [m.addVar(lb=0, ub=hi[i] - lo[i]) for i in range(n)]
        for i in range(n):
            m.addCons(u[i] >= P[i] - Pinc[i]); m.addCons(u[i] >= Pinc[i] - P[i])
        m.addCons(quicksum(u) <= R)
    ev = m.addVar(lb=None)
    m.addCons(ev >= quicksum(s.k * (s.al[i] + s.be[i] * P[i] + s.ga[i] * P[i] * P[i]) + s.ze[i] * exp(s.la[i] * P[i]) for i in range(n)))
    m.setObjective(ev, 'minimize'); m.optimize()
    return m.getDualbound(), m.getStatus()


class Reach:
    def __init__(self, s, v, ramp_r):
        self.s = s; self.PD = float(v['PD']); self.Pinc = np.array(v['Pinc'])
        self.Einc = s.E(self.Pinc); self.rr = ramp_r
        # the cap E^tar = (1 - c) E(P_inc). For a reachable request with a slack cap the target emission F^tar_E lies below
        # the cap, and compliance is measured against the cap, not against the target emission.
        self.Et = (1.0 - float(v['cut'])) * self.Einc
        self.Et_target = float(v['Ft'][1])
        self.Pe = emin_lifted(s, self.Pinc, self.PD, s.lo, s.hi, BIG, [self.Pinc])[1]
        self.Emin = s.E(self.Pe)
        self.eta_phys = float(min(1.0, (self.Einc - self.Emin) / (self.Einc - self.Et)))

    def box(self, rho=None, tau=None):
        s = self.s
        b = np.asarray(rho, float) * s.span if tau is None else self.rr * tau / 60.0
        b = np.broadcast_to(b, self.Pinc.shape)
        return np.maximum(s.lo, self.Pinc - b), np.minimum(s.hi, self.Pinc + b)

    def emin(self, R, rho=None, tau=None):
        lo, hi = self.box(rho, tau)
        return emin_lifted(self.s, self.Pinc, self.PD, lo, hi, R, [self.Pinc, self.Pe])

    def eta(self, R, rho=None, tau=None):
        E, _ = self.emin(R, rho, tau)
        return float(min(1.0, max(0.0, (self.Einc - E) / (self.Einc - self.Et))))

    def E_thr(self, eps):
        """Largest emission that meets eta >= (1 - eps) eta_phys."""
        return self.Einc - ((1 - eps) * self.eta_phys - NUM_TOL) * (self.Einc - self.Et)


def bisect_up(ok, lo, hi, steps=STEPS):
    if not ok(hi):
        return np.nan
    for _ in range(steps):
        m = 0.5 * (lo + hi)
        if ok(m): hi = m
        else: lo = m
    return hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--no-scip', action='store_true')
    ap.add_argument('--case', default='10', choices=['10', '40'])
    ap.add_argument('--eps', default='', help='comma-separated eps_E values for a band and budget run without unit sets '
                    'and certificate, written to compliance_match_p<part>.csv (matched tolerances, eps_E = sqrt(2) eps)')
    ap.add_argument('--calibrated', default='', help='comma-separated gap tolerances eps; each request uses '
                    'eps_E = eps d(P_inc) / eta_E (compliance_calibrated_p<part>.csv)')
    a = ap.parse_args()
    eps_list = tuple(float(x) for x in a.eps.split(',')) if a.eps else EPS_E
    cal = tuple(float(x) for x in a.calibrated.split(',')) if a.calibrated else ()
    out = RAW_ADD / 'compliance'; out.mkdir(parents=True, exist_ok=True)
    from eeldattain.runio import campaign_escale
    s = System(str(COEFFS[a.case]), escale=campaign_escale(a.case)); Rq, vk = load_requests(a.case)
    global R0, R_HI
    if a.case == '10':
        _, _, ramp, _ = load_eeld10(); rr = np.minimum(ramp.UR_MW_per_h.values, ramp.DR_MW_per_h.values)
        kmax, tag = 3, ''
    else:                            # no ramp data for the 40-unit benchmark, sets of at most two units
        rr = None; kmax, tag = 2, '_e40'; R0 = float(Rq.R0.iloc[0]); R_HI = 3000.0
    if a.eps:
        tag += '_match'
    if cal:
        tag += '_calibrated'
    rows = []
    for i, row in Rq.iterrows():
        if i % a.parts != a.part:
            continue
        t0 = time.perf_counter()
        v = vk[row.key]; Z = Reach(s, v, rr)
        base = dict(hour=int(row.hour), w=float(row.w), cut=float(row.cut), reachable=bool(row.reachable),
                    eta_phys=Z.eta_phys, eta_phys_stored=float(row.eta_E), eta_ref=Z.eta(R0, RHO0),
                    eta_ref_tau=Z.eta(R0, tau=TAU0) if rr is not None else np.nan,
                    eta_at_rho_star=Z.eta(BIG, row.rho_star) if np.isfinite(row.rho_star) else np.nan,
                    eta_at_R_star=Z.eta(row.R_star, 1.0) if np.isfinite(row.R_star) else np.nan,
                    G_ref=float(row.G_ref), G_phys=float(row.G_phys), rho_star=float(row.rho_star), R_star=float(row.R_star))
        if cal:
            Fi = s.F(Z.Pinc); Ft = np.array(v['Ft']); sc = np.array(v['scale'])
            d0 = float(np.linalg.norm((Fi - Ft) / sc))
            todo = [(eg * d0 / Z.eta_phys, eg) for eg in cal]
        else:
            todo = [(e, np.nan) for e in eps_list]
        for j, (eps, eg) in enumerate(todo):
            thr = (1 - eps) * Z.eta_phys - NUM_TOL
            need = base['eta_ref'] < thr
            r = dict(base, eps_E=eps, eps_gap=eg, need_E=bool(need))
            if need:
                okr = lambda x: Z.eta(BIG, x) >= thr
                okR = lambda x: Z.eta(x, 1.0) >= thr
                okt = lambda x: Z.eta(BIG, tau=x) >= thr
                # the lower end of the interval suffices when the budget alone binds (as for rho* in the campaign)
                r['rho_star_E'] = RHO0 if okr(RHO0) else bisect_up(okr, RHO0, 1.0)
                r['R_star_E'] = bisect_up(okR, 0.0, R_HI)
                r['tau_star_E'] = (TAU0 if okt(TAU0) else bisect_up(okt, TAU0, TAU_HI)) if rr is not None else np.nan
                if j == 0 and not a.eps and not cal:   # unit sets and the certificate for the main tolerance
                    best_k, sets = np.nan, []
                    if Z.eta(BIG, RHO0) >= thr:
                        best_k = 0
                    else:
                        for k in range(1, kmax + 1):
                            for S in itertools.combinations(range(s.n), k):
                                rv = np.full(s.n, RHO0); rv[list(S)] = 1.0
                                if Z.eta(BIG, rv) >= thr:
                                    sets.append('-'.join(str(u + 1) for u in S))
                            if sets:
                                best_k = k; break
                        if not sets:
                            best_k = kmax + 1
                    r.update(S_size_E=best_k, S_sets_E=';'.join(sets), S_n_minimal_E=len(sets))
                    if not a.no_scip:
                        Ethr = Z.E_thr(eps)
                        for name, val, step, fn in (('rho', r['rho_star_E'], 0.95 / 2 ** STEPS, lambda x: (BIG, Z.box(x))),
                                                    ('R', r['R_star_E'], R_HI / 2 ** STEPS, lambda x: (x, Z.box(1.0)))):
                            if not np.isfinite(val):
                                continue
                            Rb, (lo, hi) = fn(val)
                            E_at, _ = emin_lifted(s, Z.Pinc, Z.PD, lo, hi, Rb, [Z.Pinc, Z.Pe])
                            r[f'cert_suff_{name}'] = bool(E_at <= Ethr)
                            below = val - step
                            if (name == 'rho' and below < RHO0 - 1e-12) or below < -1e-12:
                                r[f'cert_min_{name}'] = True       # the value is the lower end of the interval
                                continue
                            Rb, (lo, hi) = fn(below)
                            lb, st = scip_emin_lb(s, Z.Pinc, Z.PD, lo, hi, Rb)
                            r[f'cert_min_{name}'] = bool(lb > Ethr)
                            r[f'scip_status_{name}'] = st
            rows.append(r)
        rows[-1]['time_s'] = time.perf_counter() - t0
        k0 = -len(todo)
        print(i, row.key, round(rows[k0]['eta_ref'], 3), rows[k0].get('rho_star_E'), rows[k0].get('R_star_E'),
              rows[k0].get('tau_star_E'), rows[k0].get('S_size_E'), round(time.perf_counter() - t0, 1), flush=True)
        pd.DataFrame(rows).to_csv(out / f'compliance{tag}_p{a.part}.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
