"""Operational dispatch model: valve-point cost, standard DEED emission, Kron losses, MW bands, ramp coupling.

Power balance is sum(P) = D + P_L(P), with P_L(P) = P^T B P (B in 1/MW). Bands are given in MW per unit,
either as rho * (Pmax - Pmin) (uniform surrogate) or as ramp_rate * tau / 60 (ramp-derived, tau in minutes).
"""
import numpy as np
from scipy.optimize import minimize

from eeldattain.model import SolveStats


class OpsSystem:
    def __init__(self, coef, B=None, ramp=None, escale=1.0):
        """escale multiplies the polynomial part of the emission (1e-2 in the printed 40-unit convention)."""
        self.n = len(coef)
        self.lo = coef.Pmin.to_numpy(float); self.hi = coef.Pmax.to_numpy(float); self.span = self.hi - self.lo
        self.a, self.b, self.c, self.e, self.f = (coef[k].to_numpy(float) for k in 'abcef')
        self.al, self.be, self.ga, self.ze, self.la = (coef[k].to_numpy(float) for k in ['alpha', 'beta', 'gamma', 'zeta', 'lambda'])
        self.al, self.be, self.ga = escale * self.al, escale * self.be, escale * self.ga
        self.B = np.zeros((self.n, self.n)) if B is None else np.asarray(B, float)
        self.rate = None if ramp is None else np.minimum(ramp.UR_MW_per_h.to_numpy(float), ramp.DR_MW_per_h.to_numpy(float))

    # objectives
    def C(self, P): return float(np.sum(self.a * P**2 + self.b * P + self.c + np.abs(self.e * np.sin(self.f * (self.lo - P)))))
    def E(self, P): return float(np.sum(self.al + self.be * P + self.ga * P**2 + self.ze * np.exp(self.la * P)))
    def F(self, P): return np.array([self.C(P), self.E(P)])
    def gC(self, P):
        z = self.f * (self.lo - P)
        return 2 * self.a * P + self.b - np.sign(self.e * np.sin(z)) * self.e * np.cos(z) * self.f
    def gE(self, P): return self.be + 2 * self.ga * P + self.ze * self.la * np.exp(self.la * P)
    def loss(self, P): return float(P @ self.B @ P)

    # balance
    def residual(self, P, PD): return float(P.sum() - PD - self.loss(P))
    def residual_jac(self, P): return 1.0 - 2.0 * (self.B @ P)

    def balance(self, P, PD, lo, hi):
        """Proportional headroom redistribution until sum(P) = PD + P_L(P) inside [lo, hi]."""
        P = np.clip(np.asarray(P, float), lo, hi)
        for _ in range(200):
            r = PD + self.loss(P) - P.sum()
            if abs(r) < 1e-8:
                return P
            room = (hi - P) if r > 0 else (P - lo)
            if room.sum() <= 1e-12:
                return None
            P = np.clip(P + r * room / room.sum(), lo, hi)
        return P if abs(self.residual(P, PD)) < 1e-6 else None

    def rand_feasible(self, PD, rng, lo=None, hi=None):
        lo = self.lo if lo is None else lo; hi = self.hi if hi is None else hi
        for _ in range(500):
            P = self.balance(rng.uniform(lo, hi), PD, lo, hi)
            if P is not None:
                return P
        raise RuntimeError('no feasible sample')

    def solve(self, PD, fun, jac, x0s, lo=None, hi=None, cons=(), tol=1e-6):
        lo = self.lo if lo is None else lo; hi = self.hi if hi is None else hi
        C = [{'type': 'eq', 'fun': lambda P: self.residual(P, PD), 'jac': self.residual_jac}, *cons]
        best = None
        for x0 in x0s:
            r = minimize(fun, np.clip(x0, lo, hi), jac=jac, method='SLSQP', bounds=list(zip(lo, hi)),
                         constraints=C, options={'maxiter': 2000, 'ftol': 1e-12})
            P = r.x
            ok = (abs(self.residual(P, PD)) < 1e-5 and np.all(P >= lo - 1e-7) and np.all(P <= hi + 1e-7)
                  and all(np.min(np.atleast_1d(c['fun'](P))) >= -tol for c in cons))
            if ok:
                v = fun(P)
                if best is None or v < best[0]:
                    best = (v, P.copy())
        SolveStats.calls += 1
        if best is None:
            SolveStats.no_feasible_endpoint += 1
        return best

    # anchors, incumbents, targets
    def anchors(self, PD, rng, K=100, sweep=9):
        x0 = [self.rand_feasible(PD, rng) for _ in range(K)]
        Pc = self.solve(PD, self.C, self.gC, x0)[1]
        Pe = self.solve(PD, self.E, self.gE, x0)[1]
        ideal = np.array([self.C(Pc), self.E(Pe)]); nadir = np.array([self.C(Pe), self.E(Pc)])
        front, prev = [Pc], Pc
        for t in np.linspace(0, 1, sweep + 2)[1:-1]:
            P = self.eps_point(PD, nadir[1] - t * (nadir[1] - ideal[1]), [prev, Pe] + x0[:4])
            if P is not None:
                front.append(P); prev = P
        front.append(Pe)
        return dict(Pc=Pc, Pe=Pe, ideal=ideal, nadir=nadir, front=front, x0=x0)

    def weighted(self, w, ideal, nadir):
        sc = np.maximum(nadir - ideal, 1e-9)
        f = lambda P: w * (self.C(P) - ideal[0]) / sc[0] + (1 - w) * (self.E(P) - ideal[1]) / sc[1]
        g = lambda P: w * self.gC(P) / sc[0] + (1 - w) * self.gE(P) / sc[1]
        return f, g

    def incumbent(self, PD, w, A, K=100):
        f, g = self.weighted(w, A['ideal'], A['nadir'])
        return self.solve(PD, f, g, A['x0'][:K] + A['front'])[1]

    def eps_point(self, PD, Emax, x0s):
        con = {'type': 'ineq', 'fun': lambda P: Emax - self.E(P), 'jac': lambda P: -self.gE(P)}
        b = self.solve(PD, self.C, self.gC, x0s, cons=(con,), tol=1e-6 * Emax)
        return None if b is None else b[1]

    # bands
    def band_uniform(self, rho): return np.broadcast_to(np.asarray(rho, float), (self.n,)) * self.span
    def band_ramp(self, tau_min): return self.rate * float(tau_min) / 60.0


class OpsRequest:
    """Incumbent-anchored request. Bands are absolute (MW) half-widths around the incumbent."""

    def __init__(self, sysm, PD, Pinc, Ft, scale, anchors_list, pool_cap=15, extra_cons=()):
        self.s, self.PD, self.Pinc, self.Ft = sysm, PD, np.asarray(Pinc, float), np.asarray(Ft, float)
        self.sc = np.asarray(scale, float); self.Finc = sysm.F(self.Pinc)
        self.e0 = float(np.linalg.norm((self.Finc - self.Ft) / self.sc))
        if not np.isfinite(self.e0) or self.e0 <= 0:
            raise ValueError('degenerate request: the target coincides with the incumbent (zero distance)')
        self.pool = [np.asarray(p, float) for p in anchors_list if p is not None]
        self.n_anchor = len(self.pool); self.pool_cap = pool_cap
        self.extra_cons = list(extra_cons)          # e.g. ramp constraints to fixed neighbouring hours
        self.fallbacks = 0                          # solves in which no start gave an accepted end point

    def d(self, P): return float(np.linalg.norm((self.s.F(P) - self.Ft) / self.sc))

    def grad(self, P):
        r = (self.s.F(P) - self.Ft) / self.sc; nr = max(np.linalg.norm(r), 1e-12)
        return r[0] / (nr * self.sc[0]) * self.s.gC(P) + r[1] / (nr * self.sc[1]) * self.s.gE(P)

    def box(self, band_mw):
        b = np.asarray(band_mw, float)
        return np.maximum(self.s.lo, self.Pinc - b), np.minimum(self.s.hi, self.Pinc + b)

    def _shrink(self, P, R):
        dd = np.abs(P - self.Pinc).sum()
        return P if dd <= R else self.Pinc + (R / dd) * (P - self.Pinc)

    def gap(self, R, band_mw, rng, k=4, extra_starts=()):
        lo, hi = self.box(band_mw)
        cons = list(self.extra_cons)
        if R < 1e8:
            cons.append({'type': 'ineq', 'fun': lambda P: R - np.abs(P - self.Pinc).sum(), 'jac': lambda P: -np.sign(P - self.Pinc)})
        x0 = [self.Pinc.copy()]
        for _ in range(4 * k):
            if len(x0) > k: break
            P = self.s.balance(rng.uniform(lo, hi), self.PD, lo, hi)
            if P is not None: x0.append(self._shrink(P, R))
        for e in list(extra_starts) + self.pool:
            P = self.s.balance(e, self.PD, lo, hi)
            if P is not None: x0.append(self._shrink(P, R))
        b = self.s.solve(self.PD, self.d, self.grad, x0, lo=lo, hi=hi, cons=tuple(cons))
        if b is None:
            self.fallbacks += 1
            return 1.0, self.Pinc.copy()
        self.pool.append(b[1].copy())
        if len(self.pool) > self.pool_cap:
            del self.pool[self.n_anchor]
        return min(b[0], self.e0) / self.e0, b[1]


def ramp_cons_vec(sysm, P_other):
    """Vector form: rate - |P - P_other| >= 0 componentwise, with Jacobians."""
    r = sysm.rate; n = sysm.n
    return [{'type': 'ineq', 'fun': lambda P: r - (P - P_other), 'jac': lambda P: -np.eye(n)},
            {'type': 'ineq', 'fun': lambda P: r - (P_other - P), 'jac': lambda P: np.eye(n)}]


def window_solve(sysm, PDs, fun_list, jac_list, x0s, boxes, extra=()):
    """Three-hour problem over X = [P_1, P_2, P_3] with per-hour balance, boxes and one-hour ramp limits.
    fun_list/jac_list give the objective as a sum over hours (entries may be None)."""
    n = sysm.n; r = sysm.rate
    def split(X): return X[:n], X[n:2 * n], X[2 * n:]
    def f(X):
        return float(sum(fn(p) for fn, p in zip(fun_list, split(X)) if fn is not None))
    def g(X):
        return np.concatenate([jn(p) if jn is not None else np.zeros(n) for jn, p in zip(jac_list, split(X))])
    cons = []
    for t in range(3):
        cons.append({'type': 'eq', 'fun': (lambda X, t=t: sysm.residual(split(X)[t], PDs[t])),
                     'jac': (lambda X, t=t: np.concatenate([sysm.residual_jac(split(X)[t]) if u == t else np.zeros(n) for u in range(3)]))})
    D = np.zeros((n, 3 * n))
    for t in range(2):
        M = np.zeros((n, 3 * n)); M[:, (t + 1) * n:(t + 2) * n] = np.eye(n); M[:, t * n:(t + 1) * n] = -np.eye(n)
        cons.append({'type': 'ineq', 'fun': (lambda X, M=M: r - M @ X), 'jac': (lambda X, M=M: -M)})
        cons.append({'type': 'ineq', 'fun': (lambda X, M=M: r + M @ X), 'jac': (lambda X, M=M: M)})
    cons += list(extra)
    lo = np.concatenate([b[0] for b in boxes]); hi = np.concatenate([b[1] for b in boxes])
    best = None
    for x0 in x0s:
        res = minimize(f, np.clip(x0, lo, hi), jac=g, method='SLSQP', bounds=list(zip(lo, hi)), constraints=cons,
                       options={'maxiter': 3000, 'ftol': 1e-12})
        X = res.x
        ok = (all(abs(sysm.residual(p, PD)) < 1e-5 for p, PD in zip(split(X), PDs)) and np.all(X >= lo - 1e-7)
              and np.all(X <= hi + 1e-7) and all(np.min(np.atleast_1d(c['fun'](X))) >= -1e-6 for c in cons if c['type'] == 'ineq'))
        if ok and (best is None or f(X) < best[0]):
            best = (f(X), X.copy())
    SolveStats.calls += 1
    if best is None:
        SolveStats.no_feasible_endpoint += 1
    return best
