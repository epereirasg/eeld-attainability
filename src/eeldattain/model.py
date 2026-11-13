"""Nonsmooth economic-emission dispatch model and constrained solvers used by the campaign.

Cost has valve-point terms, C_i = a_i P_i^2 + b_i P_i + c_i + |e_i sin(f_i (Pmin_i - P_i))| in $/h.
Emission follows the standard DEED form E_i = alpha_i + beta_i P_i + gamma_i P_i^2 + zeta_i exp(lambda_i P_i)
in lb/h. Setting ``escale`` reproduces the 1e-2 scaling of the quadratic emission terms used by the
earlier code. Power balance is lossless, sum(P) = PD (losses are in ``eeldops.model``).

Tolerances shared by all solves (do not change without rerunning the campaign):
  balance residual < 1e-5 MW, bounds violated by < 1e-7 MW, inequality constraints >= -tol (default 1e-6).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .paths import DATA as _DATA

DATA = str(_DATA)          # kept as a string for backward compatibility with the campaign scripts
BALANCE_TOL = 1e-5
BOUND_TOL = 1e-7


class SolveStats:
    """Counts of multistart SLSQP calls in which no end point passed the feasibility check."""
    calls = 0
    no_feasible_endpoint = 0

    @classmethod
    def reset(cls):
        cls.calls = 0
        cls.no_feasible_endpoint = 0


class System:
    """Lossless N-unit system read from a coefficient CSV (columns Pmin, Pmax, a, b, c, e, f, alpha..lambda)."""

    def __init__(self, coeff_csv: str, escale: float = 1.0):
        d = pd.read_csv(coeff_csv)
        need = ['Pmin', 'Pmax', 'a', 'b', 'c', 'e', 'f', 'alpha', 'beta', 'gamma', 'zeta', 'lambda']
        missing = [k for k in need if k not in d.columns]
        if missing:
            raise ValueError(f'{coeff_csv}: missing coefficient columns {missing}')
        if d[need].isna().any().any() or (d.Pmax <= d.Pmin).any():
            raise ValueError(f'{coeff_csv}: missing values or Pmax <= Pmin')
        self.n = len(d)
        self.lo = d.Pmin.to_numpy(float); self.hi = d.Pmax.to_numpy(float); self.span = self.hi - self.lo
        self.a, self.b, self.c, self.e, self.f = (d[k].to_numpy(float) for k in 'abcef')
        self.al, self.be, self.ga, self.ze, self.la = (d[k].to_numpy(float) for k in ['alpha', 'beta', 'gamma', 'zeta', 'lambda'])
        self.k = float(escale)

    def C(self, P):
        return float(np.sum(self.a * P**2 + self.b * P + self.c + np.abs(self.e * np.sin(self.f * (self.lo - P)))))

    def E(self, P):
        return float(np.sum(self.k * (self.al + self.be * P + self.ga * P**2) + self.ze * np.exp(self.la * P)))

    def F(self, P):
        return np.array([self.C(P), self.E(P)])

    def gC(self, P):
        """Gradient of C. The valve-point term is differentiated almost everywhere (sign at the kinks)."""
        z = self.f * (self.lo - P)
        return 2 * self.a * P + self.b - np.sign(self.e * np.sin(z)) * self.e * np.cos(z) * self.f

    def gE(self, P):
        return self.k * (self.be + 2 * self.ga * P) + self.ze * self.la * np.exp(self.la * P)

    def rand_feasible(self, PD, rng, lo=None, hi=None):
        lo = self.lo if lo is None else lo; hi = self.hi if hi is None else hi
        for _ in range(200):
            P = balance(rng.uniform(lo, hi), PD, lo, hi)
            if P is not None:
                return P
        raise RuntimeError('no feasible sample')


def balance(P, PD, lo, hi):
    """Proportional headroom redistribution to meet sum(P) = PD inside [lo, hi]."""
    P = np.clip(np.asarray(P, float), lo, hi)
    for _ in range(80):
        r = PD - P.sum()
        if abs(r) < 1e-9:
            return P
        room = (hi - P) if r > 0 else (P - lo)
        if room.sum() <= 1e-12:
            return None
        P = np.clip(P + r * room / room.sum(), lo, hi)
    return P if abs(P.sum() - PD) < 1e-6 else None


def solve(sysm, PD, fun, jac, x0s, lo=None, hi=None, cons=(), tol=1e-6):
    """Multistart SLSQP. Returns (value, P) of the best feasible end point, or None.

    An end point is accepted only if it meets the balance, the bounds and every inequality within the
    module tolerances. ``None`` means that no start produced an accepted end point. The event is counted
    in ``SolveStats`` so that runners can report it instead of hiding it.
    """
    lo = sysm.lo if lo is None else lo; hi = sysm.hi if hi is None else hi
    cons = (*cons, *getattr(sysm, 'extra_cons', ()))   # network line limits (eeldattain.network), none otherwise
    C = [{'type': 'eq', 'fun': lambda P: P.sum() - PD, 'jac': lambda P: np.ones_like(P)}, *cons]
    best = None
    for x0 in x0s:
        r = minimize(fun, np.clip(x0, lo, hi), jac=jac, method='SLSQP', bounds=list(zip(lo, hi)),
                     constraints=C, options={'maxiter': 2000, 'ftol': 1e-12})
        P = r.x
        ok = (abs(P.sum() - PD) < 1e-5 and np.all(P >= lo - 1e-7) and np.all(P <= hi + 1e-7)
              and all(np.all(c['fun'](P) >= -tol) for c in cons))
        if ok:
            v = fun(P)
            if best is None or v < best[0]:
                best = (v, P.copy())
    SolveStats.calls += 1
    if best is None:
        SolveStats.no_feasible_endpoint += 1
    return best


class Request:
    """Incumbent, target and scale for one (hour, weight, cut) request."""

    def __init__(self, sysm, PD, Pinc, Ft, scale, Pfr, Pc, Pe, pool_cap=15):
        self.s, self.PD, self.Pinc, self.Ft = sysm, PD, Pinc, Ft
        self.pool_cap = pool_cap
        self.sc = np.asarray(scale, float)
        self.Finc = sysm.F(Pinc)
        self.e0 = float(np.linalg.norm((self.Finc - Ft) / self.sc))
        if not np.isfinite(self.e0) or self.e0 <= 0:
            raise ValueError('degenerate request: the target coincides with the incumbent (zero distance)')
        self.pool = [p for p in (Pfr, Pc, Pe) if p is not None]
        self.fallbacks = 0

    def d(self, P):
        return float(np.linalg.norm((self.s.F(P) - self.Ft) / self.sc))

    def grad(self, P):
        r = (self.s.F(P) - self.Ft) / self.sc
        nr = max(np.linalg.norm(r), 1e-12)
        return r[0] / (nr * self.sc[0]) * self.s.gC(P) + r[1] / (nr * self.sc[1]) * self.s.gE(P)

    def band(self, rho):
        r = np.broadcast_to(np.asarray(rho, float), self.Pinc.shape)
        return np.maximum(self.s.lo, self.Pinc - r * self.s.span), np.minimum(self.s.hi, self.Pinc + r * self.s.span)

    def gap(self, R, rho, rng, k=4):
        """Bounded attainability G*(R, rho) by multistart SLSQP, and its optimizer.

        Starts are the incumbent, up to k random balanced points of the band (shrunk into the budget) and
        the pool of target-directed points. R >= 1e8 means no budget constraint. The value is clipped at 1
        because the incumbent is always admissible. If no start yields an accepted end point, the
        incumbent is returned (G = 1, a valid upper bound) and ``self.fallbacks`` is incremented.
        """
        lo, hi = self.band(rho)
        cons = () if R >= 1e8 else ({'type': 'ineq', 'fun': lambda P: R - np.abs(P - self.Pinc).sum(),
                                      'jac': lambda P: -np.sign(P - self.Pinc)},)
        x0 = [self.Pinc.copy()]
        for _ in range(4 * k):
            if len(x0) > k:
                break
            P = balance(rng.uniform(lo, hi), self.PD, lo, hi)
            if P is not None:
                x0.append(self._shrink(P, R))
        for e in self.pool:                      # target-directed starts projected onto the bounded set
            P = balance(e, self.PD, lo, hi)
            if P is not None:
                x0.append(self._shrink(P, R))
        b = solve(self.s, self.PD, self.d, self.grad, x0, lo=lo, hi=hi, cons=cons)
        if b is None:
            self.fallbacks += 1
            return 1.0, self.Pinc.copy()
        self.pool.append(b[1].copy())
        if len(self.pool) > self.pool_cap:  # keep the three anchors plus the most recent optima
            del self.pool[3]
        return min(b[0], self.e0) / self.e0, b[1]

    def _shrink(self, P, R):
        dd = np.abs(P - self.Pinc).sum()
        return P if dd <= R else self.Pinc + (R / dd) * (P - self.Pinc)


def weighted_point(sysm, PD, w, ideal, nadir, x0s):
    sc = np.maximum(nadir - ideal, 1e-9)
    f = lambda P: w * (sysm.C(P) - ideal[0]) / sc[0] + (1 - w) * (sysm.E(P) - ideal[1]) / sc[1]
    g = lambda P: w * sysm.gC(P) / sc[0] + (1 - w) * sysm.gE(P) / sc[1]
    return solve(sysm, PD, f, g, x0s)


def eps_point(sysm, PD, Emax, x0s):
    con = {'type': 'ineq', 'fun': lambda P: Emax - sysm.E(P), 'jac': lambda P: -sysm.gE(P)}
    b = solve(sysm, PD, sysm.C, sysm.gC, x0s, cons=(con,), tol=1e-6 * Emax)
    return None if b is None else b[1]


def anchors(sysm, PD, rng, K=100, sweep=9):
    """Single-objective optima and an epsilon sweep of the front, all by multistart SLSQP."""
    x0 = [sysm.rand_feasible(PD, rng) for _ in range(K)]
    Pc = solve(sysm, PD, sysm.C, sysm.gC, x0)[1]
    Pe = solve(sysm, PD, sysm.E, sysm.gE, x0)[1]
    ideal = np.array([sysm.C(Pc), sysm.E(Pe)]); nadir = np.array([sysm.C(Pe), sysm.E(Pc)])
    front, prev = [Pc], Pc
    for t in np.linspace(0, 1, sweep + 2)[1:-1]:
        P = eps_point(sysm, PD, nadir[1] - t * (nadir[1] - ideal[1]), [prev, Pe] + x0[:4])
        if P is not None:
            front.append(P); prev = P
    front.append(Pe)
    return dict(Pc=Pc, Pe=Pe, ideal=ideal, nadir=nadir, front=front, x0=x0)


def efficient_incumbent(sysm, PD, w, A, K=100):
    b = weighted_point(sysm, PD, w, A['ideal'], A['nadir'], A['x0'][:K] + A['front'])
    return b[1]
