"""Bounded-gap helpers of the pilot, used by the method comparison for instance generation and G_ref.

Frozen copy of the helper functions of the pilot-study code. The pilot campaign driver
(``run_hour`` and the command-line entry point) is omitted because it is not part of the published
pipeline.
"""
import numpy as np
from .core import Sys, nlp

def band(s, Pinc, rho):
    r = np.broadcast_to(np.asarray(rho, float), Pinc.shape)
    return np.maximum(s.lo, Pinc - r * s.rng_), np.minimum(s.hi, Pinc + r * s.rng_)

def balance(P, PD, lo, hi):
    P = np.clip(P, lo, hi)
    for _ in range(60):
        r = PD - P.sum()
        if abs(r) < 1e-9: break
        room = (hi - P) if r > 0 else (P - lo)
        if room.sum() <= 1e-12: return None
        P = np.clip(P + r * room / room.sum(), lo, hi)
    return P if abs(P.sum() - PD) < 1e-6 else None

def starts_bounded(s, Pinc, PD, lo, hi, R, rng, k):
    out = [Pinc.copy()]
    for _ in range(4 * k):
        if len(out) > k: break
        P = balance(rng.uniform(lo, hi), PD, lo, hi)
        if P is None: continue
        d = np.abs(P - Pinc).sum()
        if d > R: P = Pinc + (R / d) * (P - Pinc)
        out.append(P)
    return out

def target_obj(s, Ft, sc):
    def f(P): return float(np.linalg.norm((s.F(P) - Ft) / sc))
    def g(P):
        r = (s.F(P) - Ft) / sc; nr = max(np.linalg.norm(r), 1e-12)
        return r[0] / (nr * sc[0]) * s.gC(P) + r[1] / (nr * sc[1]) * s.gE(P)
    return f, g

POOL = []   # known good dispatches for the current request (front point, previous optima)

def G_bounded(s, Pinc, PD, Ft, sc, e0, R, rho, rng, k=5, extra=()):
    lo, hi = band(s, Pinc, rho)
    f, g = target_obj(s, Ft, sc)
    cons = () if R >= 1e4 else ({'type': 'ineq', 'fun': lambda P: R - np.abs(P - Pinc).sum(),
                                  'jac': lambda P: -np.sign(P - Pinc)},)
    x0 = starts_bounded(s, Pinc, PD, lo, hi, R, rng, k)
    for e in list(extra) + POOL:
        P = balance(np.asarray(e, float), PD, lo, hi)
        if P is None: continue
        d = np.abs(P - Pinc).sum()
        if d > R: P = Pinc + (R / d) * (P - Pinc)
        x0.append(P)
    b = nlp(s, PD, f, g, x0, lo=lo, hi=hi, cons=cons, tol=1e-6)
    if b and len(POOL) < 12: POOL.append(b[1].copy())
    return (min(b[0], e0) / e0, b[1]) if b else (1.0, Pinc)

def efficient_incumbent(s, PD, w, ideal, nadir, rng, K):
    sc = np.maximum(nadir - ideal, 1e-9)
    f = lambda P: w * (s.C(P) - ideal[0]) / sc[0] + (1 - w) * (s.E(P) - ideal[1]) / sc[1]
    g = lambda P: w * s.gC(P) / sc[0] + (1 - w) * s.gE(P) / sc[1]
    return nlp(s, PD, f, g, [s.rand_feas(PD, rng) for _ in range(K)])[1]

def eps_front(s, PD, Emax, rng, K, x0s=()):
    con = {'type': 'ineq', 'fun': lambda P: Emax - s.E(P), 'jac': lambda P: -s.gE(P)}
    b = nlp(s, PD, s.C, s.gC, list(x0s) + [s.rand_feas(PD, rng) for _ in range(K)], cons=(con,), tol=1e-6 * Emax)
    return None if b is None else b[1]
