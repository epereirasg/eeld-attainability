"""Shared helpers for the additional experiments.

They rebuild the stored 10-unit requests of results/raw/attainability from their stored vectors, so new
quantities are computed for exactly the incumbents, targets and scales behind the reported numbers.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from eeldattain.model import System, Request, balance
from eeldattain.paths import COEFFS, RAW_ATTAIN, RAW

RAW_ADD = Path(os.environ.get('EELD_RAW_ADD', RAW / 'additional'))
EPS = 0.05
BIG = 1e9


def load_requests(case='10', distinct=True):
    tags = ('e10a', 'e10b') if case == '10' else ('e40a', 'e40b')
    R = pd.concat([pd.read_csv(RAW_ATTAIN / f'requests_{t}.csv') for t in tags], ignore_index=True)
    V = [json.loads(l) for t in tags for l in open(RAW_ATTAIN / f'vectors_{t}.jsonl')]
    vk = {(int(v['hour']), float(v['w']), float(v['cut'])): v for v in V}
    if distinct:
        R = R[R.duplicate_of_w.isna()].reset_index(drop=True)
    if os.environ.get('EELD_HOURS'):          # restrict a rerun to some hours (merged afterwards)
        R = R[R.hour.isin([int(x) for x in os.environ['EELD_HOURS'].split(',')])].reset_index(drop=True)
    R['key'] = [(int(h), float(w), float(c)) for h, w, c in zip(R.hour, R.w, R.cut)]
    return R, vk


def make_request(sysm, v, pool_cap=15):
    Pinc = np.array(v['Pinc']); Pfr = np.array(v['Pfr'])
    req = Request(sysm, float(v['PD']), Pinc, np.array(v['Ft']), np.array(v['scale']), Pfr, None, None,
                  pool_cap=pool_cap)
    if v.get('P_ref') is not None:
        req.pool.append(np.array(v['P_ref']))
    return req


def emin_bounded(req, R, rho):
    """Minimum emission over the bounded set A_h(R, rho). E is convex and A_h is a polytope, so the
    problem is convex and SLSQP returns its global minimum (checked from two starts)."""
    s = req.s
    lo, hi = req.band(rho)
    cons = [{'type': 'eq', 'fun': lambda P: P.sum() - req.PD, 'jac': lambda P: np.ones_like(P)}]
    if R < BIG:
        cons.append({'type': 'ineq', 'fun': lambda P: R - np.abs(P - req.Pinc).sum(),
                     'jac': lambda P: -np.sign(P - req.Pinc)})
    best = None
    starts = [req.Pinc.copy()]
    for p in req.pool[:1]:
        q = balance(p, req.PD, lo, hi)
        if q is not None:
            starts.append(req._shrink(q, R))
    for x0 in starts:
        r = minimize(s.E, np.clip(x0, lo, hi), jac=s.gE, method='SLSQP', bounds=list(zip(lo, hi)),
                     constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
        P = r.x
        ok = abs(P.sum() - req.PD) < 1e-5 and np.all(P >= lo - 1e-7) and np.all(P <= hi + 1e-7) and \
            (R >= BIG or np.abs(P - req.Pinc).sum() <= R + 1e-6)
        if ok and (best is None or s.E(P) < best[0]):
            best = (s.E(P), P.copy())
    if best is None:
        return s.E(req.Pinc), req.Pinc.copy()
    return best


def eta_bounded(req, Emin_b, E_tar):
    Einc = req.Finc[1]
    return float(min(1.0, max(0.0, (Einc - Emin_b) / (Einc - E_tar))))


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def lap(self):
        t = time.perf_counter(); d = t - self.t; self.t = t
        return d
