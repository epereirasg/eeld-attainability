"""Global solves with SCIP for the anchors, the incumbents and the targets of the lossless model.

The campaign builds every anchor, incumbent and target by multistart SLSQP. On the valve-point cost these
solves can end in a local minimum. With ``certify=True`` the campaign passes each of them through
``improve``. SCIP (spatial branch and bound) solves the same program to a relative gap of 1e-6, SLSQP polishes
the SCIP point, and the polished point replaces the multistart point only when it is better by more than a
relative 1e-7. Hours whose multistart points are already global therefore keep them bit for bit, and the random
stream of the campaign is unchanged, because SCIP draws no random numbers from it.

Programs (P in Omega_h in every case):
  'C'   min C(P)                         minimum-cost anchor P^C, or the target of Eq. (5) with ``Ecap``
  'E'   min E(P)                         minimum-emission anchor P^E
  'W'   min w (C - C_id)/sC + (1 - w)(E - E_id)/sE      incumbent of Eq. (4)
"""
from __future__ import annotations
import time
import numpy as np

REL_ADOPT = 1e-7


def _scip_model(s, PD, time_limit, relgap):
    from pyscipopt import Model, quicksum, sin, exp
    m = Model(); m.hideOutput()
    m.setParam('timing/clocktype', 1)          # CPU time, so that a busy machine does not shorten the limit
    m.setParam('limits/time', time_limit); m.setParam('limits/gap', relgap)
    n = s.n
    P = [m.addVar(lb=s.lo[i], ub=s.hi[i], name=f'P{i}') for i in range(n)]
    B = getattr(s, 'B', None)
    if B is not None and np.any(B != 0):      # Kron losses (eeldops.model.OpsSystem): sum P = PD + P^T B P
        m.addCons(quicksum(P) == PD + quicksum(float(B[i, j]) * P[i] * P[j] for i in range(n) for j in range(n)))
    else:
        m.addCons(quicksum(P) == PD)
    if hasattr(s, 'reserve_cap'):            # spinning reserve of the reserve variant: sum min(Pmax - P, cap) >= share PD
        r = [m.addVar(lb=0, ub=float(s.reserve_cap[i]), name=f'r{i}') for i in range(n)]
        for i in range(n):
            m.addCons(r[i] <= s.hi[i] - P[i])
        m.addCons(quicksum(r) >= s.reserve_share * PD)
    vp = []
    for i in range(n):
        if s.e[i] == 0:
            continue
        # explicit bounded sine argument (the formulation of experiments/additional/scip_certificate.py)
        wa = m.addVar(lb=s.f[i] * (s.lo[i] - s.hi[i]), ub=0.0, name=f'w{i}')
        m.addCons(wa == s.f[i] * (s.lo[i] - P[i]))
        z = m.addVar(lb=-1, ub=1, name=f'z{i}'); m.addCons(z == sin(wa))
        y = m.addVar(lb=0, ub=1, name=f'y{i}'); m.addCons(y == abs(z))
        vp.append(s.e[i] * y)
    Cexpr = quicksum(s.a[i] * P[i] * P[i] + s.b[i] * P[i] + s.c[i] for i in range(n)) + quicksum(vp)
    k = getattr(s, 'k', 1.0)
    Eexpr = quicksum(k * (s.al[i] + s.be[i] * P[i] + s.ga[i] * P[i] * P[i]) for i in range(n)) + \
        quicksum(s.ze[i] * exp(s.la[i] * P[i]) for i in range(n))
    cv = m.addVar(lb=None, name='C'); ev = m.addVar(lb=None, name='E')
    m.addCons(cv == Cexpr); m.addCons(ev == Eexpr)
    return m, P, cv, ev


def scip_solve(s, PD, kind, Ecap=None, w=None, ideal=None, nadir=None, time_limit=120.0, relgap=1e-6):
    """SCIP solve of one program. Returns dict(P, lb, status, time_s), P = None if SCIP found no point."""
    m, P, cv, ev = _scip_model(s, PD, time_limit, relgap)
    if Ecap is not None:
        m.addCons(ev <= Ecap)
    if kind == 'C':
        m.setObjective(cv, 'minimize')
    elif kind == 'E':
        m.setObjective(ev, 'minimize')
    elif kind == 'W':
        sc = np.maximum(np.asarray(nadir, float) - np.asarray(ideal, float), 1e-9)
        m.setObjective(w * (cv - ideal[0]) / sc[0] + (1 - w) * (ev - ideal[1]) / sc[1], 'minimize')
    else:
        raise ValueError(kind)
    t0 = time.perf_counter(); m.optimize(); el = time.perf_counter() - t0
    Pb = np.array([m.getVal(x) for x in P]) if m.getNSols() > 0 else None
    return dict(P=Pb, lb=m.getDualbound(), status=m.getStatus(), time_s=el)


def _objective(s, kind, w=None, ideal=None, nadir=None):
    if kind == 'C':
        return s.C, s.gC
    if kind == 'E':
        return s.E, s.gE
    sc = np.maximum(np.asarray(nadir, float) - np.asarray(ideal, float), 1e-9)
    f = lambda P: w * (s.C(P) - ideal[0]) / sc[0] + (1 - w) * (s.E(P) - ideal[1]) / sc[1]
    g = lambda P: w * s.gC(P) / sc[0] + (1 - w) * s.gE(P) / sc[1]
    return f, g


def _feasible(s, PD, P, Ecap):
    res = (lambda P: s.residual(P, PD)) if hasattr(s, 'residual') else (lambda P: P.sum() - PD)
    if P is not None and hasattr(s, 'reserve_cap') and np.minimum(s.hi - P, s.reserve_cap).sum() < s.reserve_share * PD - 1e-6:
        return False
    return (P is not None and abs(res(P)) < 1e-5 and np.all(P >= s.lo - 1e-7) and np.all(P <= s.hi + 1e-7)
            and (Ecap is None or s.E(P) <= Ecap * (1 + 1e-6)))


def improve(s, PD, kind, P_ms, Ecap=None, w=None, ideal=None, nadir=None, time_limit=120.0, log=None):
    """Return the better of the multistart point ``P_ms`` and the polished SCIP point.

    ``log`` (a list) receives one record per call with the two values, SCIP's lower bound and the decision.
    """
    from .model import solve
    f, g = _objective(s, kind, w, ideal, nadir)
    r = scip_solve(s, PD, kind, Ecap=Ecap, w=w, ideal=ideal, nadir=nadir, time_limit=time_limit)
    cand = []
    if _feasible(s, PD, r['P'], Ecap):
        cand.append(np.clip(r['P'], s.lo, s.hi))
        cons = () if Ecap is None else ({'type': 'ineq', 'fun': lambda P: Ecap - s.E(P), 'jac': lambda P: -s.gE(P)},)
        tol = 1e-6 * (Ecap if Ecap is not None else 1.0)
        if hasattr(s, 'residual'):           # loss model, own balance constraint
            pol = s.solve(PD, f, g, [cand[0]], cons=cons, tol=tol)
        else:
            pol = solve(s, PD, f, g, [cand[0]], cons=cons, tol=tol)
        if pol is not None and _feasible(s, PD, pol[1], Ecap):
            cand.append(pol[1])
    best = None if P_ms is None else (f(P_ms), P_ms)
    v_ms = np.nan if P_ms is None else best[0]
    for P in cand:
        v = f(P)
        if best is None or v < best[0] - REL_ADOPT * max(abs(best[0]), 1.0):
            best = (v, P.copy())
    adopted = P_ms is None or best[1] is not P_ms
    if log is not None:
        log.append(dict(kind=kind, w=w, Ecap=Ecap, v_multistart=v_ms, v_final=best[0] if best else np.nan,
                        scip_lb=r['lb'], scip_status=r['status'], scip_time_s=r['time_s'], adopted=bool(adopted)))
    return None if best is None else best[1]


def certify_anchors(s, PD, A, log=None, sweep=9):
    """Pass P^C and P^E through ``improve``. If either changes, recompute the ideal and nadir values and the
    epsilon sweep of the front that seeds the incumbents, exactly as ``model.anchors`` builds them."""
    from .model import eps_point
    Pc = improve(s, PD, 'C', A['Pc'], log=log)
    Pe = improve(s, PD, 'E', A['Pe'], log=log)
    if Pc is A['Pc'] and Pe is A['Pe']:
        return A
    ideal = np.array([s.C(Pc), s.E(Pe)]); nadir = np.array([s.C(Pe), s.E(Pc)])
    x0 = A['x0']
    front, prev = [Pc], Pc
    for t in np.linspace(0, 1, sweep + 2)[1:-1]:
        P = eps_point(s, PD, nadir[1] - t * (nadir[1] - ideal[1]), [prev, Pe] + x0[:4])
        if P is not None:
            front.append(P); prev = P
    front.append(Pe)
    return dict(Pc=Pc, Pe=Pe, ideal=ideal, nadir=nadir, front=front, x0=x0)
