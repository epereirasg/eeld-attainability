"""Dissection of the fixed-neighbour window study (RQ10).

For every window hour h (2-23) and weight w, the script rebuilds the ramp-feasible incumbent schedule
exactly as run_window does (same anchors, seeds and starts), and measures, with both neighbours frozen:

* the ramp headroom of every unit at hour h, up and down, alone and inside the reference band b(tau0);
* whether the admissible set of hour h is locked, i.e. some direction (up or down) has no headroom on
  any unit, so that no redispatch can keep the power balance;
* how far the per-hour efficient incumbents are from ramp feasibility (incumbent-construction check).

As an alternative construction, it also builds the ramp-feasible schedule closest (least squares) to
the per-hour efficient incumbents and recomputes the fixed-neighbour gaps on it.

Output: results/raw/supplementary/window_lock.csv (one row per hour, weight and construction) and
window_lock_gaps.csv (fixed-neighbour gaps of the alternative construction).
"""
import argparse, os, time
import numpy as np, pandas as pd
from eeldattain.paths import RAW_SUPP, RAW_OPS, ROOT
from eeldattain.runio import guard_config
from eeldops import experiments as X
from eeldops.model import window_solve, ramp_cons_vec

TOL = 1e-3      # MW; headroom below this counts as none


def headroom(s, Pm, Ph, Pp, band=None):
    lo = np.maximum.reduce([s.lo, Pm - s.rate, Pp - s.rate])
    hi = np.minimum.reduce([s.hi, Pm + s.rate, Pp + s.rate])
    if band is not None:
        lo = np.maximum(lo, Ph - band); hi = np.minimum(hi, Ph + band)
    return hi - Ph, Ph - lo


def ramp_violation(s, Ps):
    return float(max(np.max(np.abs(Ps[t + 1] - Ps[t]) - s.rate) for t in range(len(Ps) - 1)))


def closest_schedule(s, PDs, INC3, starts):
    n = s.n
    fs = [(lambda P, r=r: float(np.sum((P - r) ** 2))) for r in INC3]
    gs = [(lambda P, r=r: 2 * (P - r)) for r in INC3]
    return window_solve(s, PDs, fs, gs, starts, [(s.lo, s.hi)] * 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(RAW_SUPP))
    ap.add_argument('--hours', default='2-23')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lo_h, hi_h = (int(x) for x in a.hours.split('-'))
    guard_config(a.out, 'window_lock', dict(hours=a.hours, OPS=X.OPS, tol=TOL, cache=str((RAW_OPS / 'cache').relative_to(ROOT))))
    X.RES = str(RAW_OPS)                       # reuse the stored anchors and incumbents (JSON cache)
    s, _, dem = X.system(); n = s.n
    band = s.band_ramp(X.OPS['tau0']); R0 = X.OPS['R0']
    rows, gaps = [], []
    for h in range(max(2, lo_h), min(23, hi_h) + 1):
        PDs = [float(dem.loc[dem.hour == t, 'PD'].iloc[0]) for t in (h - 1, h, h + 1)]
        As, INCs = zip(*[X.hour_setup(s, t, PD, 'ops') for t, PD in zip((h - 1, h, h + 1), PDs)])
        for w in X.OPS['weights']:
            rng = np.random.default_rng(X.seed_of(3, h, w))         # identical to run_window
            fs, gs = zip(*[s.weighted(w, A['ideal'], A['nadir']) for A in As])
            INC3 = [INCs[t][w] for t in range(3)]
            starts = [np.concatenate(INC3)] + [np.concatenate([s.rand_feasible(PD, rng) for PD in PDs]) for _ in range(5)]
            sol_eff = window_solve(s, PDs, fs, gs, starts, [(s.lo, s.hi)] * 3)
            sol_cls = closest_schedule(s, PDs, INC3, starts)
            viol = ramp_violation(s, INC3)
            for cons_name, sol in (('efficient_window', sol_eff), ('closest_to_hourly', sol_cls)):
                base = dict(hour=h, w=w, PD_prev=PDs[0], PD=PDs[1], PD_next=PDs[2], construction=cons_name,
                            hourly_incumbents_ramp_violation_mw=viol)
                if sol is None:
                    rows.append(dict(base, schedule_found=False)); continue
                Xs = sol[1]; Pm, Ph, Pp = Xs[:n], Xs[n:2 * n], Xs[2 * n:]
                up, dn = headroom(s, Pm, Ph, Pp); upb, dnb = headroom(s, Pm, Ph, Pp, band)
                rows.append(dict(base, schedule_found=True,
                                 units_up=int((up > TOL).sum()), units_down=int((dn > TOL).sum()),
                                 up_total_mw=float(up[up > 0].sum()), down_total_mw=float(dn[dn > 0].sum()),
                                 units_up_band=int((upb > TOL).sum()), units_down_band=int((dnb > TOL).sum()),
                                 locked=bool((up > TOL).sum() == 0 or (dn > TOL).sum() == 0),
                                 locked_direction=('up' if (up > TOL).sum() == 0 else '') + ('down' if (dn > TOL).sum() == 0 else ''),
                                 units_at_pmax=int((Ph > s.hi - TOL).sum()), units_at_pmin=int((Ph < s.lo + TOL).sum()),
                                 shift_from_hourly_mw=float(np.abs(Ph - INC3[1]).sum())))
                if cons_name == 'closest_to_hourly':
                    for cut in X.OPS['cuts']:
                        sd = X.seed_of(4, h, w, cut); rng2 = np.random.default_rng(sd)
                        req, Pfr, Ft, reach, eta = X.build_request(s, PDs[1], As[1], Ph, cut)
                        Ga, _ = req.gap(R0, band, rng2, k=X.OPS['kb'])
                        req.extra_cons = ramp_cons_vec(s, Pm) + ramp_cons_vec(s, Pp)
                        Gb, _ = req.gap(R0, band, rng2, k=X.OPS['kb'])
                        gaps.append(dict(hour=h, w=w, cut=cut, seed=sd, construction=cons_name, G_single=Ga, G_fixed=Gb,
                                         solver_fallbacks=req.fallbacks))
            print(f'[window_lock] h={h} w={w} viol={viol:.1f} '
                  f'eff_locked={rows[-2].get("locked")} closest_locked={rows[-1].get("locked")}', flush=True)
    pd.DataFrame(rows).to_csv(f'{a.out}/window_lock.csv', index=False)
    pd.DataFrame(gaps).to_csv(f'{a.out}/window_lock_gaps.csv', index=False)
    print('DONE', time.strftime('%Y-%m-%dT%H:%M:%S'))


if __name__ == '__main__':
    main()
