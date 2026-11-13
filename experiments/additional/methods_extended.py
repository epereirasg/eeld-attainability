"""Extended method comparison in the default configuration: every hour, reachable and unreachable requests, ten seeds.

The nine configurations of experiments/robustness/run.py and experiments/methods_dfo/run.py use eight hours, reachable
requests and one method seed per instance. This runner keeps the default configuration (500 calls, R = 250 MW,
rho = 0.20, target-linked scale, efficient incumbent) and widens the design to all 24 hours, weights 0.25, 0.5 and 0.75,
cuts of 2.5% and 5%, unreachable requests included. An unreachable request keeps E_tar and takes C(P^E) as its cost
target, as in the main campaign. Every stochastic method runs with ten seeds (5, 11, ..., 59). The repaired
procedures and the adapted generators run once per seed. COBYQA and SLSQP-FD are deterministic and run once.
CMA-ES runs once per seed. The certified reference G_ref is the multistart bounded optimum of the robustness study.
With --variant convex the valve-point term is removed from the cost of the reference and of every method, which
separates the effect of nonsmoothness from that of the coupled power balance.
Output: results/raw/additional/methods_extended/raw_p<part>.csv, or methods_extended_convex/ with --variant convex
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robustness'))
import run as RB                                         # experiments/robustness/run.py (instances, methods, repair)
from scipy.optimize import minimize
import cma
from cf_eeld.repair import BoundedProjector
from cf_eeld.config import ExperimentConfig, SolverConfig
from eeldattain.paths import RAW

OUT = RAW / 'additional' / 'methods_extended'
SEEDS = [5 + 6 * i for i in range(10)]


class Stop(Exception):
    pass


def build(s, h, PD, w, cut, rng):
    """Instance of the robustness study, with unreachable requests kept (target at C(P^E), E_tar)."""
    x0 = [s.rand_feas(PD, rng) for _ in range(100)]
    Pc = RB.nlp(s, PD, s.C, s.gC, x0)[1]; Pe = RB.nlp(s, PD, s.E, s.gE, x0)[1]
    ideal = np.array([s.C(Pc), s.E(Pe)]); nadir = np.array([s.C(Pe), s.E(Pc)])
    Pinc = RB.AT.efficient_incumbent(s, PD, w, ideal, nadir, rng, 100)
    Finc = s.F(Pinc); Et = (1 - cut) * Finc[1]
    if Et < ideal[1]:
        return dict(Pinc=Pinc, Finc=Finc, Ft=np.array([s.C(Pe), Et]), Pfr=Pe, Pe=Pe, Pc=Pc, reachable=False)
    Pfr = RB.AT.eps_front(s, PD, Et, rng, 30, x0s=(Pinc, Pe))
    if Pfr is None:
        return None
    return dict(Pinc=Pinc, Finc=Finc, Ft=s.F(Pfr), Pfr=Pfr, Pe=Pe, Pc=Pc, reachable=True)


def dfo(m, seed, case, s, PD, Pinc, Gfun, B, R, rho):
    lo, hi = RB.AT.band(s, Pinc, rho)
    feas = lambda P: (abs(P.sum() - PD) < 1e-4 and np.all(P >= lo - 1e-6) and np.all(P <= hi + 1e-6)
                      and np.abs(P - Pinc).sum() <= R + 1e-5)
    cons = [{'type': 'eq', 'fun': lambda P: P.sum() - PD}, {'type': 'ineq', 'fun': lambda P: R - np.abs(P - Pinc).sum()}]
    st = {'n': 0, 'best': 1.0}
    try:
        if m in ('COBYQA', 'SLSQP_FD'):
            def f(P):
                if st['n'] >= B: raise Stop
                st['n'] += 1; v = Gfun(P)
                if feas(P) and v < st['best']: st['best'] = v
                return v
            if m == 'COBYQA':
                minimize(f, Pinc, method='COBYQA', bounds=list(zip(lo, hi)), constraints=cons, options={'maxfev': B + 10})
            else:
                minimize(f, Pinc, method='SLSQP', bounds=list(zip(lo, hi)), constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14})
        else:
            proj = BoundedProjector(case=case, incumbent=Pinc, demand=PD, redispatch_cap_mw=R, rho=rho, feasibility_tol=1e-6)
            z0 = case.encode(Pinc); zl, zh = lo[case.indep_idx], hi[case.indep_idx]
            def fz(z):
                P, _ = proj.repair_full(np.asarray(z), previous_z_or_p=Pinc, context='cma', log_event=False)
                if st['n'] >= B: raise Stop
                st['n'] += 1; v = Gfun(P)
                if feas(P) and v < st['best']: st['best'] = v
                return v
            es = cma.CMAEvolutionStrategy(z0, 0.1 * float(np.mean(zh - zl)),
                                          {'bounds': [list(zl), list(zh)], 'verbose': -9, 'seed': seed, 'maxfevals': B})
            while not es.stop():
                X = es.ask(); es.tell(X, [fz(x) for x in X])
    except Stop:
        pass
    return st['best'], st['n']


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--variant', default='', choices=['', 'convex'])
    a = ap.parse_args()
    out = OUT if not a.variant else OUT.parent / f'methods_extended_{a.variant}'
    out.mkdir(parents=True, exist_ok=True)
    f = out / f'raw_p{a.part}.csv'
    rows = pd.read_csv(f).to_dict('records') if f.exists() else []
    done = {(int(r['hour']), float(r['w']), float(r['cut'])) for r in rows}
    c = dict(RB.BASE); B, R, rho = c['B'], c['R'], c['rho']
    s = RB.Sys(RB.D10, c['k'])
    if a.variant == 'convex':
        s.e = np.zeros_like(s.e)                          # no valve-point term
    case = RB.repo_case(s)
    if a.variant == 'convex':
        case.cost = lambda P: s.C(np.asarray(P, float))   # the methods see the same convex cost
    dem = RB.load_demand(f'{RB.REPO}/data/pd_profile_10units.csv')
    jobs = [(h, w, cut) for h in range(1, 25) for w in (0.25, 0.5, 0.75) for cut in (0.025, 0.05)][a.part::a.parts]
    t0 = time.time()
    for h, w, cut in jobs:
        if (h, w, cut) in done:
            continue
        PD = float(dem.loc[dem.Hour == h, 'PD'].iloc[0])
        inst = build(s, h, PD, w, cut, np.random.default_rng(10000 * h + int(100 * w) + int(1000 * cut)))
        if inst is None:
            continue
        Pinc, Finc, Ft = inst['Pinc'], inst['Finc'], inst['Ft']
        ih = hashlib.sha1(np.round(np.concatenate([Pinc, Ft]), 6).tobytes()).hexdigest()[:12]
        sc = np.maximum(np.abs(Ft - Finc), 1e-6 * np.abs(Finc))
        e0 = float(np.linalg.norm((Finc - Ft) / sc))
        Gfun = lambda P: float(np.linalg.norm((s.F(P) - Ft) / sc) / e0)
        RB.AT.POOL.clear(); RB.AT.POOL.extend([inst['Pfr'], inst['Pe'], inst['Pc']])
        Gref, _ = RB.AT.G_bounded(s, Pinc, PD, Ft, sc, e0, R, rho, np.random.default_rng(1), k=4)
        Gphys, _ = RB.AT.G_bounded(s, Pinc, PD, Ft, sc, e0, 1e9, 1.0, np.random.default_rng(2), k=4)
        cfg = ExperimentConfig(runs=1, solver=SolverConfig(max_true_evals=B), baselines=RB.RR._baseline_config(B, 'scaled'))
        base = dict(hour=h, PD=PD, w=w, cut=cut, inst_hash=ih, reachable=inst['reachable'], G_ref=Gref, G_phys=Gphys)
        out = []
        for seed in SEEDS:
            for m in RB.METHODS:
                t = time.time()
                try:
                    res, _, P, _ = RB.RR.run_one(case, PD, Pinc, Ft, sc, m, 2, 2, 0.0, cfg, np.random.default_rng(seed), R, rho)
                    G = Gfun(P); ev = int(res.diagnostics.true_evals); err = ''
                except Exception as ex:
                    G, ev, err = np.nan, -1, str(ex)[:80]
                out.append(dict(base, method=m, mseed=seed, G=G, G_fallback=min(G, 1.0) if G == G else np.nan,
                                evals=ev, runtime=time.time() - t, err=err))
            t = time.time(); G, n = dfo('CMAES', seed, case, s, PD, Pinc, Gfun, B, R, rho)
            out.append(dict(base, method='CMAES', mseed=seed, G=G, G_fallback=min(G, 1.0), evals=n, runtime=time.time() - t, err=''))
        for m in ('COBYQA', 'SLSQP_FD'):
            t = time.time(); G, n = dfo(m, 0, case, s, PD, Pinc, Gfun, B, R, rho)
            out.append(dict(base, method=m, mseed=0, G=G, G_fallback=min(G, 1.0), evals=n, runtime=time.time() - t, err=''))
        rows += out
        pd.DataFrame(rows).to_csv(f, index=False)
        print(h, w, cut, 'reachable' if inst['reachable'] else 'unreachable', round(Gref, 3), round(time.time() - t0), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
