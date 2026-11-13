"""Allowance counterfactuals computed with each compared search method as the inner solver.

For every 10-unit request of the main campaign that needs a counterfactual (G*(R0, rho0) > G*_phys + eps), the
band gap G(inf, rho) and the budget gap G(R, 1) are evaluated on fixed grids by
  - the proposal's own solver (multistart SLSQP with analytic gradients, eeldattain.model.Request.gap, k = 4),
  - the adapted counterfactual generators GS-R, FACE-R and NICE-R,
  - the repaired procedures POLY, ELLP and CGM,
  - the same solver without the target-directed starts (REF_NOTAR, incumbent, random points and its own earlier optima),
  - the derivative-free optimizers SLSQP-FD and CMA-ES (seed 5),
each with the 500-call budget of the default method comparison. DiCE-R and DAACE-R are left out because, at this
budget, they never improve on the best of GS-R, FACE-R and NICE-R in the default and tight configurations of the
method comparison (results/raw/robustness), and they take 20 to 50 times longer per run. COBYQA is left out for run time (about 3 s per call budget on wide boxes).

A counterfactual computed with method m is the smallest grid value at which G_m <= G*_phys + eps, with the stored
G*_phys of the campaign as the common threshold. This favors the methods twice: they receive the correct threshold,
and the first passing grid value is taken even when G_m is not monotone. A request for which no grid value passes
would be reported to the operator as unattainable by any band (or budget).
Output: results/raw/additional/cf_with_methods_p<part>.jsonl, one line per (request, kind, value, method).
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from eeldattain.model import System, Request
from eeldattain.paths import COEFFS, RAW_ATTAIN, RAW
RAW_ADD = RAW / "additional"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robustness'))
import run as RB                      # legacy generators and procedures (eeld-cf at 07e983a)
import cma
from cf_eeld.repair import BoundedProjector
from cf_eeld.config import ExperimentConfig, SolverConfig

RHO_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0)
R_GRID = (25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0)
LEGACY = ['GS', 'FACE', 'NICE_R', 'POLY', 'ELLP', 'CGM']
B, EPS, RINF = 500, 0.05, 1e9


class Stop(Exception):
    pass


def dfo(method, s, case, PD, Pinc, Gfun, R, rho, seed=5):
    lo, hi = RB.AT.band(s, Pinc, rho)
    feas = lambda P: (abs(P.sum() - PD) < 1e-4 and np.all(P >= lo - 1e-6) and np.all(P <= hi + 1e-6)
                      and np.abs(P - Pinc).sum() <= R + 1e-5)
    cons = [{'type': 'eq', 'fun': lambda P: P.sum() - PD}]
    if R < 1e8:
        cons.append({'type': 'ineq', 'fun': lambda P: R - np.abs(P - Pinc).sum()})
    st = {'n': 0, 'best': 1.0}
    try:
        if method in ('COBYQA', 'SLSQP_FD'):
            def f(P):
                if st['n'] >= B: raise Stop
                st['n'] += 1; v = Gfun(P)
                if feas(P) and v < st['best']: st['best'] = v
                return v
            if method == 'COBYQA':
                minimize(f, Pinc, method='COBYQA', bounds=list(zip(lo, hi)), constraints=cons, options={'maxfev': B + 10})
            else:
                minimize(f, Pinc, method='SLSQP', bounds=list(zip(lo, hi)), constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14})
        else:
            proj = BoundedProjector(case=case, incumbent=Pinc, demand=PD, redispatch_cap_mw=min(R, 1e6), rho=rho, feasibility_tol=1e-6)
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
    return st['best']


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--out', default=str(RAW_ADD))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = f'{a.out}/cf_with_methods_p{a.part}.jsonl'
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r['hour'], r['w'], r['cut'], r['kind'], r['value']))
    req = pd.concat([pd.read_csv(f, dtype={'S_units': str}) for f in sorted(glob.glob(f'{RAW_ATTAIN}/requests_e10*.csv'))])
    req = req[req.duplicate_of_w.isna() & req.rho_star.notna()].sort_values(['hour', 'w', 'cut']).reset_index(drop=True)
    # seeded random order, so that a run stopped early leaves a random sample of the requests
    req = req.iloc[np.random.default_rng(20261102).permutation(len(req))].reset_index(drop=True)
    if os.environ.get('EELD_KEYS'):           # rerun of given requests only, e.g. "1,0.5,0.05;3,0.75,0.025"
        keys = {tuple(float(x) for x in k.split(',')) for k in os.environ['EELD_KEYS'].split(';')}
        req = req[[(float(h), float(w), float(c)) in keys for h, w, c in zip(req.hour, req.w, req.cut)]].reset_index(drop=True)
    V = {(v['hour'], v['w'], v['cut']): v for f in sorted(glob.glob(f'{RAW_ATTAIN}/vectors_e10*.jsonl')) for v in map(json.loads, open(f))}
    S = System(str(COEFFS['10']))
    s = RB.Sys(RB.D10, 1.0); case = RB.repo_case(s)
    cfg = ExperimentConfig(runs=1, solver=SolverConfig(max_true_evals=B), baselines=RB.RR._baseline_config(B, 'scaled'))
    t0 = time.time()
    for i, r in req.iterrows():
        if i % a.nparts != a.part:
            continue
        v = V[(r.hour, r.w, r.cut)]
        Pinc, Ft, sc, Pfr = (np.array(v[k]) for k in ('Pinc', 'Ft', 'scale', 'Pfr'))
        PD = float(v['PD']); thr = float(r.G_phys) + EPS
        e0 = float(np.linalg.norm((S.F(Pinc) - Ft) / sc))
        Gfun = lambda P: min(1.0, float(np.linalg.norm((S.F(P) - Ft) / sc)) / e0)
        rq = Request(S, PD, Pinc, Ft, sc, Pfr, None, None)
        rq_nt = Request(S, PD, Pinc, Ft, sc, None, None, None)      # no target-directed start
        rng_nt = np.random.default_rng(int(r.hour) * 1000 + int(100 * r.w) + int(1000 * r.cut) + 7)
        rng = np.random.default_rng(int(r.hour) * 1000 + int(100 * r.w) + int(1000 * r.cut))
        for kind, grid in (('rho', RHO_GRID), ('R', R_GRID)):
            for x in grid:
                if (int(r.hour), float(r.w), float(r.cut), kind, x) in done:
                    continue
                R, rho = (RINF, x) if kind == 'rho' else (x, 1.0)
                rows = []
                t = time.time(); G = rq.gap(R, rho, rng, k=4)[0]
                rows.append(dict(method='REF', G=G, runtime=time.time() - t))
                t = time.time(); G = rq_nt.gap(R, rho, rng_nt, k=4)[0]
                rows.append(dict(method='REF_NOTAR', G=G, runtime=time.time() - t))
                for m in LEGACY:
                    t = time.time()
                    try:
                        _, _, P, _ = RB.RR.run_one(case, PD, Pinc, Ft, sc, m, 2, 2, 0.0, cfg, np.random.default_rng(5), min(R, 1e6), rho)
                        G = Gfun(P)
                    except Exception as ex:
                        G = np.nan
                    rows.append(dict(method=m, G=G, runtime=time.time() - t))
                for m in ('SLSQP_FD', 'CMAES'):
                    t = time.time(); G = dfo(m, s, case, PD, Pinc, Gfun, R, rho)
                    rows.append(dict(method=m, G=G, runtime=time.time() - t))
                with open(out, 'a') as f:
                    for d in rows:
                        f.write(json.dumps(dict(hour=int(r.hour), w=float(r.w), cut=float(r.cut), kind=kind, value=x,
                                                thr=thr, G_phys=float(r.G_phys), rho_star=float(r.rho_star),
                                                R_star=float(r.R_star), **d)) + '\n')
        print('part', a.part, 'request', i, int(r.hour), r.w, r.cut, round(time.time() - t0), flush=True)
    print('part', a.part, 'DONE', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
