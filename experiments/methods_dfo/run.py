"""General derivative-free optimizers on the robustness-study instances (RQ5).

Instances are regenerated with the robustness-study generator and seeds, so every run pairs with the
POLY/ELLP/CGM and counterfactual-baseline runs stored in experiments/robustness.
Methods: COBYQA (SciPy), finite-difference SLSQP, and CMA-ES (three seeds) on the repaired reduced space.
Only feasible evaluated points can become the returned recommendation, and the incumbent is the fallback.
"""
import hashlib, sys, time, argparse, os
from pathlib import Path
import numpy as np, pandas as pd
from eeldattain.paths import RAW_DFO
from eeldattain.runio import guard_config
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robustness'))
import run as RB                      # robustness generator, identical seeds and instances
from scipy.optimize import minimize
import cma
from cf_eeld.repair import BoundedProjector

CONFIGS = {'default': dict(), 'budget200': dict(B=200), 'budget2000': dict(B=2000), 'allow50_005': dict(R=50.0, rho=0.05)}


class Stop(Exception):
    pass


CONFIGS_ALL = {**CONFIGS, **{k: RB.CONFIGS[k] for k in ('allow100_010', 'scale_cloud', 'emission_repo', 'inc_random', 'method_seed2')}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--hours', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default=str(RAW_DFO), help='output directory (default: results/raw/methods_dfo)')
    ap.add_argument('--configs', default=','.join(CONFIGS), help='comma-separated configurations of the robustness study')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    configs = {k: CONFIGS_ALL[k] for k in a.configs.split(',')}
    guard_config(a.out, a.tag, dict(hours=a.hours, configs=configs, base=RB.BASE, cma_seeds=[5, 11, 17]))
    dem = RB.load_demand(f'{RB.REPO}/data/pd_profile_10units.csv')
    rows, t0 = [], time.time()
    fout = f'{a.out}/raw_{a.tag}.csv'
    if os.path.exists(fout):                       # resume after an interruption (complete blocks only)
        rows = pd.read_csv(fout).to_dict('records')
    done = {(int(r['hour']), float(r['w'])) for r in rows} if rows else set()
    done_h = {int(r['hour']) for r in rows} if rows else set()
    for h in [int(x) for x in a.hours.split(',')]:
        if h in done_h:
            continue
        PD = float(dem.loc[dem.Hour == h, 'PD'].iloc[0])
        for w in (0.25, 0.5, 0.75):
            for cut in (0.025, 0.05):
                cache = {}
                for cname, over in configs.items():
                    c = dict(RB.BASE, **over); B, R, rho = c['B'], c['R'], c['rho']
                    # the instance of the robustness study for this configuration (same generator, rng and scale)
                    s = RB.Sys(RB.D10, c['k']); case = RB.repo_case(s)
                    key = (c['k'], c['inc'])
                    if key not in cache:
                        cache[key] = RB.build(s, h, PD, w, cut, c['inc'], np.random.default_rng(10000 * h + int(100 * w) + int(1000 * cut)))
                    inst = cache[key]
                    if inst is None:
                        continue
                    Pinc, Finc, Ft = inst['Pinc'], inst['Finc'], inst['Ft']
                    ih = hashlib.sha1(np.round(np.concatenate([Pinc, Ft]), 6).tobytes()).hexdigest()[:12]   # identifies repeated instances
                    if c['scale'] == 'target':
                        sc = np.maximum(np.abs(Ft - Finc), 1e-6 * np.abs(Finc))
                    else:
                        sc = RB.cloud_scale(s, PD, np.random.default_rng(h))
                    e0 = float(np.linalg.norm((Finc - Ft) / sc))
                    Gfun = lambda P, s=s, Ft=Ft, sc=sc, e0=e0: float(np.linalg.norm((s.F(P) - Ft) / sc) / e0)
                    cma_seeds = [5, 11, 17] if c['mseed'] == 5 else [c['mseed'], c['mseed'] + 6, c['mseed'] + 12]
                    lo, hi = RB.AT.band(s, Pinc, rho)
                    feas = lambda P: (abs(P.sum() - PD) < 1e-4 and np.all(P >= lo - 1e-6) and np.all(P <= hi + 1e-6)
                                      and np.abs(P - Pinc).sum() <= R + 1e-5)
                    cons = [{'type': 'eq', 'fun': lambda P: P.sum() - PD},
                            {'type': 'ineq', 'fun': lambda P: R - np.abs(P - Pinc).sum()}]
                    for m, seed in [('COBYQA', 0), ('SLSQP_FD', 0)] + [('CMAES', q) for q in cma_seeds]:
                        st = {'n': 0, 'best': 1.0}; t = time.time()
                        try:
                            if m in ('COBYQA', 'SLSQP_FD'):
                                def f(P):
                                    if st['n'] >= B: raise Stop
                                    st['n'] += 1; v = Gfun(P)
                                    if feas(P) and v < st['best']: st['best'] = v
                                    return v
                                if m == 'COBYQA':
                                    minimize(f, Pinc, method='COBYQA', bounds=list(zip(lo, hi)), constraints=cons,
                                             options={'maxfev': B + 10})
                                else:
                                    minimize(f, Pinc, method='SLSQP', bounds=list(zip(lo, hi)), constraints=cons,
                                             options={'maxiter': 10000, 'ftol': 1e-14})
                            else:
                                proj = BoundedProjector(case=case, incumbent=Pinc, demand=PD, redispatch_cap_mw=R,
                                                        rho=rho, feasibility_tol=1e-6)
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
                        rows.append(dict(config=cname, hour=h, PD=PD, w=w, cut=cut, method=m, mseed=seed, G=st['best'], inst_hash=ih,
                                         G_fallback=min(st['best'], 1.0), evals=st['n'], runtime=time.time() - t, B=B, R=R, rho=rho))
        pd.DataFrame(rows).to_csv(f'{a.out}/raw_{a.tag}.csv', index=False)
        print(a.tag, h, round(time.time() - t0), flush=True)
    print(a.tag, 'DONE', flush=True)


if __name__ == '__main__':
    main()
