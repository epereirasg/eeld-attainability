"""Evaluation-limited regime: F as a black-box oracle with very small budgets and noise (RQ5c, Table 11).

All methods see F(P) = [C(P), E(P)] only through evaluations; none receives an analytic gradient.
The repaired procedures (POLY, ELLP, CGM) build their own surrogate directions, GS-R is the strongest
counterfactual generator of the main comparison, COBYQA is model-based derivative-free, SLSQP-FD uses
finite-difference gradients (each difference costs evaluations), and CMA-ES runs with three seeds.
Instances are the 25 instances of the default configuration of the method comparison (efficient
incumbent, R = 250 MW, rho = 0.20, target-linked scale), regenerated with the same seeds.

Regimes: budgets of 25, 50 and 100 calls without noise, and budgets of 100 and 500 calls with additive
Gaussian noise of standard deviation sigma * s on each objective (s is the target-linked scale, so
sigma = 0.05 is 5% of the change the request asks of each objective). Methods choose their returned
dispatch from the values they observe; the returned dispatch is then scored with the exact F.
The certified reference G_ref is the bounded multistart optimum of experiments/robustness.
Output: results/raw/supplementary/oracle_<tag>.csv (the stored run uses the tag a)
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from eeldattain.paths import RAW_SUPP
from eeldattain.runio import guard_config, append_rows
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robustness'))
import run as RB                      # instance generator and legacy procedures, identical seeds
import cma
from cf_eeld.repair import BoundedProjector
from cf_eeld.config import ExperimentConfig, SolverConfig

REGIMES = [(25, 0.0), (50, 0.0), (100, 0.0), (100, 0.05), (500, 0.05)]
LEGACY = ['POLY', 'ELLP', 'CGM', 'GS']
DFO = [('COBYQA', 0), ('SLSQP_FD', 0), ('CMAES', 5), ('CMAES', 11), ('CMAES', 17)]


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--hours', default='1,4,7,10,13,16,19,22')
    ap.add_argument('--out', default=str(RAW_SUPP))
    ap.add_argument('--tag', default='a')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    guard_config(a.out, f'oracle_{a.tag}', dict(hours=a.hours, regimes=REGIMES, legacy=LEGACY, dfo=DFO, base=RB.BASE))
    out = f'{a.out}/oracle_{a.tag}.csv'
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out); done = set(zip(d.hour, d.w, d.cut, d.B, d.sigma, d.method, d.mseed))
    dem = RB.load_demand(f'{RB.REPO}/data/pd_profile_10units.csv')
    R, rho = RB.BASE['R'], RB.BASE['rho']
    for h in [int(x) for x in a.hours.split(',')]:
        PD = float(dem.loc[dem.Hour == h, 'PD'].iloc[0])
        for w in (0.25, 0.5, 0.75):
            for cut in (0.025, 0.05):
                s = RB.Sys(RB.D10, 1.0); case = RB.repo_case(s)
                inst = RB.build(s, h, PD, w, cut, 'efficient', np.random.default_rng(10000 * h + int(100 * w) + int(1000 * cut)))
                if inst is None:
                    continue
                Pinc, Finc, Ft = inst['Pinc'], inst['Finc'], inst['Ft']
                sc = np.maximum(np.abs(Ft - Finc), 1e-6 * np.abs(Finc)); e0 = float(np.linalg.norm((Finc - Ft) / sc))
                Gtrue = lambda P: float(np.linalg.norm((s.F(P) - Ft) / sc) / e0)
                RB.AT.POOL.clear(); RB.AT.POOL.extend([inst['Pfr'], inst['Pe'], inst['Pc']])
                Gref, _ = RB.AT.G_bounded(s, Pinc, PD, Ft, sc, e0, R, rho, np.random.default_rng(1), k=4)
                lo, hi = RB.AT.band(s, Pinc, rho)
                feas = lambda P: (abs(P.sum() - PD) < 1e-4 and np.all(P >= lo - 1e-6) and np.all(P <= hi + 1e-6)
                                  and np.abs(P - Pinc).sum() <= R + 1e-5)
                true_cost = case.cost
                for B, sigma in REGIMES:
                    for m, mseed in [(x, 0) for x in LEGACY] + DFO:
                        if (h, w, cut, B, sigma, m, mseed) in done:
                            continue
                        nrng = np.random.default_rng(900_000 + 1000 * h + int(100 * w) + int(1000 * cut) + B + 7 * mseed)
                        noisy_F = lambda P: s.F(P) + (sigma * sc * nrng.standard_normal(2) if sigma > 0 else 0.0)
                        t = time.time(); err = ''; P_ret = Pinc.copy(); n_ev = 0
                        try:
                            if m in LEGACY:
                                case.F_full = lambda P: noisy_F(np.asarray(P, float))
                                cfg = ExperimentConfig(runs=1, solver=SolverConfig(max_true_evals=B),
                                                       baselines=RB.RR._baseline_config(B, 'scaled'))
                                res, _, P, _ = RB.RR.run_one(case, PD, Pinc, Ft, sc, m, 2, 2, 0.0, cfg,
                                                             np.random.default_rng(RB.BASE['mseed']), R, rho)
                                P_ret = np.asarray(P, float); n_ev = int(res.diagnostics.true_evals)
                            else:
                                st = {'n': 0, 'best_obs': np.inf, 'P': Pinc.copy()}

                                def observe(P):
                                    if st['n'] >= B: raise Stop
                                    st['n'] += 1
                                    v = float(np.linalg.norm((noisy_F(P) - Ft) / sc) / e0)
                                    if feas(P) and v < st['best_obs']:
                                        st['best_obs'] = v; st['P'] = np.asarray(P, float).copy()
                                    return v
                                try:
                                    if m == 'COBYQA':
                                        minimize(observe, Pinc, method='COBYQA', bounds=list(zip(lo, hi)),
                                                 constraints=[{'type': 'eq', 'fun': lambda P: P.sum() - PD},
                                                              {'type': 'ineq', 'fun': lambda P: R - np.abs(P - Pinc).sum()}],
                                                 options={'maxfev': B + 10})
                                    elif m == 'SLSQP_FD':
                                        minimize(observe, Pinc, method='SLSQP', bounds=list(zip(lo, hi)),
                                                 constraints=[{'type': 'eq', 'fun': lambda P: P.sum() - PD},
                                                              {'type': 'ineq', 'fun': lambda P: R - np.abs(P - Pinc).sum()}],
                                                 options={'maxiter': 10000, 'ftol': 1e-14})
                                    else:
                                        proj = BoundedProjector(case=case, incumbent=Pinc, demand=PD, redispatch_cap_mw=R,
                                                                rho=rho, feasibility_tol=1e-6)
                                        z0 = case.encode(Pinc); zl, zh = lo[case.indep_idx], hi[case.indep_idx]
                                        def fz(z):
                                            P, _ = proj.repair_full(np.asarray(z), previous_z_or_p=Pinc, context='cma', log_event=False)
                                            return observe(P)
                                        es = cma.CMAEvolutionStrategy(z0, 0.1 * float(np.mean(zh - zl)),
                                                                      {'bounds': [list(zl), list(zh)], 'verbose': -9, 'seed': mseed, 'maxfevals': B})
                                        while not es.stop():
                                            X = es.ask(); es.tell(X, [fz(x) for x in X])
                                except Stop:
                                    pass
                                P_ret, n_ev = st['P'], st['n']
                        except Exception as ex:          # recorded, never silently replaced
                            err = f'{type(ex).__name__}: {str(ex)[:100]}'
                        finally:
                            case.F_full = lambda P: np.array([true_cost(P), s.E(np.asarray(P, float))])
                        ok = feas(P_ret)
                        G = Gtrue(P_ret) if (ok and not err) else np.nan
                        append_rows(out, [dict(hour=h, PD=PD, w=w, cut=cut, B=B, sigma=sigma, method=m, mseed=mseed,
                                               G=G, G_fallback=min(G, 1.0) if G == G else 1.0, G_ref=Gref,
                                               shortfall=(min(G, 1.0) if G == G else 1.0) - Gref, feasible=ok,
                                               evals=n_ev, runtime=time.time() - t, err=err)])
                print(f'[oracle] h={h} w={w} cut={cut} done', flush=True)
    print('DONE', time.strftime('%Y-%m-%dT%H:%M:%S'))


if __name__ == '__main__':
    main()
