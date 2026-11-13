"""Robustness study: repaired search procedures vs EELD-adapted counterfactual baselines.

Factors varied one at a time around a default configuration (efficient incumbent, standard emission,
target-linked scaling, R_max = 250 MW, rho = 0.20, 500 evaluations, method seed 5).
Every method uses the pure target loss (alpha = 0, p = q = 2) so all methods pursue the same objective.
The exact bounded optimum G_ref is computed for each instance and allowance as the attainable reference.
"""
import hashlib
import sys
import time
import argparse
import os
import numpy as np, pandas as pd
from eeldattain.paths import RAW_ROBUST, legacy_repo
from eeldattain.runio import guard_config
REPO = str(legacy_repo())          # earlier eeld-cf code base at commit 07e983a (scripts/fetch_legacy_repo.sh)
sys.path[:0] = [f'{REPO}/src', f'{REPO}/scripts']
from pilotcore.core import Sys, D10, nlp
from pilotcore import attain as AT
import run_eeld_targeted as RR
from cf_eeld.eeld import EELD10, load_coeffs, load_demand
from cf_eeld.config import ExperimentConfig, SolverConfig

METHODS = ['POLY', 'ELLP', 'CGM', 'GS', 'FACE', 'NICE_R', 'DAACE_R', 'DiCE']
CONFIGS = {
    'default':        dict(),
    'budget200':      dict(B=200),
    'budget2000':     dict(B=2000),
    'allow100_010':   dict(R=100.0, rho=0.10),
    'allow50_005':    dict(R=50.0, rho=0.05),
    'scale_cloud':    dict(scale='cloud'),
    'emission_repo':  dict(k=0.01),
    'inc_random':     dict(inc='random'),
    'method_seed2':   dict(mseed=11),
}
BASE = dict(B=500, R=250.0, rho=0.20, scale='target', k=1.0, inc='efficient', mseed=5)

def repo_case(s):
    case = EELD10(load_coeffs(f'{REPO}/data/eeld10_coeffs_from_paper_verified.csv'), slack_idx=9)
    case.emission = lambda P: s.E(np.asarray(P, float))   # same emission convention as the pilot system
    return case

def build(s, h, PD, w, cut, inc, rng):
    x0 = [s.rand_feas(PD, rng) for _ in range(100)]
    Pc = nlp(s, PD, s.C, s.gC, x0)[1]; Pe = nlp(s, PD, s.E, s.gE, x0)[1]
    ideal = np.array([s.C(Pc), s.E(Pe)]); nadir = np.array([s.C(Pe), s.E(Pc)])
    Pinc = AT.efficient_incumbent(s, PD, w, ideal, nadir, rng, 100) if inc == 'efficient' else s.rand_feas(PD, rng)
    Finc = s.F(Pinc); Et = (1 - cut) * Finc[1]
    if Et < ideal[1]: return None
    Pfr = AT.eps_front(s, PD, Et, rng, 30, x0s=(Pinc, Pe))
    if Pfr is None: return None
    return dict(Pinc=Pinc, Finc=Finc, Ft=s.F(Pfr), Pfr=Pfr, Pe=Pe, Pc=Pc,
                dominated=bool(s.C(Pfr) < Finc[0] - 1e-9))

def cloud_scale(s, PD, rng, n=500):
    Y = np.array([s.F(s.rand_feas(PD, rng)) for _ in range(n)])
    return np.quantile(Y, 0.9, axis=0) - np.quantile(Y, 0.1, axis=0)

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--hours', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default=str(RAW_ROBUST), help='output directory (default: results/raw/robustness)')
    ap.add_argument('--configs', default=','.join(CONFIGS), help='comma-separated subset of CONFIGS')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    configs = {k: CONFIGS[k] for k in a.configs.split(',')}
    guard_config(a.out, a.tag, dict(hours=a.hours, configs=configs, base=BASE, methods=METHODS))
    hours = [int(x) for x in a.hours.split(',')]
    dem = load_demand(f'{REPO}/data/pd_profile_10units.csv')
    rows, t0 = [], time.time()
    fout = f'{a.out}/raw_{a.tag}.csv'
    if os.path.exists(fout):                       # resume after an interruption (complete blocks only)
        rows = pd.read_csv(fout).to_dict('records')
    done = {(int(r['hour']), float(r['w'])) for r in rows} if rows else set()
    done_h = {int(r['hour']) for r in rows} if rows else set()
    for h in hours:
        PD = float(dem.loc[dem.Hour == h, 'PD'].iloc[0])
        for w in (0.25, 0.5, 0.75):
            if (h, w) in done:
                continue
            for cut in (0.025, 0.05):
                cache = {}
                for cname, over in configs.items():
                    c = dict(BASE, **over)
                    s = Sys(D10, c['k']); case = repo_case(s)
                    key = (c['k'], c['inc'])
                    if key not in cache:
                        cache[key] = build(s, h, PD, w, cut, c['inc'], np.random.default_rng(10000 * h + int(100 * w) + int(1000 * cut)))
                    inst = cache[key]
                    if inst is None:
                        continue
                    Pinc, Finc, Ft = inst['Pinc'], inst['Finc'], inst['Ft']
                    ih = hashlib.sha1(np.round(np.concatenate([Pinc, Ft]), 6).tobytes()).hexdigest()[:12]   # identifies repeated instances
                    if c['scale'] == 'target':
                        sc = np.maximum(np.abs(Ft - Finc), 1e-6 * np.abs(Finc))
                    else:
                        sc = cloud_scale(s, PD, np.random.default_rng(h))
                    e0 = float(np.linalg.norm((Finc - Ft) / sc))
                    Gfun = lambda P: float(np.linalg.norm((s.F(P) - Ft) / sc) / e0)
                    AT.POOL.clear(); AT.POOL.extend([inst['Pfr'], inst['Pe'], inst['Pc']])
                    Gref, _ = AT.G_bounded(s, Pinc, PD, Ft, sc, e0, c['R'], c['rho'], np.random.default_rng(1), k=4)
                    cfg = ExperimentConfig(runs=1, solver=SolverConfig(max_true_evals=c['B']),
                                           baselines=RR._baseline_config(c['B'], 'scaled'))
                    for m in METHODS:
                        t = time.time()
                        try:
                            res, _, P, _ = RR.run_one(case, PD, Pinc, Ft, sc, m, 2, 2, 0.0, cfg,
                                                      np.random.default_rng(c['mseed']), c['R'], c['rho'])
                            G = Gfun(P); ev = int(res.diagnostics.true_evals); err = ''
                            mw = float(np.abs(P - Pinc).sum())
                        except Exception as ex:
                            G, ev, err, mw = np.nan, -1, str(ex)[:80], np.nan
                        rows.append(dict(config=cname, hour=h, PD=PD, w=w, cut=cut, method=m, G=G, inst_hash=ih,
                                         G_fallback=min(G, 1.0) if G == G else np.nan, G_ref=Gref,
                                         dominated_target=inst['dominated'], evals=ev, mw=mw,
                                         runtime=time.time() - t, err=err, **{k: c[k] for k in BASE}))
            pd.DataFrame(rows).to_csv(f'{a.out}/raw_{a.tag}.csv', index=False)
        print(a.tag, h, round(time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
