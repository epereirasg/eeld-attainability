"""Ablations of the method comparison: tuned generator settings and the components of the reference solver.

Instances are those of the robustness study (experiments/robustness/run.py, same generator and seeds), so every
run pairs with the stored runs.

Part 1, generator settings. Each adapted generator runs under its stored setting and under alternatives that move
it toward the pure target loss or widen its search (500 calls, method seed 5):
  GS-R     initial radius 0.05, radius growth 1.5, 50 samples per shell with 10 shells
  FACE-R   no density and no path penalty (best target point of the sample), 20 neighbors
  NICE-R   target weight 1.0, pool of 500 dispatches
  DAACE-R  no path penalty
  DiCE-R   no proximity and no diversity term
Per instance, the best value over all settings of a generator (and over all generators) is an oracle-tuned
generator, which favors the generators.

Part 2, reference solver. The bounded multistart SLSQP of the robustness study is re-run
  full     incumbent, 4 random starts, target dispatch, least-emission and least-cost anchors (stored setting)
  notar    as full without the target dispatch
  noanch   incumbent and 4 random starts only
  inc      incumbent only
Output: results/raw/additional/ablation_generators.csv and ablation_reference.csv
"""
import argparse, dataclasses, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from eeldattain.paths import RAW
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robustness'))
import run as RB
from cf_eeld.config import ExperimentConfig, SolverConfig

SETTINGS = {
    'GS': {'stored': {}, 'r05': dict(gs_initial_radius=0.05), 'g15': dict(gs_radius_growth=1.5),
           's50': dict(gs_samples_per_shell=50, gs_max_shells=10)},
    'FACE': {'stored': {}, 'pure': dict(face_lambda_density=0.0, face_lambda_path=0.0), 'k20': dict(face_k_neighbors=20)},
    'NICE_R': {'stored': {}, 'lt1': dict(nice_lambda_target=1.0), 'pool500': dict(nice_pool_size=500)},
    'DAACE_R': {'stored': {}, 'nopath': dict(daace_lambda_path=0.0)},
    'DiCE': {'stored': {}, 'pure': dict(dice_lambda_prox=0.0, dice_lambda_div=0.0)},   # 50 candidates took 216 s per run
}
GEN_CFGS = {'default': dict(), 'allow50_005': dict(R=50.0, rho=0.05)}
REF_CFGS = {'default': dict(), 'allow100_010': dict(R=100.0, rho=0.10), 'allow50_005': dict(R=50.0, rho=0.05)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--hours', default='1,4,7,10,13,16,19,22')
    ap.add_argument('--part', default='both', choices=['gen', 'ref', 'both'])
    ap.add_argument('--out', default=str(RAW / 'additional'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dem = RB.load_demand(f'{RB.REPO}/data/pd_profile_10units.csv')
    G_rows, R_rows, t0 = [], [], time.time()
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
                Gfun = lambda P: float(np.linalg.norm((s.F(P) - Ft) / sc) / e0)
                key = dict(hour=h, PD=PD, w=w, cut=cut)
                if a.part in ('ref', 'both'):
                    for cname, over in REF_CFGS.items():
                        c = dict(RB.BASE, **over)
                        for var, pool, k in (('full', [inst['Pfr'], inst['Pe'], inst['Pc']], 4), ('notar', [inst['Pe'], inst['Pc']], 4),
                                             ('noanch', [], 4), ('inc', [], 0)):
                            RB.AT.POOL.clear(); RB.AT.POOL.extend(pool); t = time.time()
                            G, _ = RB.AT.G_bounded(s, Pinc, PD, Ft, sc, e0, c['R'], c['rho'], np.random.default_rng(1), k=k)
                            R_rows.append(dict(key, config=cname, variant=var, G=float(G), runtime=time.time() - t))
                if a.part in ('gen', 'both'):
                    for cname, over in GEN_CFGS.items():
                        c = dict(RB.BASE, **over)
                        base = RB.RR._baseline_config(c['B'], 'scaled')
                        for m, sets in SETTINGS.items():
                            for sname, fields in sets.items():
                                cfg = ExperimentConfig(runs=1, solver=SolverConfig(max_true_evals=c['B']),
                                                       baselines=dataclasses.replace(base, **fields))
                                t = time.time()
                                try:
                                    _, _, P, _ = RB.RR.run_one(case, PD, Pinc, Ft, sc, m, 2, 2, 0.0, cfg,
                                                               np.random.default_rng(c['mseed']), c['R'], c['rho'])
                                    G = Gfun(P)
                                except Exception as ex:
                                    G = np.nan
                                G_rows.append(dict(key, config=cname, method=m, setting=sname, G=G,
                                                   G_fallback=min(G, 1.0) if G == G else np.nan, runtime=time.time() - t))
                if R_rows: pd.DataFrame(R_rows).to_csv(f'{a.out}/ablation_reference.csv', index=False)
                if G_rows: pd.DataFrame(G_rows).to_csv(f'{a.out}/ablation_generators.csv', index=False)
        print('hour', h, round(time.time() - t0), flush=True)
    print('DONE', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
