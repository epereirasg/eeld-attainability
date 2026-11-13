"""Exhaustive pair test for the 40-unit unit counterfactual (RQ11, Table 12).

Experiment X7 classified a request as |S*| >= 3 after testing every single unit and every pair of
band-active units. This script tests all C(40, 2) = 780 pairs, including pairs with a unit that is
off its band limit at the budget-free optimum, on one |S*| >= 3 request per demand level.

The setup reproduces X7 exactly (stored vectors, seed rule, strong re-solve, light warm-started
solves); the script checks that the re-solve and the best single unit match the stored X7 row.
Output: results/raw/supplementary/pairs40.csv (one row per request) and pairs40_all.csv (one row per pair).
"""
import argparse, copy, itertools, json, os, time
import numpy as np, pandas as pd
from eeldattain.model import System, Request
from eeldattain.paths import COEFFS, RAW_ATTAIN, RAW_OPS, RAW_SUPP
from eeldattain.runio import guard_config
from eeldops.experiments import seed_of


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(RAW_SUPP))
    ap.add_argument('--per-level', type=int, default=1, help='|S*|>=3 requests tested per demand level (4 covers all 18)')
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    guard_config(a.out, f'pairs40_p{a.part}', dict(per_level=a.per_level, part=a.part, parts=a.parts, rho0=0.05, eps=0.05))
    from eeldattain.runio import campaign_escale
    s = System(str(COEFFS['40']), escale=campaign_escale('40'))
    V = {}
    for f in sorted(os.listdir(RAW_ATTAIN)):
        if f.startswith('vectors_e40'):
            for line in open(RAW_ATTAIN / f):
                v = json.loads(line); V[(v['PD'], v['w'], v['cut'])] = v
    X7 = pd.read_csv(RAW_OPS / 'sets40_a_p0.csv', dtype={'exact_size': str})
    pick = X7[X7.exact_size == '>=3'].sort_values(['PD', 'w', 'cut']).groupby('PD').head(a.per_level).reset_index(drop=True)
    pick = pick.iloc[a.part::a.parts]
    summary, allpairs = [], []
    fs, fa = f'{a.out}/pairs40_p{a.part}.csv', f'{a.out}/pairs40_all_p{a.part}.csv'
    if os.path.exists(fs):                      # resume: keep the requests already tested
        summary = pd.read_csv(fs).to_dict('records'); allpairs = pd.read_csv(fa).to_dict('records')
    done = {(x['PD'], x['w'], x['cut']) for x in summary}
    for r in pick.itertuples():
        if (r.PD, r.w, r.cut) in done:
            continue
        t0 = time.time(); v = V[(r.PD, r.w, r.cut)]
        sd = seed_of(7, int(v['PD'] / 100), v['w'], v['cut']); rng = np.random.default_rng(sd)
        req = Request(s, v['PD'], np.array(v['Pinc']), np.array(v['Ft']), np.array(v['scale']), np.array(v['Pfr']), None, None, pool_cap=8)
        rho0 = 0.05; thr = r.G_phys + 0.05
        G_rf, P_rf = req.gap(1e9, rho0, rng, k=4)
        lo0, hi0 = req.band(rho0)
        act = [j for j in range(s.n) if min(P_rf[j] - lo0[j], hi0[j] - P_rf[j]) < 1e-3 and (hi0[j] < s.hi[j] - 1e-6 or lo0[j] > s.lo[j] + 1e-6)]
        light = copy.copy(req); light.pool_cap = 2; light.fallbacks = 0

        def G_with(S, warm=()):
            rv = np.full(s.n, rho0); rv[list(S)] = 1.0
            light.pool = [P_rf] + [w_ for w_ in warm if w_ is not None]
            return light.gap(1e9, rv, rng, k=1)

        singles = {j: G_with([j]) for j in range(s.n)}
        single_best = min(g for g, _ in singles.values())
        for i, j in itertools.combinations(range(s.n), 2):
            g, _ = G_with((i, j), warm=[singles[i][1], singles[j][1]])
            allpairs.append(dict(PD=r.PD, w=r.w, cut=r.cut, i=i + 1, j=j + 1, i_active=i in act, j_active=j in act,
                                 G=g, closes=bool(g <= thr)))
        P = pd.DataFrame([x for x in allpairs if (x['PD'], x['w'], x['cut']) == (r.PD, r.w, r.cut)])
        P['kind'] = np.where(P.i_active & P.j_active, 'both_active', np.where(P.i_active | P.j_active, 'one_active', 'none_active'))
        best = P.loc[P.G.idxmin()]
        summary.append(dict(PD=r.PD, w=r.w, cut=r.cut, seed=sd, thr=thr, G_Rfree=G_rf, G_Rfree_X7=r.G_Rfree,
                            single_best=single_best, single_best_X7=r.single_best, n_active=len(act),
                            pairs_tested=len(P), pairs_closing=int(P.closes.sum()),
                            closing_both_active=int(P[P.kind == 'both_active'].closes.sum()),
                            closing_one_active=int(P[P.kind == 'one_active'].closes.sum()),
                            closing_none_active=int(P[P.kind == 'none_active'].closes.sum()),
                            best_pair=f'{best.i}-{best.j}', best_pair_G=best.G, best_pair_kind=best.kind,
                            margin_to_thr=best.G - thr, solver_fallbacks=req.fallbacks + light.fallbacks,
                            runtime_s=time.time() - t0))
        print(f'[pairs40] PD={r.PD} w={r.w} cut={r.cut} closing={summary[-1]["pairs_closing"]} '
              f'best={best.G:.3f} thr={thr:.3f} t={summary[-1]["runtime_s"]:.0f}s', flush=True)
        pd.DataFrame(summary).to_csv(fs, index=False)
        pd.DataFrame(allpairs).to_csv(fa, index=False)
    print('DONE', time.strftime('%Y-%m-%dT%H:%M:%S'))


if __name__ == '__main__':
    main()
