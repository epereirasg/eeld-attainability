"""Exhaustive unit counterfactual over all units on the 10-unit system.

For every distinct stored request it re-solves G*(inf, rho_0) and G*(inf, rho_S) for every set S of at most
three units drawn from all ten units (up to 175 sets), not only from the band-active units. Larger sets
are skipped once a smaller set closes the gap at the strictest tolerance analysed (eps = 0.02). Any tolerance eps can
then be applied afterwards, which gives the exact size class |S*| <= 3, the number of distinct minimal sets
(diversity) and the tolerance sensitivity, without a further solve.
"""
import argparse, itertools, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *

RHO0 = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    a = ap.parse_args()
    out = RAW_ADD / 'unit_sets'; out.mkdir(parents=True, exist_ok=True)
    s = System(str(COEFFS['10']))
    Rq, vk = load_requests()
    f = out / f'sets_p{a.part}.csv'
    rows = pd.read_csv(f).to_dict('records') if f.exists() else []   # resume after an interruption
    done = {(r['hour'], r['w'], r['cut']) for r in rows}
    for i, row in Rq.iterrows():
        if i % a.parts != a.part or (row.hour, row.w, row.cut) in done:
            continue
        rng = np.random.default_rng(20261128 + i)
        req = make_request(s, vk[row.key], pool_cap=6); T = Timer()
        Gp = req.gap(BIG, 1.0, rng)[0]
        GRf, PRf = req.gap(BIG, RHO0, rng)
        lo, hi = req.band(RHO0)
        pins = [j for j in range(s.n) if min(PRf[j] - lo[j], hi[j] - PRf[j]) < 1e-3
                and (hi[j] < s.hi[j] - 1e-6 or lo[j] > s.lo[j] + 1e-6)]
        base = dict(hour=row.hour, w=row.w, cut=row.cut, G_phys=Gp, G_Rfree=GRf, G_phys_stored=row.G_phys,
                    G_ref_stored=row.G_ref, pins='-'.join(str(j + 1) for j in pins), t_base=T.lap())
        thr_min = min(Gp, row.G_phys) + 0.02       # strictest tolerance analysed (eps = 0.02)
        for k in (1, 2, 3):
            if any(r['G'] <= thr_min for r in rows[-200:] if r.get('_req') == i and r['k'] < k):
                break                                # a smaller set already closes at every tolerance analysed
            for S in itertools.combinations(range(s.n), k):
                rv = np.full(s.n, RHO0); rv[list(S)] = 1.0
                T.lap(); G = req.gap(BIG, rv, rng)[0]
                rows.append(dict(base, _req=i, k=k, S='-'.join(str(j + 1) for j in S),
                                 band_active=all(j in pins for j in S), G=G, t=T.lap()))
        print(i, row.hour, row.w, row.cut, round(time.time()), flush=True)
        pd.DataFrame(rows).to_csv(f, index=False)
    pd.DataFrame(rows).to_csv(out / f'sets_p{a.part}.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
