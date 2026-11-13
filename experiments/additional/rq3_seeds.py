"""Seed reproducibility of the allowance counterfactuals (RQ3 clause).

For every distinct 10-unit request that needs a counterfactual (G_ref > G_phys + 0.05), the physical gap, rho*
and R* are re-solved from the stored request vectors with a new random stream (seed offset 777) and the same
bisection settings as the campaign (8 steps, rho in [0.05, 1], R in [0, 1000] MW). The rule asks
that the differences stay within the bisection resolution, 0.004 for rho and 8 MW for R.
Output: results/raw/additional/rq3_seeds.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'attainability'))
from run import CFG, bisect

OUT = RAW_ADD / 'rq3_seeds.csv'
RAW_ADD.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    a = ap.parse_args()
    cfg = CFG['10']; R0, r0 = cfg['ref']
    s = System(str(COEFFS['10']))
    Rq, vk = load_requests('10')
    need = Rq[Rq.G_ref > Rq.G_phys + EPS].reset_index(drop=True)
    out = OUT.with_name(f'rq3_seeds_p{a.part}.csv')
    prev = sorted(RAW_ADD.glob('rq3_seeds_p*.csv'))           # resume across any earlier split into parts
    done = set(pd.concat([pd.read_csv(f) for f in prev]).key.astype(str)) if prev else set()
    for j, r in need.iterrows():
        if j % a.parts != a.part or str(r.key) in done:
            continue
        req = make_request(s, vk[r.key], pool_cap=15)
        rng = np.random.default_rng(777 + 1000 * r.key[0] + int(1000 * r.key[1]) + int(10000 * r.key[2]))
        t = Timer()
        Gp, _ = req.gap(BIG, 1.0, rng, k=cfg['kb'])
        thr = Gp + cfg['eps']
        rho = bisect(lambda x: req.gap(BIG, x, rng, k=cfg['kb'])[0], r0, 1.0, thr, cfg['bis'])
        Rs = bisect(lambda x: req.gap(x, 1.0, rng, k=cfg['kb'])[0], 0.0, cfg['R_hi'], thr, cfg['bis'])
        row = dict(key=str(r.key), hour=r.key[0], w=r.key[1], cut=r.key[2], G_phys=r.G_phys, G_phys_new=Gp,
                   rho_star=r.rho_star, rho_star_new=rho, R_star=r.R_star, R_star_new=Rs, time_s=t.lap())
        pd.DataFrame([row]).to_csv(out, mode='a', header=not out.exists(), index=False)
        print(j, r.key, round(r.rho_star, 4), round(rho, 4), round(r.R_star, 1), round(Rs, 1), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
