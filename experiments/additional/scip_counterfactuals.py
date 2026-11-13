"""Global check of the band and budget counterfactuals of the 10-unit campaign.

For every distinct request with a stored rho* (R*) it solves, with SCIP, the gap one bisection step below the
stored value, G*(inf, rho* - 0.95/256) and G*(R* - 1000/256, 1), and the gap at the stored value. The stored
counterfactual is certified minimal when the lower bound below it exceeds G*_phys + eps, and certified valid
when the upper bound at it is at most G*_phys + eps (both up to the 0.01 tolerance of the certificate).
Output: results/raw/additional/scip/counterfactuals.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from scip_certificate import scip_gap

D_RHO, D_R = 0.95 / 256, 1000.0 / 256


def main():
    s = System(str(COEFFS['10'])); Rq, vk = load_requests()
    f = RAW_ADD / 'scip' / 'counterfactuals.csv'
    rows = pd.read_csv(f).to_dict('records') if f.exists() else []
    done = {(r['hour'], r['w'], r['cut'], r['cf']) for r in rows}
    for r in Rq.itertuples():
        thr = r.G_phys + EPS
        for cf, val in (('rho', r.rho_star), ('R', r.R_star)):
            if not np.isfinite(val) or (r.hour, r.w, r.cut, cf) in done:
                continue
            if cf == 'rho':
                below = scip_gap(s, vk[r.key], BIG, max(0.05, val - D_RHO), time_limit=60)
                at = scip_gap(s, vk[r.key], BIG, val, time_limit=60)
            else:
                below = scip_gap(s, vk[r.key], max(0.0, val - D_R), 1.0, time_limit=60)
                at = scip_gap(s, vk[r.key], val, 1.0, time_limit=60)
            rows.append(dict(hour=r.hour, w=r.w, cut=r.cut, cf=cf, value=val, thr=thr,
                             lb_below=below['G_lb'], ub_below=below['G_ub'], st_below=below['status'],
                             lb_at=at['G_lb'], ub_at=at['G_ub'], st_at=at['status']))
            print(r.hour, r.w, r.cut, cf, round(val, 4), round(thr, 4), round(float(below['G_lb']), 4), round(float(at['G_ub']), 4), flush=True)
        pd.DataFrame(rows).to_csv(f, index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
