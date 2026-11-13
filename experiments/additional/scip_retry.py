"""Re-solve with a longer time limit every stored SCIP cell that stopped at the time
limit of scip_all_cells.py with a bound more than 0.01 below the stored value, so that its global lower bound closes. Output: results/raw/additional/scip/retry.csv."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from scip_certificate import scip_gap


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--time-limit', type=float, default=900)
    a = ap.parse_args()
    out = RAW_ADD / 'scip'
    D = pd.concat([pd.read_csv(f) for f in sorted(out.glob('cells10_p*.csv'))])
    T = D[(D.status == 'timelimit') & (D.G_stored - D.G_lb > 0.01)]   # cells whose 10 s bound does not yet certify
    s = System(str(COEFFS['10'])); _, vk = load_requests()
    f = out / 'retry.csv'
    rows = pd.read_csv(f).to_dict('records') if f.exists() else []
    done = {(r['hour'], r['w'], r['cut'], r['kind'], r['R'], r['rho']) for r in rows}
    for r in T.itertuples():
        k = (r.hour, r.w, r.cut, r.kind, r.R, r.rho)
        if k in done:
            continue
        R = BIG if not np.isfinite(r.R) else float(r.R)
        res = scip_gap(s, vk[(int(r.hour), float(r.w), float(r.cut))], R, r.rho, time_limit=a.time_limit)
        rows.append(dict(hour=r.hour, w=r.w, cut=r.cut, kind=r.kind, R=r.R, rho=r.rho, G_stored=r.G_stored, **res))
        print(k, round(r.G_stored, 4), round(float(res['G_lb']), 4), res['status'], round(res['time_s'], 0), flush=True)
        pd.DataFrame(rows).to_csv(f, index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
