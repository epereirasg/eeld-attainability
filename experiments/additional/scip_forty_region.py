"""Branch and bound on the study-region cells of the primary 40-unit campaign.

For every distinct request it solves G*(R, rho) for R in {100, 250, 500} MW and rho in {0.05, 0.10, 0.20} with SCIP
and records the global lower bound, the SCIP primal value and the stored campaign value. A cell is certified when the
stored value exceeds the lower bound by at most 0.01. A cell with a stored value of at most 0.01 needs no solve,
because G* >= 0 gives that bound. Output: results/raw/additional/scip/forty_region_p<part>.csv
"""
import argparse, glob, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from scip_certificate import scip_gap


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--time-limit', type=float, default=120.0)
    a = ap.parse_args()
    from eeldattain.runio import campaign_escale
    s = System(str(COEFFS['40']), escale=campaign_escale('40'))   # convention recorded with the stored campaign
    Rq, vk = load_requests('40')
    G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW_ATTAIN / 'grid_e40*.csv')))])
    G = G[(G.kind == 'grid') & G.R.isin([100, 250, 500]) & G.rho.isin([0.05, 0.1, 0.2])]
    st = {(int(h), float(w), float(c), float(R), float(r)): g for h, w, c, R, r, g in zip(G.hour, G.w, G.cut, G.R, G.rho, G.G)}
    out = RAW_ADD / 'scip'; out.mkdir(parents=True, exist_ok=True)
    f = out / f'forty_region_p{a.part}.csv'
    rows = pd.read_csv(f).to_dict('records') if f.exists() else []
    done = {(r['hour'], r['w'], r['cut'], r['R'], r['rho']) for r in rows}
    jobs = [(row, R, rho) for _, row in Rq.iterrows() for R in (100.0, 250.0, 500.0) for rho in (0.05, 0.1, 0.2)]
    for j, (row, R, rho) in enumerate(jobs):
        if j % a.parts != a.part or (row.hour, row.w, row.cut, R, rho) in done:
            continue
        Gst = st[(int(row.hour), float(row.w), float(row.cut), R, rho)]
        if Gst <= 0.01:      # G* >= 0, so the bound 0 already settles the cell within 0.01
            r = dict(G_lb=0.0, G_ub=float(Gst), status='zero_lower_bound', time_s=0.0, nodes=0)
        else:
            r = scip_gap(s, vk[row.key], R, rho, time_limit=a.time_limit)
        rows.append(dict(hour=row.hour, PD=row.PD, w=row.w, cut=row.cut, R=R, rho=rho, G_stored=Gst, **r))
        print(row.hour, row.w, row.cut, R, rho, round(Gst, 4), round(float(r['G_lb']), 4), round(float(r['G_ub']), 4),
              r['status'], round(r['time_s']), flush=True)
        pd.DataFrame(rows).to_csv(f, index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
