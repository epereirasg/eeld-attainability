"""Branch and bound on the 40-unit reference cells of the primary 40-unit campaign.

For every distinct request it solves the reference cell G*(200 MW, 0.05) and the physical cell G*(inf, 1) with SCIP
(time limit 600 s per cell by default) and records the global lower bound, the SCIP primal value and the stored campaign
value. A cell is certified when the stored value exceeds the lower bound by at most 0.01. A cell with a stored value of at most 0.01 needs no solve,
because G* >= 0 gives that bound. When the SCIP dispatch is
better than the stored one by more than 1e-4, the improvement is recorded (column G_ub) so that the analysis can report it.
Output: results/raw/additional/scip/forty_cells_p<part>.csv
"""
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from scip_certificate import scip_gap


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--time-limit', type=float, default=600.0)
    a = ap.parse_args()
    from eeldattain.runio import campaign_escale
    s = System(str(COEFFS['40']), escale=campaign_escale('40'))   # convention recorded with the stored campaign
    Rq, vk = load_requests('40')
    out = RAW_ADD / 'scip'; out.mkdir(parents=True, exist_ok=True)
    f = out / f'forty_cells_p{a.part}.csv'
    rows = pd.read_csv(f).to_dict('records') if f.exists() else []
    done = {(r['hour'], r['w'], r['cut'], r['kind']) for r in rows}
    for i, row in Rq.iterrows():
        if i % a.parts != a.part:
            continue
        for kind, R, rho, Gst in (('reference', 200.0, 0.05, row.G_ref), ('physical', BIG, 1.0, row.G_phys)):
            if (row.hour, row.w, row.cut, kind) in done:
                continue
            if Gst <= 0.01:      # G* >= 0, so the bound 0 already settles the cell within 0.01
                r = dict(G_lb=0.0, G_ub=float(Gst), status='zero_lower_bound', time_s=0.0, nodes=0)
            else:
                r = scip_gap(s, vk[row.key], R, rho, time_limit=a.time_limit)
            rows.append(dict(hour=row.hour, PD=row.PD, w=row.w, cut=row.cut, kind=kind, G_stored=Gst, **r))
            print(row.hour, row.w, row.cut, kind, round(Gst, 4), round(float(r['G_lb']), 4), round(float(r['G_ub']), 4),
                  r['status'], round(r['time_s']), flush=True)
            pd.DataFrame(rows).to_csv(f, index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
