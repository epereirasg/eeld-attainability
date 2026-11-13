"""Global certification of every stored 10-unit cell with SCIP.

Cells: the physical set, the reference, budget-free, band-free and alternative allowances, the 20 grid cells
and the 7 budget-only frontier cells of every distinct request (main campaign, results/raw/attainability).
For each cell it records SCIP's certified lower bound G_lb and the primal value G_ub of G*(R, rho). The
campaign value is certified when G_stored - G_lb <= 0.01, the tolerance of the empirical certificate.
"""
import argparse
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from scip_certificate import scip_gap


def cells_of(row, G):
    out = [('physical', BIG, 1.0, row.G_phys), ('reference', 50.0, 0.05, row.G_ref),
           ('budget_free', BIG, 0.05, row.G_Rfree), ('band_free', 50.0, 1.0, row.G_rhofree),
           ('alternative', 100.0, 0.10, row.G_alt)]
    g = G[(G.hour == row.hour) & (G.w == row.w) & (G.cut == row.cut)]
    out += [(k, float(R), float(rho), float(v)) for k, R, rho, v in zip(g.kind, g.R, g.rho, g.G)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--time-limit', type=float, default=10)
    a = ap.parse_args()
    out = RAW_ADD / 'scip'; out.mkdir(parents=True, exist_ok=True)
    s = System(str(COEFFS['10']))
    Rq, vk = load_requests()
    G = pd.concat([pd.read_csv(RAW_ATTAIN / f'grid_e10{t}.csv') for t in 'ab'])
    rows, done = [], set()
    f = out / f'cells10_p{a.part}.csv'
    if f.exists():                            # resume: keep requests with all their cells stored
        old = pd.read_csv(f)
        cnt = old.groupby(['hour', 'w', 'cut']).size()
        done = {(int(h), float(w), float(c)) for (h, w, c), n in cnt.items() if n >= 32}
        rows = [r for r in old.to_dict('records') if (int(r['hour']), float(r['w']), float(r['cut'])) in done]
    for i, row in Rq.iterrows():
        if i % a.parts != a.part or row.key in done:
            continue
        for kind, R, rho, Gst in cells_of(row, G):
            r = scip_gap(s, vk[row.key], R, rho, time_limit=a.time_limit)
            rows.append(dict(hour=row.hour, w=row.w, cut=row.cut, kind=kind, R=R if R < BIG else np.inf, rho=rho,
                             G_stored=Gst, **r))
        print(i, row.hour, row.w, row.cut, round(sum(x['time_s'] for x in rows[-32:]), 1), flush=True)
        if i % 8 in (0, 1):
            pd.DataFrame(rows).to_csv(out / f'cells10_p{a.part}.csv', index=False)
    pd.DataFrame(rows).to_csv(out / f'cells10_p{a.part}.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
