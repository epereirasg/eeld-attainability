"""Global certification of every stored cell of the 30-bus network campaign with SCIP.

Cost and emission are smooth (no valve-point term) and the DC line limits are linear, so SCIP closes each cell.
For each distinct request of each network it solves the physical cell, the reference allowance and every grid
cell, and compares the global lower bound with the stored multistart value.
Output: results/raw/additional/network/scip_{dc,copper}.csv
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import BIG
from scip_certificate import scip_gap
from network30 import MULT
from eeldattain.network import NetSystem
from eeldattain.paths import RAW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', required=True, choices=('dc', 'copper'))
    ap.add_argument('--time-limit', type=float, default=60)
    a = ap.parse_args()
    d = RAW / 'additional' / 'network' / a.network; tag = f'n30{a.network}'
    Rq = pd.DataFrame([json.loads(l) for l in open(d / f'requests_{tag}.jsonl')])
    Rq = Rq[Rq.duplicate_of_w.isna()]
    G = pd.read_csv(d / f'grid_{tag}.csv'); G = G[G.kind == 'grid']
    V = {(int(v['hour']), float(v['w']), float(v['cut'])): v for v in map(json.loads, open(d / f'vectors_{tag}.jsonl'))}
    out = d.parent / f'scip_{a.network}.csv'
    rows = pd.read_csv(out).to_dict('records') if out.exists() else []
    done = {(r['hour'], r['w'], r['cut'], r['R'], r['rho']) for r in rows}
    systems = {}
    for r in Rq.itertuples():
        h = int(r.hour); key = (h, float(r.w), float(r.cut))
        if h not in systems:
            systems[h] = NetSystem(MULT[h], limits=(a.network == 'dc'))
        s = systems[h]; ln = (s.net.Hg, s.net.f0, s.net.rate) if s.limits else None
        cells = [(np.inf, 1.0, r.G_phys, 'physical'), (r.R0, r.rho0, r.G_ref, 'reference')]
        g = G[(G.hour == h) & (G.w == r.w) & (G.cut == r.cut)]
        cells += [(float(x.R), float(x.rho), float(x.G), 'grid') for x in g.itertuples()]
        for R, rho, Gst, kind in cells:
            if (h, key[1], key[2], R, rho) in done:
                continue
            res = scip_gap(s, V[key], BIG if not np.isfinite(R) else R, rho, time_limit=a.time_limit, lines=ln)
            rows.append(dict(hour=h, w=key[1], cut=key[2], R=R, rho=rho, kind=kind, G_stored=Gst, **res))
            done.add((h, key[1], key[2], R, rho))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(key, len(rows), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
