"""The part of each DC-network gap that the line limits cause.

For every distinct request of the DC campaign (incumbent, target and scale computed with line limits), the
physical gap and the gap at the reference allowance are re-solved on the same units without line limits. The
difference G*(with limits) - G*(without limits) is the part of the gap that the network causes for that request. The relaxed set contains the limited one, so the relaxed value is capped
at the stored value (the raw re-solve is kept as *_raw).
Output: results/raw/additional/network/split_dc.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import BIG, make_request
from network30 import MULT, CFG30
from eeldattain.network import NetSystem
from eeldattain.paths import RAW


def main():
    d = RAW / 'additional' / 'network' / 'dc'
    Rq = pd.DataFrame([json.loads(l) for l in open(d / 'requests_n30dc.jsonl')])
    Rq = Rq[Rq.duplicate_of_w.isna()]
    V = {(int(v['hour']), float(v['w']), float(v['cut'])): v for v in map(json.loads, open(d / 'vectors_n30dc.jsonl'))}
    rows = []
    for r in Rq.itertuples():
        key = (int(r.hour), float(r.w), float(r.cut))
        free = NetSystem(MULT[key[0]], limits=False); lim = NetSystem(MULT[key[0]], limits=True)
        rng = np.random.default_rng(4242 + 100 * key[0] + int(100 * key[1]) + int(1000 * key[2]))
        req = make_request(free, V[key])
        Gp, Pp = req.gap(BIG, 1.0, rng, k=CFG30['kb'])
        G0, P0 = req.gap(CFG30['ref'][0], CFG30['ref'][1], rng, k=CFG30['kb'])
        Pinc = np.array(V[key]['Pinc']); Pref = np.array(V[key]['P_ref'])
        Gl, Pl = make_request(lim, V[key]).gap(BIG, 1.0, rng, k=CFG30['kb'])   # optimizer of the limited physical cell
        rows.append(dict(hour=key[0], w=key[1], cut=key[2], G_phys=r.G_phys, G_phys_nolim=min(Gp, r.G_phys), G_phys_nolim_raw=Gp,
                         G_ref=r.G_ref, G_ref_nolim=min(G0, r.G_ref), G_ref_nolim_raw=G0,
                         bind_inc='|'.join(lim.net.label(i) for i in lim.net.binding(Pinc)),
                         G_phys_lim_resolve=Gl, bind_phys='|'.join(lim.net.label(i) for i in lim.net.binding(Pl)),
                         bind_ref='|'.join(lim.net.label(i) for i in lim.net.binding(Pref)),
                         nolim_phys_violates=not lim.line_ok(Pp), nolim_ref_violates=not lim.line_ok(P0)))
        print(key, round(r.G_phys, 4), round(Gp, 4), round(r.G_ref, 4), round(G0, 4), rows[-1]['bind_ref'], flush=True)
    pd.DataFrame(rows).to_csv(d.parent / 'split_dc.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
