import os
"""Budget-free gap of the loss model under uniform bands, for part (c) of the RQ2 rule in RQ8.

The operational campaign stored G*(R0, rho0) with uniform bands and the band-free gap G*(R0, 1), but the
budget-free gap G*(inf, rho0) only for ramp bands. This script computes G*(inf, rho0 = 0.05) with uniform bands
for every distinct loss-model request that needs a counterfactual (G_ref_uniform > G_phys + 0.05), from the
stored incumbents, targets and scales and the cached anchors of each hour.
Output: results/raw/additional/ops_uniform_relief.csv
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from eeldattain.paths import RAW_OPS, RAW
from eeldattain.runio import load_cache
from eeldops.experiments import OPS, seed_of, system
from eeldops.model import OpsRequest

OUT = Path(os.environ.get('EELD_RAW_ADD', RAW / 'additional')) / 'ops_uniform_relief.csv'


def main():
    s, _, _ = system()
    R = pd.read_csv(RAW_OPS / 'ops_requests_a.csv', dtype={'S_units': str})
    if os.environ.get('EELD_HOURS'):          # rerun of some hours, merged afterwards
        R = R[R.hour.isin([int(x) for x in os.environ['EELD_HOURS'].split(',')])]
    R = R[R.duplicate_of_w.isna() & (R.G_ref_uniform > R.G_phys + OPS['eps'])]
    V = {(int(v['hour']), float(v['w']), float(v['cut'])): v for v in map(json.loads, open(RAW_OPS / 'ops_vectors_a.jsonl'))}
    done = set()
    if OUT.exists():
        done = {(int(h), float(w), float(c)) for h, w, c in pd.read_csv(OUT)[['hour', 'w', 'cut']].itertuples(index=False)}
    for r in R.itertuples():
        key = (int(r.hour), float(r.w), float(r.cut))
        if key in done:
            continue
        v = V[key]; A, _ = load_cache(RAW_OPS / 'cache' / f'ops_h{key[0]}.json')
        req = OpsRequest(s, float(v['PD']), v['Pinc'], v['Ft'], v['scale'], [np.array(v['Pfr']), A['Pc'], A['Pe']])
        rng = np.random.default_rng(seed_of(9, *key))
        G, P = req.gap(1e9, s.band_uniform(OPS['rho0']), rng, k=OPS['kb'], extra_starts=[np.array(v['P_ref'])])
        G0, _ = req.gap(OPS['R0'], s.band_uniform(OPS['rho0']), rng, k=OPS['kb'])     # re-solve of the stored value
        row = dict(hour=key[0], w=key[1], cut=key[2], G_phys=r.G_phys, G_ref_uniform=r.G_ref_uniform,
                   G_ref_uniform_resolve=G0, G_bandfree=r.G_bandfree, G_Rfree_uniform=G, fallbacks=req.fallbacks)
        pd.DataFrame([row]).to_csv(OUT, mode='a', header=not OUT.exists(), index=False)
        print(key, round(r.G_ref_uniform, 4), round(G0, 4), round(G, 4), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
