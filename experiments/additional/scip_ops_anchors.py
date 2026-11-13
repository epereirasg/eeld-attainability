"""SCIP check of the anchors and targets of the operational (Kron-loss) model of the 10-unit system.

For every hour it solves min C and min E over the loss-model feasible set, sum P = D + P^T B P, and for every distinct
reachable request min C subject to E <= E_tar (Eq. 5 with losses), with src/eeldattain/globalopt.py. It also checks
every weighted-sum incumbent (Eq. 4 with losses). It compares
the polished SCIP values with the cached multistart anchors and the stored targets.
Output: results/raw/additional/scip/ops_anchors.csv, ops_incumbents.csv and ops_targets.csv
"""
import json
import os
import numpy as np, pandas as pd
from eeldattain.paths import RAW_OPS, RAW
from eeldattain.runio import load_cache
from eeldattain import globalopt
from eeldops.experiments import system

OUT = RAW / 'additional' / 'scip'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, _, dem = system()
    rows = []
    for h in ([] if os.environ.get('EELD_TARGETS_ONLY') or os.environ.get('EELD_INC_ONLY') else range(1, 25)):
        PD = float(dem.loc[dem.hour == h, 'PD'].iloc[0])
        A, INC = load_cache(f'{RAW_OPS}/cache/ops_h{h}.json')
        for kind, P0 in (('C', A['Pc']), ('E', A['Pe'])):
            log = []
            P = globalopt.improve(s, PD, kind, np.asarray(P0), log=log)
            rows.append(dict(hour=h, PD=PD, **log[0]))
            print(h, kind, log[0]['v_multistart'], log[0]['v_final'], log[0]['scip_lb'], log[0]['adopted'], flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'ops_anchors.csv', index=False)
    rows = []
    for h in ([] if os.environ.get('EELD_TARGETS_ONLY') else range(1, 25)):
        PD = float(dem.loc[dem.hour == h, 'PD'].iloc[0])
        A, INC = load_cache(f'{RAW_OPS}/cache/ops_h{h}.json')
        for w, P0 in INC.items():
            log = []
            globalopt.improve(s, PD, 'W', np.asarray(P0), w=float(w), ideal=A['ideal'], nadir=A['nadir'], log=log)
            rows.append(dict(log[0], hour=h, PD=PD, w=float(w)))
            print('incumbent', h, w, log[0]['v_multistart'], log[0]['v_final'], log[0]['adopted'], flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'ops_incumbents.csv', index=False)
    if os.environ.get('EELD_INC_ONLY'):
        print('DONE'); return
    R = pd.read_csv(f'{RAW_OPS}/ops_requests_a.csv')
    V = {(v['hour'], v['w'], v['cut']): v for v in map(json.loads, open(f'{RAW_OPS}/ops_vectors_a.jsonl'))}
    rows = []
    for r in R.itertuples():
        if not getattr(r, 'reachable', False) or not pd.isna(getattr(r, 'duplicate_of_w', np.nan)):
            continue
        v = V[(r.hour, r.w, r.cut)]; PD = float(v['PD']); Pfr = np.asarray(v['Pfr']); Et = float(v['Ft'][1]) if 'E_tar' not in R.columns else float(r.E_tar)
        log = []
        globalopt.improve(s, PD, 'C', Pfr, Ecap=Et, log=log)
        rows.append(dict(log[0], hour=r.hour, w=r.w, cut=r.cut))
        print('target', r.hour, r.w, r.cut, log[0]['v_multistart'], log[0]['v_final'], log[0]['adopted'], flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'ops_targets.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
