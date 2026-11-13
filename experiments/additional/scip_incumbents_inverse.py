"""Inverse efficiency check of the 10-unit incumbents with SCIP.

scip_anchors.py certifies that no dispatch with E(P) <= E(P_inc) is cheaper than P_inc. This script adds the inverse
problem  min E(P)  s.t.  P in Omega_h, C(P) <= C(P_inc),  which excludes a dispatch with the same cost and lower
emission. Both checks together certify that P_inc is Pareto efficient up to the solver tolerances.
Output: results/raw/additional/scip/incumbents_inverse.csv
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scip_anchors import build
from eeldattain.model import System
from eeldattain.paths import COEFFS, RAW_ATTAIN, RAW

out = RAW / 'additional' / 'scip'; os.makedirs(out, exist_ok=True)
s = System(str(COEFFS['10']))
req = pd.concat([pd.read_csv(f, dtype={'S_units': str}) for f in sorted(glob.glob(f'{RAW_ATTAIN}/requests_e10*.csv'))])
req = req[req.duplicate_of_w.isna()].drop_duplicates(['hour', 'w']).sort_values(['hour', 'w'])
V = {(v['hour'], v['w']): v for f in sorted(glob.glob(f'{RAW_ATTAIN}/vectors_e10*.jsonl')) for v in map(json.loads, open(f))}
rows = []
for r in req.itertuples():
    v = V[(r.hour, r.w)]; Pinc = np.array(v['Pinc']); PD = float(v['PD'])
    Ci, Ei = s.C(Pinc), s.E(Pinc)
    b = build(s, PD, 'E', Ccap=Ci * (1 + 1e-9))
    rows.append(dict(hour=int(r.hour), w=float(r.w), C_inc=Ci, E_inc=Ei, Emin_lb=b['lb'], Emin_best=b['E_best'],
                     rel_excess=(Ei - b['lb']) / Ei, status=b['status'], time_s=b['time_s']))
    pd.DataFrame(rows).to_csv(out / 'incumbents_inverse.csv', index=False)
    print(r.hour, r.w, rows[-1]['rel_excess'], b['status'], flush=True)
print('DONE')
