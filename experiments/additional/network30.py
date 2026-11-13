"""The attainability campaign on a documented 30-bus network variant with DC line limits.

The variant (src/eeldattain/network.py) uses the IEEE 30-bus network and loads of MATPOWER case_ieee30, the
line ratings of MATPOWER case30 (Alsac and Stott), the six-unit cost and emission coefficients of Abido (2003,
Table I), generator limits of 5 to 150 MW and the generator buses 1, 2, 5, 8, 11, 13. Five load levels scale
every bus load by 0.8, 0.9, 1.0, 1.1 and 1.2. Each level is run twice with the per-request procedure of
experiments/attainability/run.py, once with the DC line limits (network 'dc') and once on a copper plate
(network 'copper'). Line limits enter every solve (anchors, incumbents, targets, gaps, counterfactuals).

Before the campaign the script checks that the copper-plate anchors at the nominal load reproduce the
minimum-cost and minimum-emission dispatches of Abido's Table II (Case 1).

Output: results/raw/additional/network/{dc,copper}/ with the schema of results/raw/attainability.
Run:  python experiments/additional/network30.py --network dc     (and --network copper)
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'attainability'))
import run as A                                            # experiments/attainability/run.py
from eeldattain.network import NetSystem
from eeldattain.model import anchors
from eeldattain.paths import RAW
from eeldattain.runio import guard_config

MULT = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2}             # level index -> load multiplier
CFG30 = dict(coeffs='data/eeld30_coeffs.csv', K=40, kb=4, weights=(0.25, 0.5, 0.75), cuts=(0.025, 0.05, 0.10),
             R_grid=(10, 25, 50, 100, 250), rho_grid=(0.02, 0.05, 0.10, 0.20),
             R_front=(5, 10, 20, 40, 80, 160, 320), ref=(50.0, 0.05), alt=(100.0, 0.10),
             eps=0.05, bis=8, R_hi=1000.0, set_mode='exhaustive', set_max=3, skip_cloud=True, checkpoint=True)
# Abido (2003), Table II, Case 1 (lossless, no line limits), outputs in p.u. of 100 MVA, cost in $/h
ABIDO_PC = np.array([0.1095, 0.2997, 0.5245, 1.0160, 0.5247, 0.3596]); ABIDO_CMIN = 600.11
ABIDO_PE = np.array([0.4058, 0.4592, 0.5380, 0.3830, 0.5379, 0.5101])


def check_abido():
    s = NetSystem(1.0, limits=False)
    Ad = anchors(s, s.net.PD, np.random.default_rng(1), K=20)
    dc = np.abs(Ad['Pc'] / 100 - ABIDO_PC).max(); de = np.abs(Ad['Pe'] / 100 - ABIDO_PE).max()
    res = dict(PD=s.net.PD, Cmin=s.C(Ad['Pc']), Cmin_abido=ABIDO_CMIN, maxdiff_Pc_pu=dc, maxdiff_Pe_pu=de,
               E_Pe=s.E(Ad['Pe']), E_Pc=s.E(Ad['Pc']), E_abidoPc=s.E(100 * ABIDO_PC), E_abidoPe=s.E(100 * ABIDO_PE))
    ok = abs(res['Cmin'] - ABIDO_CMIN) < 0.05 and dc < 1e-3 and de < 1e-3
    return ok, res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--network', required=True, choices=('dc', 'copper'))
    ap.add_argument('--levels', default='1,2,3,4,5')
    a = ap.parse_args()
    out = RAW / 'additional' / 'network' / a.network
    out.mkdir(parents=True, exist_ok=True)
    ok, res = check_abido()
    json.dump(res, open(out.parent / 'abido_check.json', 'w'), indent=1)
    print('Abido Table II check', ok, res, flush=True)
    if not ok:
        raise SystemExit('the copper-plate anchors do not reproduce Abido Table II')
    tag = f'n30{a.network}'
    guard_config(out, tag, dict(network=a.network, levels=a.levels, mult=MULT, cfg=CFG30))
    A.OUT = str(out)
    fr, fg, fv = out / f'requests_{tag}.csv', out / f'grid_{tag}.csv', out / f'vectors_{tag}.jsonl'
    fj = out / f'requests_{tag}.jsonl'          # requests differ in columns (counterfactuals only when needed)

    def emit(r, g, v):
        with open(fj, 'a') as f: f.write(json.dumps(r, default=float) + '\n')
        pd.DataFrame([json.loads(l) for l in open(fj)]).to_csv(fr, index=False)
        pd.DataFrame(g).to_csv(fg, mode='a', header=not fg.exists(), index=False)
        with open(fv, 'a') as f: f.write(json.dumps(v) + '\n')

    prev = pd.DataFrame([json.loads(l) for l in open(fj)]) if fj.exists() else pd.DataFrame(columns=['hour', 'w', 'cut'])
    t0 = time.time()
    for h in [int(x) for x in a.levels.split(',')]:
        skip = {(float(x.w), float(x.cut)) for x in prev[prev.hour == h].itertuples()}
        if len(skip) == len(CFG30['weights']) * len(CFG30['cuts']):
            continue
        sysm = NetSystem(MULT[h], limits=(a.network == 'dc'))
        A.System = lambda path, _s=sysm: _s                  # run_hour builds its system through A.System
        A.run_hour('30', CFG30, h, sysm.net.PD, seed=A.seed_hour('30', h), ckpt=dict(skip=skip, emit=emit))
        print(tag, h, round(time.time() - t0), flush=True)
    print(tag, 'DONE', flush=True)


if __name__ == '__main__':
    main()
