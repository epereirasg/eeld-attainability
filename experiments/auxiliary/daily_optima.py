"""Daily single-objective optima of the 10-unit profile without losses or ramps, standard and scaled emission.

Writes results/raw/daily_optima.json, the source of the magnitudes quoted in the experimental design
section (emission-convention check).
"""
import argparse, json, time
import numpy as np, pandas as pd
from eeldattain.model import System, solve
from eeldattain.paths import COEFFS, DEMAND, RAW

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--out', default=str(RAW / 'daily_optima.json'), help='output JSON file (default: results/raw/daily_optima.json)')
a = ap.parse_args()
dem = pd.read_csv(DEMAND['10']); out = {'seed': 11, 'starts': 30, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
for tag, esc in [('standard', 1.0), ('scaled', 1e-2)]:
    s = System(str(COEFFS['10']), escale=esc); rng = np.random.default_rng(11); C = E = 0.0
    for PD in dem.PD:
        x0 = [s.rand_feasible(PD, rng) for _ in range(30)]
        bc, be = solve(s, PD, s.C, s.gC, x0), solve(s, PD, s.E, s.gE, x0)
        if bc is None or be is None:
            raise RuntimeError(f'no feasible single-objective optimum at PD={PD}')
        C += bc[0]; E += be[0]
    out[tag] = dict(cost_usd=C, emission_lb=E)
    print(tag, f'{C:.4g} $', f'{E:.4g} lb', flush=True)
json.dump(out, open(a.out, 'w'), indent=1)
