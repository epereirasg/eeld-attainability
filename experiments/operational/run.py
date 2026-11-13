"""Operational extension X1-X4 and X7 (losses, ramp-derived bands, three-hour window, seeds, EELD40 sets).

Usage: python experiments/operational/run.py --exp all|ops|window|seeds|sets40 [--out DIR]
Each run checks or writes ``config_<exp>_<tag>.json`` in the output directory, and every output row
stores its seed (base seed 20261023).
"""
import argparse, json, os, time
from eeldops import experiments as X
from eeldattain.runio import guard_config

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--exp', default='all', choices=['all', 'ops', 'window', 'seeds', 'sets40'])
ap.add_argument('--hours', default='1-24', help='range lo-hi of the 24-hour profile')
ap.add_argument('--tag', default='a')
ap.add_argument('--part', type=int, default=0, help='sets40 only: index of this worker')
ap.add_argument('--parts', type=int, default=1, help='sets40 only: number of workers')
ap.add_argument('--out', default=X.RES, help='output directory (default: results/raw/operational)')
ap.add_argument('--attain-dir', default=X.ATT, help='sets40 only: EELD40 attainability outputs (default: results/raw/attainability)')
a = ap.parse_args()
X.ATT = os.path.abspath(a.attain_dir)
lo, hi = (int(x) for x in a.hours.split('-'))
if not (1 <= lo <= hi <= 24):
    raise SystemExit('--hours must be a range inside 1-24')
if not (0 <= a.part < a.parts):
    raise SystemExit('--part must satisfy 0 <= part < parts')
hours = list(range(lo, hi + 1))
X.RES = os.path.abspath(a.out)
os.makedirs(X.RES, exist_ok=True)
cfg = dict(exp=a.exp, hours=a.hours, tag=a.tag, part=a.part, parts=a.parts, OPS=X.OPS, BASE_SEED=X.BASE_SEED)
guard_config(X.RES, f'{a.exp}_{a.tag}' + (f'_p{a.part}' if a.exp == 'sets40' else ''), cfg)
print('CONFIG', json.dumps(cfg, default=str), flush=True)
if a.exp in ('all', 'ops'): X.run_ops(hours, a.tag)
if a.exp in ('all', 'window'): X.run_window(hours, a.tag)
if a.exp in ('all', 'seeds'): X.run_seeds(a.tag, hours=tuple(h for h in X.SEED_HOURS if lo <= h <= hi))
if a.exp in ('all', 'sets40'): X.run_sets40(a.tag, a.part, a.parts)
print('DONE', a.exp, time.strftime('%Y-%m-%dT%H:%M:%S'), flush=True)
