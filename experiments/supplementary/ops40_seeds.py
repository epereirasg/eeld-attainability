"""Seed replication of the stylized 40-unit operational cells (RQ9, seed sensitivity).

Every stored request of experiments/supplementary/ops40.py (both models, with and without losses) is re-solved
with three independent random-start seeds and four random starts per solve (the campaign used one).
The cells are the physical gap, the reference gaps with uniform and ramp-derived bands, and the two
single-instrument relaxations. The spread across seeds measures the solver noise of each cell.
Output: results/raw/supplementary/ops40_seeds.csv (or ops40_seeds_p<part>.csv with --parts > 1)
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from eeldattain.paths import RAW_SUPP
from eeldattain.runio import guard_config, append_rows
from eeldops.model import OpsRequest
from eeldops.experiments import seed_of
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('ops40', Path(__file__).with_name('ops40.py'))
O40 = importlib.util.module_from_spec(spec); spec.loader.exec_module(O40)
SEEDS = tuple(int(x) for x in os.environ.get("EELD_SEEDS", "1,2,3").split(","))


def cells(s, req, CFG):
    R0 = CFG['R0']
    return {'G_phys': (1e9, s.span), 'G_ref_uniform': (R0, s.band_uniform(CFG['rho0'])),
            'G_ref_ramp': (R0, s.band_ramp(CFG['tau0'])), 'G_Rfree_ramp': (1e9, s.band_ramp(CFG['tau0'])),
            'G_bandfree': (R0, s.span)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--vectors', default=str(RAW_SUPP / 'ops40full_p0_vectors.jsonl'), help='comma-separated vector files')
    ap.add_argument('--out', default=str(RAW_SUPP))
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = 'ops40_seeds' if a.parts == 1 else f'ops40_seeds_p{a.part}'
    guard_config(a.out, tag, dict(seeds=SEEDS, k=a.k, CFG=O40.CFG, part=a.part, parts=a.parts))
    out = f'{a.out}/{tag}.csv'
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out); done = set(zip(d.PD, d.w, d.cut, d.model, d.rep))
    sL, sB, assumed = O40.stylized_system()
    V = [json.loads(l) for f in a.vectors.split(',') for l in open(f)]
    anchors = {}
    for j, v in enumerate(V):
        if j % a.parts != a.part:
            continue
        s = sB if v['model'] == 'loss' else sL
        key = (v['model'], v['PD'])
        if key not in anchors and any((v['PD'], v['w'], v['cut'], v['model'], r) not in done for r in SEEDS):
            # same anchor seed as ops40.py, so the target-directed starts are those of the stored run
            anchors[key] = s.anchors(v['PD'], np.random.default_rng(seed_of(9, int(v['PD'] / 100))), K=O40.CFG['K'])
        for rep in SEEDS:
            if (v['PD'], v['w'], v['cut'], v['model'], rep) in done:
                continue
            t0 = time.time(); sd = seed_of(10, int(v['PD'] / 100), v['w'], v['cut'], rep); rng = np.random.default_rng(sd)
            A = anchors[key]
            req = OpsRequest(s, v['PD'], v['Pinc'], v['Ft'], v['scale'], [v['Pfr'], A['Pc'], A['Pe']], pool_cap=6)
            row = dict(PD=v['PD'], w=v['w'], cut=v['cut'], model=v['model'], rep=rep, seed=sd, k=a.k)
            for name, (R, band) in cells(s, req, O40.CFG).items():
                row[name] = req.gap(R, band, rng, k=a.k)[0]
            row['solver_fallbacks'] = req.fallbacks; row['runtime_s'] = time.time() - t0
            append_rows(out, [row])
            print(f"[ops40_seeds] PD={v['PD']} w={v['w']} cut={v['cut']} {v['model']} rep={rep} "
                  f"G_ref_ramp={row['G_ref_ramp']:.4f} t={row['runtime_s']:.0f}s", flush=True)
    print('DONE', time.strftime('%Y-%m-%dT%H:%M:%S'))


if __name__ == '__main__':
    main()
