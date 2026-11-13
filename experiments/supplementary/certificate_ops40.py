"""Solver certificate for the new cells of the stylized 40-unit operational subset (RQ4, Table 13).

Every reference cell of experiments/supplementary/ops40.py (physical gap, reference gaps with uniform and
ramp-derived bands, budget-free and band-free relaxations, for both models) is re-solved independently by
differential evolution in the reduced space. The last unit is the slack. With losses, the slack output
solves the Kron balance exactly (a quadratic in one variable), so every DE point meets the balance; band,
budget and slack-limit violations are penalized. The DE point is then polished by SLSQP started only
from it. A cell fails the certificate if the independent solve improves on the stored value by more
than 0.01.
Output: results/raw/supplementary/certificate_ops40_p<part>.csv (or <name>.csv). The stored run certified each half of
the replication separately (results and vectors of part 0, then of part 1 with --name certificate_ops40_p1).
"""
import argparse, json, os, time
from pathlib import Path
import importlib.util
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
from eeldattain.paths import RAW_SUPP
from eeldattain.runio import guard_config, append_rows
from eeldops.model import OpsRequest

spec = importlib.util.spec_from_file_location('ops40', Path(__file__).with_name('ops40.py'))
O40 = importlib.util.module_from_spec(spec); spec.loader.exec_module(O40)


def slack_output(s, z, PD):
    """Output of the last unit that closes sum(P) = PD + P'BP for the other outputs z (nan if none)."""
    n = s.n; B = s.B; Bzz = B[:n - 1, :n - 1]; b = B[n - 1, :n - 1] @ z; Bnn = B[n - 1, n - 1]
    c0 = z.sum() - PD - z @ Bzz @ z
    if Bnn <= 0:
        return PD - z.sum() + z @ Bzz @ z if abs(1 - 2 * b) < 1e-12 else -c0 / (1 - 2 * b)
    disc = (1 - 2 * b) ** 2 + 4 * Bnn * c0
    if disc < 0:
        return np.nan
    return ((1 - 2 * b) - np.sqrt(disc)) / (2 * Bnn)        # the root near the lossless solution


def de_cell(req, R, band, seed, maxiter, pop):
    s, PD, Pinc = req.s, req.PD, req.Pinc
    lo, hi = req.box(band)

    def full(z):
        return np.append(z, slack_output(s, z, PD))

    def viol(P):
        if not np.isfinite(P[-1]):
            return 1e3
        v = max(0.0, lo[-1] - P[-1]) + max(0.0, P[-1] - hi[-1])
        if R < 1e8:
            v += max(0.0, np.abs(P - Pinc).sum() - R)
        return v

    def obj(z):
        P = full(z); v = viol(P)
        if v >= 1e3:
            return 10.0 + v
        return req.d(np.clip(P, lo, hi)) / req.e0 + 10.0 * v

    r = differential_evolution(obj, list(zip(lo[:-1], hi[:-1])), seed=seed, maxiter=maxiter, popsize=pop,
                               tol=1e-10, polish=False, init='sobol')
    P0 = full(r.x)
    cons = () if R >= 1e8 else ({'type': 'ineq', 'fun': lambda P: R - np.abs(P - Pinc).sum(),
                                 'jac': lambda P: -np.sign(P - Pinc)},)
    vals, status = [1.0], 'none'
    if np.isfinite(P0[-1]):
        b = s.solve(PD, req.d, req.grad, [np.clip(P0, lo, hi)], lo=lo, hi=hi, cons=cons)
        if b is not None:
            vals.append(b[0] / req.e0); status = 'polished'
        if viol(P0) < 1e-6 and abs(s.residual(P0, PD)) < 1e-5:
            vals.append(req.d(P0) / req.e0); status = status if status == 'polished' else 'raw'
    return min(vals), status


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--results', default=str(RAW_SUPP / 'ops40full_p0.csv'), help='comma-separated ops40 result files')
    ap.add_argument('--vectors', default=str(RAW_SUPP / 'ops40full_p0_vectors.jsonl'), help='comma-separated vector files')
    ap.add_argument('--out', default=str(RAW_SUPP))
    ap.add_argument('--maxiter', type=int, default=150); ap.add_argument('--pop', type=int, default=10)
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--name', default=None, help='output stem (default certificate_ops40_p<part>)')
    a = ap.parse_args()
    name = a.name or f'certificate_ops40_p{a.part}'
    os.makedirs(a.out, exist_ok=True)
    guard_config(a.out, name, dict(maxiter=a.maxiter, pop=a.pop, part=a.part, parts=a.parts))
    out = f'{a.out}/{name}.csv'
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out); done = set(zip(d.PD, d.w, d.cut, d.model, d.cell))
    sL, sB, _ = O40.stylized_system()
    X = pd.concat([pd.read_csv(f) for f in a.results.split(',')])
    V = [json.loads(l) for f in a.vectors.split(',') for l in open(f)]
    jobs = []
    for v in V:
        s = sB if v['model'] == 'loss' else sL
        st = X[(X.PD == v['PD']) & (X.w == v['w']) & (X.cut == v['cut'])].iloc[0]
        R0 = O40.CFG['R0']
        for cell, (R, band) in {'G_phys': (1e9, s.span), 'G_ref_uniform': (R0, s.band_uniform(O40.CFG['rho0'])),
                                'G_ref_ramp': (R0, s.band_ramp(O40.CFG['tau0'])), 'G_Rfree_ramp': (1e9, s.band_ramp(O40.CFG['tau0'])),
                                'G_bandfree': (R0, s.span)}.items():
            jobs.append((v, s, cell, R, band, float(st[f'{cell}_{v["model"]}'])))
    for i, (v, s, cell, R, band, G_stored) in enumerate(jobs):
        if i % a.parts != a.part or (v['PD'], v['w'], v['cut'], v['model'], cell) in done:
            continue
        t0 = time.time()
        req = OpsRequest(s, v['PD'], v['Pinc'], v['Ft'], v['scale'], [])
        G_de, status = de_cell(req, R, band, seed=i, maxiter=a.maxiter, pop=a.pop)
        append_rows(out, [dict(PD=v['PD'], w=v['w'], cut=v['cut'], model=v['model'], cell=cell, G_stored=G_stored, G_DE=G_de,
                               improvement=G_stored - G_de, de_seed=i, de_status=status, runtime_s=time.time() - t0)])
        print(f"[cert40] {i} PD={v['PD']} w={v['w']} cut={v['cut']} {v['model']} {cell}: stored={G_stored:.4f} DE={G_de:.4f} "
              f"{status} t={time.time() - t0:.0f}s", flush=True)
    print('DONE', time.strftime('%Y-%m-%dT%H:%M:%S'))


if __name__ == '__main__':
    main()
