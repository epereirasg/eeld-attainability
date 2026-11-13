"""Solver certificate (RQ4).

1. Structural checks on every stored cell: 0 <= G <= 1, G_phys <= G, and monotonicity in R_max and rho.
2. Independent re-solve of a stratified sample of grid cells with differential evolution in the reduced
   space (last unit as slack, penalties for slack bounds and the budget), polished by SLSQP started only
   from the DE point. The certificate fails at a cell if DE improves on the campaign value by more than 0.01.
"""
import json, glob, argparse, os
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
from eeldattain.model import System, Request, solve
from eeldattain.paths import COEFFS, RAW_ATTAIN, RAW_CERT
from eeldattain.runio import guard_config
OUT = str(RAW_ATTAIN)     # campaign outputs to certify (input)
HERE = str(RAW_CERT)      # certificate outputs, overridden by --out


def grid_files(case, order=None):
    """Grid files of a case in a fixed order. The order matters for the stratified sample below."""
    if order:
        return [f'{OUT}/grid_{t}.csv' for t in order.split(',')]
    return sorted(glob.glob(f'{OUT}/grid_e{case}*.csv'))


def structural(case, order=None):
    G = pd.concat([pd.read_csv(f) for f in grid_files(case, order)])
    rows = []
    key = ['hour', 'w', 'cut']
    for k, X in G.groupby(key):
        g = X[X.kind == 'grid'].pivot_table(index='R', columns='rho', values='G').values
        f = X[X.kind == 'budget_frontier'].sort_values('R').G.values
        gp = X.G_phys.iloc[0]
        rows.append(dict(case=case, hour=k[0], w=k[1], cut=k[2],
                         viol_R=int((np.diff(g, axis=0) > 0.01).sum()), viol_rho=int((np.diff(g, axis=1) > 0.01).sum()),
                         viol_frontier=int((np.diff(f) > 0.01).sum()), viol_phys=int((g < gp - 0.01).sum() + (f < gp - 0.01).sum()),
                         out_of_range=int(((g < -1e-9) | (g > 1 + 1e-9)).sum()), cells=int(g.size + f.size)))
    return pd.DataFrame(rows)


def de_gap(req, R, rho, seed, maxiter, pop):
    s, PD, Pinc = req.s, req.PD, req.Pinc
    lo, hi = req.band(rho)
    def full(z): return np.append(z, PD - z.sum())
    def obj(z):
        P = full(z)
        v = max(0, lo[-1] - P[-1]) + max(0, P[-1] - hi[-1]) + max(0, np.abs(P - Pinc).sum() - R)
        return req.d(np.clip(P, lo, hi)) / req.e0 + 10.0 * v
    r = differential_evolution(obj, list(zip(lo[:-1], hi[:-1])), seed=seed, maxiter=maxiter, popsize=pop,
                               tol=1e-10, polish=False, init='sobol')
    cons = () if R >= 1e8 else ({'type': 'ineq', 'fun': lambda P: R - np.abs(P - Pinc).sum(), 'jac': lambda P: -np.sign(P - Pinc)},)
    P0 = full(r.x)
    b = solve(s, PD, req.d, req.grad, [P0], lo=lo, hi=hi, cons=cons)
    raw_feasible = obj(r.x) < 1.0 + 1e-6 and (max(0, lo[-1] - P0[-1]) + max(0, P0[-1] - hi[-1]) + max(0, np.abs(P0 - Pinc).sum() - R)) < 1e-6
    vals = [1.0]
    if b is not None: vals.append(b[0] / req.e0)
    if raw_feasible: vals.append(req.d(P0) / req.e0)
    # status: 'polished' (SLSQP from the DE point accepted), 'raw' (only the DE point feasible),
    # 'none' (neither; the value 1.0 is then the incumbent, not an independent solve)
    status = 'polished' if b is not None else ('raw' if raw_feasible else 'none')
    return min(vals), status


def main():
    global HERE, OUT
    ap = argparse.ArgumentParser(); ap.add_argument('--case', required=True, choices=['10', '40']); ap.add_argument('--frac', type=float, required=True)
    ap.add_argument('--maxiter', type=int, default=400); ap.add_argument('--pop', type=int, default=25)
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--out', default=HERE, help='output directory (default: results/raw/certificate)')
    ap.add_argument('--grid-order', default='', help='comma-separated grid tags in concatenation order '
                    '(default: sorted). The stored samples used e10a,e10b and e40b,e40a.')
    ap.add_argument('--limit', type=int, default=0, help='verification only: re-solve the first N sampled cells')
    ap.add_argument('--campaign', default=OUT, help='attainability outputs to certify (default: results/raw/attainability)')
    a = ap.parse_args()
    OUT = a.campaign
    HERE = a.out; os.makedirs(HERE, exist_ok=True)
    guard_config(HERE, f'e{a.case}_p{a.part}', dict(case=a.case, frac=a.frac, maxiter=a.maxiter, pop=a.pop,
                                                    part=a.part, parts=a.parts, sample_random_state=7,
                                                    grid_files=[os.path.basename(f) for f in grid_files(a.case, a.grid_order)]))
    S = structural(a.case, a.grid_order); S.to_csv(f'{HERE}/structural_e{a.case}.csv', index=False)
    print('structural', a.case, S[['viol_R', 'viol_rho', 'viol_frontier', 'viol_phys', 'out_of_range']].sum().to_dict(), 'cells', S.cells.sum(), flush=True)
    G = pd.concat([pd.read_csv(f) for f in grid_files(a.case, a.grid_order)]).reset_index(drop=True)
    V = {}
    for f in sorted(glob.glob(f'{OUT}/vectors_e{a.case}*.jsonl')):
        for line in open(f):
            v = json.loads(line); V[(v['hour'], v['w'], v['cut'])] = v
    sample = G.groupby(['kind', 'R', 'rho']).sample(frac=a.frac, random_state=7)
    sample = sample.iloc[a.part::a.parts]
    if a.limit:
        sample = sample.iloc[:a.limit]
    from eeldattain.runio import campaign_escale
    s = System(str(COEFFS[a.case]), escale=campaign_escale(a.case, OUT))
    rows = []
    only = {int(x) for x in os.environ['EELD_HOURS'].split(',')} if os.environ.get('EELD_HOURS') else None
    for i, r in enumerate(sample.itertuples()):
        if only is not None and int(r.hour) not in only:     # rerun of some hours, same sample and DE seeds
            continue
        v = V[(r.hour, r.w, r.cut)]
        req = Request(s, v['PD'], np.array(v['Pinc']), np.array(v['Ft']), np.array(v['scale']), None, None, None)
        Gde, status = de_gap(req, r.R, r.rho, seed=i, maxiter=a.maxiter, pop=a.pop)
        rows.append(dict(case=a.case, hour=r.hour, w=r.w, cut=r.cut, kind=r.kind, R=r.R, rho=r.rho, G_campaign=r.G, G_DE=Gde,
                         improvement=r.G - Gde, de_seed=i, de_status=status))
        if i % 10 == 0:
            pd.DataFrame(rows).to_csv(f'{HERE}/de_e{a.case}_p{a.part}.csv', index=False); print(a.case, a.part, i, len(sample), flush=True)
    pd.DataFrame(rows).to_csv(f'{HERE}/de_e{a.case}_p{a.part}.csv', index=False)
    print('DONE', a.case, a.part, flush=True)


if __name__ == '__main__':
    main()
