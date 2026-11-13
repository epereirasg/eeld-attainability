"""Allowance counterfactuals under the two objective scales (Table 8).

For every distinct EELD10 request of the main campaign, the script recomputes, with identical solver
settings, the physical gap, the reference gap and the counterfactuals rho*, R* and the class of the
minimal unit set S* (1, 2 or >= 3 units) under the target-linked scale s and under the cloud scale
(P90 - P10 of 500 random feasible dispatches, stored in the campaign vectors).

The anchors (and therefore the target-directed starts P^C, P^E) are rebuilt from the campaign seed of
each hour, so the start pool matches the campaign. The target-scale values are recomputed rather than
read from the campaign so that both scales share one random stream and one code path; the stored
campaign values serve as a consistency check.
Output: results/raw/supplementary/scale_cf_p<part>.csv
"""
import argparse, itertools, json, os, time
import numpy as np, pandas as pd
from eeldattain.model import System, Request, anchors
from eeldattain.paths import COEFFS, RAW_ATTAIN, RAW_SUPP
from eeldattain.runio import guard_config, append_rows
from eeldattain import globalopt

EPS, RHO0, R0, BIS, R_HI, KB = 0.05, 0.05, 50.0, 8, 1000.0, 4


def bisect(fun, lo, hi, thr, steps):
    if fun(hi) > thr:
        return np.nan
    for _ in range(steps):
        m = 0.5 * (lo + hi)
        if fun(m) <= thr: hi = m
        else: lo = m
    return hi


def pinned(req, P, rho):
    lo, hi = req.band(rho); s = req.s
    return [i for i in range(s.n) if min(P[i] - lo[i], hi[i] - P[i]) < 1e-3 and (hi[i] < s.hi[i] - 1e-6 or lo[i] > s.lo[i] + 1e-6)]


def counterfactuals(req, rng):
    Gp, _ = req.gap(1e9, 1.0, rng, k=KB)
    G0, _ = req.gap(R0, RHO0, rng, k=KB)
    out = dict(G_phys=Gp, G_ref=G0, needs_cf=bool(G0 > Gp + EPS), rho_star=np.nan, R_star=np.nan, S_class='')
    if not out['needs_cf']:
        return out
    thr = Gp + EPS
    out['rho_star'] = bisect(lambda x: req.gap(1e9, x, rng, k=KB)[0], RHO0, 1.0, thr, BIS)
    out['R_star'] = bisect(lambda x: req.gap(x, 1.0, rng, k=KB)[0], 0.0, R_HI, thr, BIS)
    G_Rfree, P_Rfree = req.gap(1e9, RHO0, rng, k=KB)
    out['G_Rfree'] = G_Rfree
    if G_Rfree <= thr:
        out['S_class'] = '0'
        return out
    pins = pinned(req, P_Rfree, RHO0); out['n_pinned'] = len(pins)
    out['S_class'] = '>=3'
    for k in (1, 2):                                 # exhaustive over all units up to size 2
        hit = None
        for S in itertools.combinations(range(req.s.n), k):
            rv = np.full(req.s.n, RHO0); rv[list(S)] = 1.0
            if req.gap(1e9, rv, rng, k=KB)[0] <= thr:
                hit = S; break
        if hit is not None:
            out['S_class'] = str(k); out['S_units'] = '-'.join(str(i + 1) for i in hit); break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(RAW_SUPP))
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    guard_config(a.out, f'scale_cf_p{a.part}', dict(eps=EPS, rho0=RHO0, R0=R0, bis=BIS, R_hi=R_HI, kb=KB,
                                                    part=a.part, parts=a.parts, set_search='exhaustive, all units, size <= 2'))
    out = f'{a.out}/scale_cf_p{a.part}.csv'
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out); done = set(zip(d.hour, d.w, d.cut, d.scale))
    s = System(str(COEFFS['10']))
    R = pd.concat([pd.read_csv(f) for f in sorted(RAW_ATTAIN.glob('requests_e10*.csv'))])
    R = R[R.duplicate_of_w.isna()]
    V = {}
    for f in sorted(RAW_ATTAIN.glob('vectors_e10*.jsonl')):
        for line in open(f):
            v = json.loads(line); V[(v['hour'], v['w'], v['cut'])] = v
    hours = sorted(R.hour.unique())[a.part::a.parts]
    if os.environ.get('EELD_HOURS'):          # rerun of some hours, merged afterwards
        hours = [h for h in hours if h in {int(x) for x in os.environ['EELD_HOURS'].split(',')}]
    for h in hours:
        Rh = R[R.hour == h]
        PD = float(Rh.PD.iloc[0])
        A = anchors(s, PD, np.random.default_rng(20261022 + 10000 + h), K=100)   # campaign seed of the hour, first use
        A = globalopt.certify_anchors(s, PD, A)                                 # the SCIP-checked anchors of the campaign
        for r in Rh.itertuples():
            v = V[(r.hour, r.w, r.cut)]
            for scale_name, key in (('target', 'scale'), ('cloud', 'scale_cloud')):
                if (r.hour, r.w, r.cut, scale_name) in done:
                    continue
                t0 = time.time()
                sd = 7_000_000 + 1000 * int(r.hour) + int(1000 * r.w) + int(10000 * r.cut)
                rng = np.random.default_rng(sd)              # same stream for both scales
                req = Request(s, PD, np.array(v['Pinc']), np.array(v['Ft']), np.array(v[key]), np.array(v['Pfr']), A['Pc'], A['Pe'])
                res = counterfactuals(req, rng)
                row = dict(hour=r.hour, w=r.w, cut=r.cut, scale=scale_name, seed=sd, **res,
                           campaign_G_ref=r.G_ref, campaign_G_phys=r.G_phys, campaign_rho_star=r.rho_star,
                           campaign_R_star=r.R_star, campaign_S_size=r.S_size,
                           campaign_G_ref_cloud=r.G_ref_cloud, campaign_G_phys_cloud=r.G_phys_cloud,
                           solver_fallbacks=req.fallbacks, runtime_s=time.time() - t0)
                append_rows(out, [row])
                print(f'[scale_cf] h={r.hour} w={r.w} cut={r.cut} {scale_name}: rho*={res["rho_star"]:.3f} '
                      f'R*={res["R_star"]:.0f} S={res["S_class"]} t={row["runtime_s"]:.0f}s', flush=True)
    print('DONE', time.strftime('%Y-%m-%dT%H:%M:%S'))


if __name__ == '__main__':
    main()
