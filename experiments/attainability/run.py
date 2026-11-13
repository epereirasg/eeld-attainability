"""Final attainability campaign (RQ1, RQ2, RQ3, RQ7) for EELD10 and EELD40.

For every request (hour, weight w, emission cut c) the script records
  - physical reach eta_E and the physical gap G*_phys,
  - the attainability grid G*(R_max, rho),
  - the budget-only frontier G*(R, rho = 1),
  - allowance counterfactuals rho*, R* and the minimal unit set S*,
  - the recommendation at the reference allowance (size, efficiency loss),
  - the same key gaps under the alternative cloud scale,
and stores every vector needed to re-solve the request.
"""
import argparse, itertools, json, os, time
import numpy as np, pandas as pd
from eeldattain.model import System, Request, anchors, efficient_incumbent, eps_point
from eeldattain.paths import COEFFS, DEMAND, RAW_ATTAIN
from eeldattain.runio import guard_config, load_cache, save_cache
from eeldattain import globalopt

# The EELD10 rows of results/raw/attainability were produced with this configuration and the
# per-hour random stream (no checkpointing). EELD40 uses per-request checkpointing and seeds.
CFG = {
    '10': dict(coeffs=str(COEFFS['10']), demand=str(DEMAND['10']), K=100, kb=4,
               weights=(0.25, 0.5, 0.75), cuts=(0.025, 0.05, 0.10),
               R_grid=(10, 25, 50, 100, 250), rho_grid=(0.02, 0.05, 0.10, 0.20),
               R_front=(5, 10, 20, 40, 80, 160, 320), ref=(50.0, 0.05), alt=(100.0, 0.10),
               eps=0.05, bis=8, R_hi=1000.0, set_mode='exhaustive', set_max=3, certify=True),
    '40': dict(coeffs=str(COEFFS['40']), demand=str(DEMAND['40']), K=40, kb=1,
               weights=(0.25, 0.75), cuts=(0.05, 0.10),
               R_grid=(50, 100, 250, 500), rho_grid=(0.05, 0.10, 0.20),
               R_front=(50, 100, 200, 400, 800), ref=(200.0, 0.05), alt=(250.0, 0.10),
               eps=0.05, bis=5, R_hi=3000.0, set_mode='greedy', set_max=3, screen=4, pool=6, skip_cloud=True, checkpoint=True),
}
OUT = str(RAW_ATTAIN)       # overridden by --out
PATH_KEYS = ('coeffs', 'demand')


def seed_hour(case, h):
    return 20261022 + 1000 * int(case) + h


def seed_request(hour_seed, w, cut):
    return hour_seed * 1000 + int(1000 * w) + int(10000 * cut)


def bisect(fun, lo, hi, thr, steps):
    if fun(hi) > thr:
        return np.nan
    for _ in range(steps):
        m = 0.5 * (lo + hi)
        if fun(m) <= thr: hi = m
        else: lo = m
    return hi


def pinned_units(req, P, rho):
    lo, hi = req.band(rho)
    s = req.s
    return [i for i in range(s.n) if min(P[i] - lo[i], hi[i] - P[i]) < 1e-3
            and (hi[i] < s.hi[i] - 1e-6 or lo[i] > s.lo[i] + 1e-6)]


def shadow_rank(req, P, units):
    """Rank band-pinned units by the KKT reduced gradient |g_i - mu| times the unit span."""
    g = req.grad(P)
    free = [i for i in range(req.s.n) if i not in units]
    mu = float(np.median(g[free])) if free else 0.0
    return sorted(units, key=lambda i: -abs(g[i] - mu) * req.s.span[i])


def minimal_set(req, cfg, rho0, thr, rng, P_rfree, validate):
    def G_with(S):
        rv = np.full(req.s.n, rho0); rv[list(S)] = 1.0
        return req.gap(1e9, rv, rng, k=cfg['kb'])[0]
    pins = pinned_units(req, P_rfree, rho0)
    res = dict(n_pinned=len(pins), S_size=np.nan, S_units='', S_closed=False, S_exact_check=np.nan)
    if cfg['set_mode'] == 'exhaustive':
        for k in range(1, cfg['set_max'] + 1):
            for S in itertools.combinations(pins, k):
                if G_with(S) <= thr:
                    res.update(S_size=k, S_units='-'.join(str(i + 1) for i in S), S_closed=True)
                    break
            if res['S_closed']:
                break
        if validate:   # exhaustive over all units up to size 2, to test the pinned-unit screening
            best = np.nan
            for k in (1, 2):
                if any(G_with(S) <= thr for S in itertools.combinations(range(req.s.n), k)):
                    best = k; break
            res['S_exact_check'] = best
    else:
        cand = shadow_rank(req, P_rfree, pins)[:cfg['screen']]
        S = []
        for k in range(1, cfg['set_max'] + 1):
            if not cand: break
            vals = {i: G_with(S + [i]) for i in cand}
            i = min(vals, key=vals.get); S.append(i); cand.remove(i)
            if vals[i] <= thr:
                res.update(S_size=k, S_units='-'.join(str(j + 1) for j in S), S_closed=True); break
    if not res['S_closed']:
        res['S_size'] = cfg['set_max'] + 1
    return res


def run_hour(case, cfg, h, PD, seed, ckpt=None):
    T0 = time.time()
    s = System(cfg['coeffs'])
    rng = np.random.default_rng(seed)
    cache = f'{OUT}/anchors_e{case}_h{h}.json' if ckpt is not None else None
    if cache and os.path.exists(cache):
        A, Y, INC = load_cache(cache)
    else:
        A = anchors(s, PD, rng, K=cfg['K'])
        Y = INC = None
    CERT = [] if cfg.get('certify') else None
    if CERT is not None:      # SCIP check of the anchors (src/eeldattain/globalopt.py), no random numbers drawn
        A = globalopt.certify_anchors(s, PD, A, log=CERT)
    Emin, Ccmin = A['ideal'][1], A['ideal'][0]
    W = (A['nadir'][1] - Emin) / A['nadir'][1]
    # cloud scale (alternative normalization): P90 - P10 over 500 random feasible dispatches
    if Y is None:
        Y = np.array([s.F(s.rand_feasible(PD, rng)) for _ in range(500)])
    if ckpt is not None and INC is None:
        INC = {w: efficient_incumbent(s, PD, w, A, K=cfg['K']) for w in cfg['weights']}
        save_cache(cache, (A, Y, INC))
    sc_cloud = np.quantile(Y, 0.9, axis=0) - np.quantile(Y, 0.1, axis=0)
    rows, grid, vecs, seen = [], [], [], {}
    for w in cfg['weights']:
        Pinc = INC[w] if ckpt is not None else efficient_incumbent(s, PD, w, A, K=cfg['K'])
        if CERT is not None:
            Pinc = globalopt.improve(s, PD, 'W', Pinc, w=w, ideal=A['ideal'], nadir=A['nadir'], log=CERT)
        key = tuple(np.round(Pinc, 4))
        dup_of = seen.get(key); seen.setdefault(key, w)
        Finc = s.F(Pinc)
        for cut in cfg['cuts']:
            if ckpt is not None:
                if (w, cut) in ckpt['skip']:
                    continue
                rng = np.random.default_rng(seed_request(seed, w, cut))
                g0 = len(grid)
            Et = (1 - cut) * Finc[1]
            reach = bool(Et >= Emin - 1e-9)
            if reach:
                Pfr = eps_point(s, PD, Et, [Pinc, A['Pe']] + A['front'])
                if CERT is not None:
                    Pfr = globalopt.improve(s, PD, 'C', Pfr, Ecap=Et, log=CERT)
                Ft = s.F(Pfr)
            else:
                Pfr = A['Pe']; Ft = np.array([s.C(A['Pe']), Et])
            sc = np.maximum(np.abs(Ft - Finc), 1e-6 * np.abs(Finc))
            req = Request(s, PD, Pinc, Ft, sc, Pfr, A['Pc'], A['Pe'], pool_cap=cfg.get('pool', 15))
            Gp, _ = req.gap(1e9, 1.0, rng, k=cfg['kb'])
            eta = min(1.0, max(0.0, (Finc[1] - Emin) / (Finc[1] - Et)))
            base = dict(case=case, hour=h, PD=PD, w=w, cut=cut,
                        seed=seed_request(seed, w, cut) if ckpt is not None else seed,
                        duplicate_of_w=dup_of, reachable=reach,
                        E_inc=Finc[1], C_inc=Finc[0], E_min=Emin, C_min=Ccmin, front_width=W, eta_E=eta,
                        C_tar=Ft[0], E_tar=Et, dC_target_pct=100 * (Ft[0] / Finc[0] - 1),
                        mw_full=float(np.abs(Pfr - Pinc).sum()) if reach else np.nan,
                        maxu_full=float(np.max(np.abs(Pfr - Pinc) / s.span)) if reach else np.nan, G_phys=Gp)
            for R in cfg['R_grid']:
                for rho in cfg['rho_grid']:
                    G, P = req.gap(R, rho, rng, k=cfg['kb'])
                    grid.append(dict(base, kind='grid', R=R, rho=rho, G=G, mw=float(np.abs(P - Pinc).sum())))
            for R in cfg['R_front']:
                G, P = req.gap(R, 1.0, rng, k=cfg['kb'])
                grid.append(dict(base, kind='budget_frontier', R=R, rho=1.0, G=G, mw=float(np.abs(P - Pinc).sum())))
            R0, r0 = cfg['ref']
            G0, P0 = req.gap(R0, r0, rng, k=cfg['kb'])
            G_Rfree, P_Rfree = req.gap(1e9, r0, rng, k=cfg['kb'])
            G_rfree, _ = req.gap(R0, 1.0, rng, k=cfg['kb'])
            Peff = eps_point(s, PD, s.E(P0), [P0, A['Pe']] + A['front'])
            r = dict(base, R0=R0, rho0=r0, G_ref=G0, G_Rfree=G_Rfree, G_rhofree=G_rfree,
                     ref_mw=float(np.abs(P0 - Pinc).sum()), ref_errx=float(np.linalg.norm((P0 - Pinc) / s.span)),
                     ref_units_moved=int(np.sum(np.abs(P0 - Pinc) > 0.5)),
                     ref_eff_loss_pct=(100 * (s.C(P0) / s.C(Peff) - 1)) if Peff is not None else np.nan,
                     ref_dE_pct=100 * (s.E(P0) / Finc[1] - 1), ref_dC_pct=100 * (s.C(P0) / Finc[0] - 1))
            # alternative cloud scale
            if not cfg.get('skip_cloud'):       # same call order as the EELD10 production run (commit 09c0100)
                reqc = Request(s, PD, Pinc, Ft, sc_cloud, Pfr, A['Pc'], A['Pe'], pool_cap=cfg.get('pool', 15))
                r.update(G_phys_cloud=reqc.gap(1e9, 1.0, rng, k=cfg['kb'])[0],
                         G_ref_cloud=reqc.gap(R0, r0, rng, k=cfg['kb'])[0],
                         G_alt_cloud=reqc.gap(*cfg['alt'], rng, k=cfg['kb'])[0])
            r['G_alt'] = req.gap(*cfg['alt'], rng, k=cfg['kb'])[0]
            thr = Gp + cfg['eps']
            if G0 > thr:
                r['rho_star'] = bisect(lambda x: req.gap(1e9, x, rng, k=cfg['kb'])[0], r0, 1.0, thr, cfg['bis'])
                r['R_star'] = bisect(lambda x: req.gap(x, 1.0, rng, k=cfg['kb'])[0], 0.0, cfg['R_hi'], thr, cfg['bis'])
                if G_Rfree > thr:
                    validate = (case == '10') and (len(rows) % 4 == 0)
                    r.update(minimal_set(req, cfg, r0, thr, rng, P_Rfree, validate))
            r['solver_fallbacks'] = req.fallbacks + (reqc.fallbacks if not cfg.get('skip_cloud') else 0)
            rows.append(r)
            vecs.append(dict(case=case, hour=h, w=w, cut=cut, PD=PD, Pinc=Pinc.tolist(), Pfr=Pfr.tolist(),
                             Ft=Ft.tolist(), scale=sc.tolist(), scale_cloud=sc_cloud.tolist(), P_ref=P0.tolist()))
            if ckpt is not None:
                ckpt['emit'](r, grid[g0:], vecs[-1])
            print('  request', h, w, cut, round(time.time() - T0), flush=True)
    if CERT is not None:
        with open(f'{OUT}/certify_e{case}_h{h}.json', 'w') as f:
            json.dump([{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()} for r in CERT],
                      f, indent=1, default=str)
    return rows, grid, vecs


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--case', required=True, choices=sorted(CFG))
    ap.add_argument('--hours', required=True, help='comma-separated hours (EELD40: demand-level indices 1-6)')
    ap.add_argument('--tag', required=True, help='output suffix, e.g. e10a')
    ap.add_argument('--out', default=OUT, help='output directory (default: results/raw/attainability)')
    a = ap.parse_args(); cfg = CFG[a.case]; OUT = a.out
    dem = pd.read_csv(cfg['demand'])
    hours = [int(x) for x in a.hours.split(',')]
    bad = sorted(set(hours) - set(dem.hour.astype(int)))
    if bad:
        raise SystemExit(f'hours {bad} are not in {cfg["demand"]}')
    os.makedirs(OUT, exist_ok=True)
    guard_config(OUT, a.tag, dict(case=a.case, hours=a.hours,
                                  cfg={k: v for k, v in cfg.items() if k not in PATH_KEYS}))
    R_, G_, V_, t0 = [], [], [], time.time()
    done = set()
    if os.path.exists(f'{OUT}/requests_{a.tag}.csv'):          # resume after an interruption
        R_ = pd.read_csv(f'{OUT}/requests_{a.tag}.csv').to_dict('records')
        G_ = pd.read_csv(f'{OUT}/grid_{a.tag}.csv').to_dict('records')
        V_ = [json.loads(l) for l in open(f'{OUT}/vectors_{a.tag}.jsonl')]
        done = {int(r['hour']) for r in R_}
    if cfg.get('checkpoint'):
        # per-request checkpointing: rows are appended as soon as each request finishes
        fr, fg, fv = (f'{OUT}/requests_{a.tag}.csv', f'{OUT}/grid_{a.tag}.csv', f'{OUT}/vectors_{a.tag}.jsonl')
        def emit(r, g, v):
            prev_r = pd.read_csv(fr) if os.path.exists(fr) else None            # rewrite, so that every column survives
            pd.concat([x for x in (prev_r, pd.DataFrame([r])) if x is not None], ignore_index=True).to_csv(fr, index=False)
            pd.DataFrame(g).to_csv(fg, mode='a', header=not os.path.exists(fg), index=False)
            with open(fv, 'a') as f: f.write(json.dumps(v) + '\n')
        prev = pd.read_csv(fr) if os.path.exists(fr) else pd.DataFrame(columns=['hour', 'w', 'cut'])
        for h in [int(x) for x in a.hours.split(',')]:
            skip = {(float(x.w), float(x.cut)) for x in prev[prev.hour == h].itertuples()}
            if len(skip) == len(cfg['weights']) * len(cfg['cuts']):
                continue
            PD = float(dem.loc[dem.hour == h, 'PD'].iloc[0])
            run_hour(a.case, cfg, h, PD, seed=seed_hour(a.case, h), ckpt=dict(skip=skip, emit=emit))
            print(a.tag, h, round(time.time() - t0), flush=True)
        print(a.tag, 'DONE', flush=True)
        return
    for h in [int(x) for x in a.hours.split(',')]:
        if h in done:
            continue
        PD = float(dem.loc[dem.hour == h, 'PD'].iloc[0])
        r, g, v = run_hour(a.case, cfg, h, PD, seed=seed_hour(a.case, h))
        R_ += r; G_ += g; V_ += v
        pd.DataFrame(R_).to_csv(f'{OUT}/requests_{a.tag}.csv', index=False)
        pd.DataFrame(G_).to_csv(f'{OUT}/grid_{a.tag}.csv', index=False)
        with open(f'{OUT}/vectors_{a.tag}.jsonl', 'w') as f:
            for v_ in V_: f.write(json.dumps(v_) + '\n')
        print(a.tag, h, round(time.time() - t0), flush=True)
    print(a.tag, 'DONE', flush=True)


if __name__ == '__main__':
    main()
