"""Operational-extension experiments X1 to X4 and X7 (losses, ramp-derived bands, three-hour window, seeds, 40-unit sets).

Every request is appended to its CSV as soon as it finishes (restart-safe). Seeds are derived from
BASE_SEED, the experiment id, the hour, the weight and the cut, and are written to every row.

Seed note: ``seed_of`` is a fixed arithmetic rule, kept unchanged so that the stored runs remain
reproducible. Within one experiment it maps a few distinct (hour, weight) pairs to the same seed
(for example hour 8 with w = 0.25 and hour 3 with w = 0.75). Each such pair still uses its own
deterministic stream, so results are reproducible, but the random starts of those requests are
not statistically independent. ``tests/test_pipeline.py`` checks the known collisions.
"""
import copy, itertools, json, os, time
import numpy as np, pandas as pd
from eeldattain.paths import RAW_OPS, RAW_ATTAIN, COEFFS
from eeldattain.runio import append_rows, load_cache, save_cache
from eeldops.data_loader import load_eeld10
from eeldops.model import OpsSystem, OpsRequest, ramp_cons_vec, window_solve

RES = str(RAW_OPS)          # output directory, overridden by experiments/operational/run.py --out
ATT = str(RAW_ATTAIN)       # stored EELD40 attainability vectors (input of run_sets40)
BASE_SEED = 20261023
OPS = dict(weights=(0.25, 0.5, 0.75), cuts=(0.05, 0.10), K=100, kb=4, eps=0.05,
           R_grid=(25, 50, 100, 250), rho_grid=(0.05, 0.10, 0.20), tau_grid=(10, 15, 30, 60),
           R0=50.0, rho0=0.05, tau0=15.0, tau_max=300.0, R_hi=1000.0, bis=8, set_max=3)


def seed_of(exp, h, w=0.0, cut=0.0, rep=0):
    return int(BASE_SEED + 100000 * exp + 1000 * h + int(1000 * w) * 10 + int(1000 * cut) + 7 * rep)


def append(path, rows):
    append_rows(path, rows)


def done_keys(path, keys):
    if not os.path.exists(path): return set()
    d = pd.read_csv(path)
    return set(tuple(x) for x in d[keys].itertuples(index=False, name=None))


def bisect(fun, lo, hi, thr, steps):
    if fun(hi) > thr: return np.nan
    for _ in range(steps):
        m = 0.5 * (lo + hi)
        if fun(m) <= thr: hi = m
        else: lo = m
    return hi


def system():
    coef, B, ramp, dem = load_eeld10()
    return OpsSystem(coef, B, ramp), OpsSystem(coef), dem


CERTIFY = os.environ.get('EELD_CERTIFY', '1') == '1'   # SCIP check of anchors, incumbents and targets (globalopt)


def certify_ops_anchors(s, PD, A, sweep=9):
    """SCIP check of the loss-model anchors, with the front sweep rebuilt as in OpsSystem.anchors when one changes."""
    from eeldattain import globalopt
    Pc = globalopt.improve(s, PD, 'C', A['Pc']); Pe = globalopt.improve(s, PD, 'E', A['Pe'])
    if Pc is A['Pc'] and Pe is A['Pe']:
        return A
    ideal = np.array([s.C(Pc), s.E(Pe)]); nadir = np.array([s.C(Pe), s.E(Pc)])
    front, prev = [Pc], Pc
    for t in np.linspace(0, 1, sweep + 2)[1:-1]:
        P = s.eps_point(PD, nadir[1] - t * (nadir[1] - ideal[1]), [prev, Pe] + A['x0'][:4])
        if P is not None:
            front.append(P); prev = P
    front.append(Pe)
    return dict(Pc=Pc, Pe=Pe, ideal=ideal, nadir=nadir, front=front, x0=A['x0'])


def hour_setup(s, h, PD, tag):
    """Anchors and incumbents with losses, cached per hour so that a restart does not recompute them."""
    os.makedirs(f'{RES}/cache', exist_ok=True)
    f = f'{RES}/cache/{tag}_h{h}.json'
    if os.path.exists(f): return load_cache(f)
    rng = np.random.default_rng(seed_of(1, h))
    A = s.anchors(PD, rng, K=OPS['K'])
    if CERTIFY:
        A = certify_ops_anchors(s, PD, A)
    INC = {w: s.incumbent(PD, w, A, K=OPS['K']) for w in OPS['weights']}
    if CERTIFY:
        from eeldattain import globalopt
        INC = {w: globalopt.improve(s, PD, 'W', P, w=w, ideal=A['ideal'], nadir=A['nadir']) for w, P in INC.items()}
    save_cache(f, (A, INC))
    return A, INC


def build_request(s, PD, A, Pinc, cut, pool_cap=15):
    Finc = s.F(Pinc); Et = (1 - cut) * Finc[1]; Emin = A['ideal'][1]
    reach = bool(Et >= Emin - 1e-9)
    if reach:
        Pfr = s.eps_point(PD, Et, [Pinc, A['Pe']] + A['front'])
        if CERTIFY:
            from eeldattain import globalopt
            Pfr = globalopt.improve(s, PD, 'C', Pfr, Ecap=Et)
        Ft = s.F(Pfr)
    else:
        Pfr = A['Pe']; Ft = np.array([s.C(A['Pe']), Et])
    sc = np.maximum(np.abs(Ft - Finc), 1e-6 * np.abs(Finc))
    req = OpsRequest(s, PD, Pinc, Ft, sc, [Pfr, A['Pc'], A['Pe']], pool_cap=pool_cap)
    eta = min(1.0, max(0.0, (Finc[1] - Emin) / (Finc[1] - Et)))
    return req, Pfr, Ft, reach, eta


def band_active(req, P, band_mw):
    lo, hi = req.box(band_mw); s = req.s
    return [i for i in range(s.n) if min(P[i] - lo[i], hi[i] - P[i]) < 1e-3 and (hi[i] < s.hi[i] - 1e-6 or lo[i] > s.lo[i] + 1e-6)]


# ---------------------------------------------------------------- X1 + X2: losses and ramp-derived bands
def run_ops(hours, tag):
    s, _, dem = system()
    out = f'{RES}/ops_requests_{tag}.csv'; gout = f'{RES}/ops_grid_{tag}.csv'
    done = done_keys(out, ['hour', 'w', 'cut'])
    for h in hours:
        PD = float(dem.loc[dem.hour == h, 'PD'].iloc[0])
        A, INC = hour_setup(s, h, PD, 'ops')
        W = (A['nadir'][1] - A['ideal'][1]) / A['nadir'][1]
        seen = {}
        for w in OPS['weights']:
            Pinc = INC[w]; key = tuple(np.round(Pinc, 4)); dup = seen.get(key); seen.setdefault(key, w)
            for cut in OPS['cuts']:
                if (h, w, cut) in done: continue
                t0 = time.time(); sd = seed_of(2, h, w, cut); rng = np.random.default_rng(sd)
                light = None
                req, Pfr, Ft, reach, eta = build_request(s, PD, A, Pinc, cut)
                kb = OPS['kb']; full = s.span
                Gp, _ = req.gap(1e9, full, rng, k=kb)
                base = dict(hour=h, PD=PD, w=w, cut=cut, seed=sd, duplicate_of_w=dup, reachable=reach, eta_E=eta,
                            front_width=W, E_inc=req.Finc[1], C_inc=req.Finc[0], loss_inc=s.loss(Pinc),
                            loss_inc_pct=100 * s.loss(Pinc) / PD, E_min=A['ideal'][1], G_phys=Gp,
                            dC_target_pct=100 * (Ft[0] / req.Finc[0] - 1),
                            mw_full=float(np.abs(Pfr - Pinc).sum()) if reach else np.nan)
                grid = []
                for R in OPS['R_grid']:
                    for rho in OPS['rho_grid']:
                        G, P = req.gap(R, s.band_uniform(rho), rng, k=kb)
                        grid.append(dict(base, band='uniform', R=R, level=rho, G=G, mw=float(np.abs(P - Pinc).sum())))
                for R in list(OPS['R_grid']) + [1e9]:
                    for tau in OPS['tau_grid']:
                        G, P = req.gap(R, s.band_ramp(tau), rng, k=kb)
                        grid.append(dict(base, band='ramp', R=R, level=tau, G=G, mw=float(np.abs(P - Pinc).sum())))
                thr = Gp + OPS['eps']; R0, tau0, rho0 = OPS['R0'], OPS['tau0'], OPS['rho0']
                G0u, _ = req.gap(R0, s.band_uniform(rho0), rng, k=kb)
                G0r, P0r = req.gap(R0, s.band_ramp(tau0), rng, k=kb)
                G_Rfree, P_Rfree = req.gap(1e9, s.band_ramp(tau0), rng, k=kb)
                G_tfree, _ = req.gap(R0, full, rng, k=kb)
                Peff = s.eps_point(PD, s.E(P0r), [P0r, A['Pe']] + A['front'])
                r = dict(base, G_ref_uniform=G0u, G_ref_ramp=G0r, G_Rfree_ramp=G_Rfree, G_bandfree=G_tfree,
                         ref_mw=float(np.abs(P0r - Pinc).sum()), ref_units_moved=int(np.sum(np.abs(P0r - Pinc) > 0.5)),
                         ref_loss=s.loss(P0r), ref_eff_loss_pct=(100 * (s.C(P0r) / s.C(Peff) - 1)) if Peff is not None else np.nan,
                         rho_star=np.nan, tau_star=np.nan, R_star=np.nan, S_size=np.nan, S_units='', S_exact_check=np.nan,
                         S_validated=False, n_band_active=np.nan)
                if G0u > thr:
                    r['rho_star'] = bisect(lambda x: req.gap(1e9, s.band_uniform(x), rng, k=kb)[0], rho0, 1.0, thr, OPS['bis'])
                if G0r > thr:
                    r['tau_star'] = bisect(lambda x: req.gap(1e9, s.band_ramp(x), rng, k=kb)[0], tau0, OPS['tau_max'], thr, OPS['bis'])
                    r['R_star'] = bisect(lambda x: req.gap(x, s.band_ramp(60.0), rng, k=kb)[0], 0.0, OPS['R_hi'], thr, OPS['bis'])
                    if G_Rfree > thr:
                        b0 = s.band_ramp(tau0); act = band_active(req, P_Rfree, b0); r['n_band_active'] = len(act)
                        light = copy.copy(req); light.pool = [Pfr, P_Rfree]; light.n_anchor = 2; light.pool_cap = 2; light.fallbacks = 0
                        def G_with(S):                       # warm-started light multistart (incumbent, 1 random, Pfr, P_Rfree)
                            b = b0.copy(); b[list(S)] = full[list(S)]
                            return light.gap(1e9, b, rng, k=1)[0]
                        for k in range(1, OPS['set_max'] + 1):
                            hit = next((S for S in itertools.combinations(act, k) if G_with(S) <= thr), None)
                            if hit is not None:
                                r.update(S_size=k, S_units='-'.join(str(i + 1) for i in hit)); break
                        if np.isnan(r['S_size']): r['S_size'] = OPS['set_max'] + 1
                        if (h + int(10 * w) + int(100 * cut)) % 3 == 0:          # validation on about a third of requests
                            r['S_validated'] = True
                            for k in (1, 2):
                                if any(G_with(S) <= thr for S in itertools.combinations(range(s.n), k)):
                                    r['S_exact_check'] = k; break
                r['runtime_s'] = time.time() - t0
                r['solver_fallbacks'] = req.fallbacks + (light.fallbacks if light is not None else 0)
                append(gout, grid); append(out, [r])
                with open(f'{RES}/ops_vectors_{tag}.jsonl', 'a') as fh:
                    fh.write(json.dumps(dict(hour=h, w=w, cut=cut, PD=PD, Pinc=Pinc.tolist(), Pfr=Pfr.tolist(),
                                             Ft=Ft.tolist(), scale=req.sc.tolist(), P_ref=P0r.tolist())) + '\n')
                print(f'[ops {tag}] h={h} w={w} cut={cut} G_phys={Gp:.3f} G_ref_ramp={G0r:.3f} S={r["S_size"]} t={r["runtime_s"]:.0f}s', flush=True)


# ---------------------------------------------------------------- X3: three-hour ramp window
def run_window(hours, tag):
    s, _, dem = system()
    out = f'{RES}/window_{tag}.csv'; done = done_keys(out, ['hour', 'w', 'cut'])
    n = s.n; R0, tau0 = OPS['R0'], OPS['tau0']
    for h in hours:
        if h < 2 or h > 23: continue
        PDs = [float(dem.loc[dem.hour == t, 'PD'].iloc[0]) for t in (h - 1, h, h + 1)]
        As, INCs = zip(*[hour_setup(s, t, PD, 'ops') for t, PD in zip((h - 1, h, h + 1), PDs)])
        for w in OPS['weights']:
            if all((h, w, c) in done for c in OPS['cuts']): continue
            rng = np.random.default_rng(seed_of(3, h, w))
            # ramp-feasible efficient incumbent schedule for the window
            fs, gs = zip(*[s.weighted(w, A['ideal'], A['nadir']) for A in As])
            starts = [np.concatenate([INCs[t][w] for t in range(3)])]
            for _ in range(5):
                starts.append(np.concatenate([s.rand_feasible(PD, rng) for PD in PDs]))
            boxes = [(s.lo, s.hi)] * 3
            sol = window_solve(s, PDs, fs, gs, starts, boxes)
            if sol is None:
                print(f'[window {tag}] h={h} w={w} no ramp-feasible incumbent schedule', flush=True); continue
            X = sol[1]; Pm, Ph, Pp = X[:n], X[n:2 * n], X[2 * n:]
            for cut in OPS['cuts']:
                if (h, w, cut) in done: continue
                t0 = time.time(); sd = seed_of(4, h, w, cut); rng = np.random.default_rng(sd)
                req, Pfr, Ft, reach, eta = build_request(s, PDs[1], As[1], Ph, cut)
                band = s.band_ramp(tau0)
                Ga, _ = req.gap(R0, band, rng, k=OPS['kb'])                      # single hour
                Gpa, _ = req.gap(1e9, s.span, rng, k=OPS['kb'])
                req.extra_cons = ramp_cons_vec(s, Pm) + [{'type': 'ineq', 'fun': (lambda P: s.rate - (Pp - P)), 'jac': (lambda P: np.eye(n))},
                                                         {'type': 'ineq', 'fun': (lambda P: s.rate - (P - Pp)), 'jac': (lambda P: -np.eye(n))}]
                Gb, _ = req.gap(R0, band, rng, k=OPS['kb'])                      # neighbours fixed
                Gpb, _ = req.gap(1e9, s.span, rng, k=OPS['kb'])
                req.extra_cons = []
                # neighbours adjustable within the same per-hour allowance; target only at hour h
                lo_m, hi_m = np.maximum(s.lo, Pm - band), np.minimum(s.hi, Pm + band)
                lo_h, hi_h = req.box(band); lo_p, hi_p = np.maximum(s.lo, Pp - band), np.minimum(s.hi, Pp + band)
                budget = []
                for t, Pref in enumerate((Pm, Ph, Pp)):
                    budget.append({'type': 'ineq', 'fun': (lambda Y, t=t, Pref=Pref: R0 - np.abs(Y[t * n:(t + 1) * n] - Pref).sum()),
                                   'jac': (lambda Y, t=t, Pref=Pref: np.concatenate([-np.sign(Y[u * n:(u + 1) * n] - Pref) if u == t else np.zeros(n) for u in range(3)]))})
                st = [np.concatenate([Pm, P, Pp]) for P in req.pool[-4:]] + [X.copy()]
                solc = window_solve(s, PDs, [None, req.d, None], [None, req.grad, None], st,
                                    [(lo_m, hi_m), (lo_h, hi_h), (lo_p, hi_p)], extra=budget)
                Gc = min(solc[0], req.e0) / req.e0 if solc is not None else np.nan
                append(out, [dict(hour=h, w=w, cut=cut, seed=sd, reachable=reach, eta_E=eta, G_single=Ga, G_fixed=Gb,
                                  G_window=Gc, G_phys_single=Gpa, G_phys_fixed=Gpb, runtime_s=time.time() - t0,
                                  solver_fallbacks=req.fallbacks, window_solve_failed=solc is None)])
                print(f'[window {tag}] h={h} w={w} cut={cut} single={Ga:.3f} fixed={Gb:.3f} window={Gc:.3f}', flush=True)


# ---------------------------------------------------------------- X4: solver-seed robustness
SEED_HOURS = (2, 5, 8, 11, 14, 17, 20, 23)


def run_seeds(tag, hours=SEED_HOURS):
    s, _, dem = system()
    out = f'{RES}/seeds_{tag}.csv'; done = done_keys(out, ['hour', 'cut', 'rep'])
    vec = {}
    for src in dict.fromkeys([str(RAW_OPS), RES]):        # stored X1 vectors, overridden by vectors of this run
        for f in sorted(x for x in os.listdir(src) if x.startswith('ops_vectors_')):
            for line in open(f'{src}/{f}'):
                v = json.loads(line); vec[(v['hour'], v['w'], v['cut'])] = v
    for h in hours:
        for cut in OPS['cuts']:
            v = vec.get((h, 0.5, cut))
            if v is None:
                print(f'[seeds {tag}] h={h} cut={cut}: no X1 vector, skipped', flush=True); continue
            for rep in (1, 2, 3):
                if (h, cut, rep) in done: continue
                sd = seed_of(5, h, 0.5, cut, rep); rng = np.random.default_rng(sd)
                req = OpsRequest(s, v['PD'], v['Pinc'], v['Ft'], v['scale'], [v['Pfr']], pool_cap=8)
                rows = []
                for R in OPS['R_grid']:
                    for tau in OPS['tau_grid']:
                        rows.append(dict(hour=h, cut=cut, rep=rep, seed=sd, R=R, tau=tau, G=req.gap(R, s.band_ramp(tau), rng, k=OPS['kb'])[0]))
                append(out, rows)
                print(f'[seeds {tag}] h={h} cut={cut} rep={rep}', flush=True)


# ---------------------------------------------------------------- X7: exhaustive unit sets on EELD40
def run_sets40(tag, part=0, parts=1):
    from eeldattain.model import System, Request
    from eeldattain.runio import campaign_escale
    s = System(str(COEFFS['40']), escale=campaign_escale('40', ATT))
    V = []
    for f in sorted(os.listdir(ATT)):
        if f.startswith('vectors_e40'):
            V += [json.loads(l) for l in open(f'{ATT}/{f}')]
    R40 = pd.concat([pd.read_csv(f'{ATT}/{f}') for f in sorted(os.listdir(ATT)) if f.startswith('requests_e40')])
    out = f'{RES}/sets40_{tag}_p{part}.csv'; done = done_keys(out, ['PD', 'w', 'cut'])
    for i, v in enumerate(V):
        if i % parts != part or (v['PD'], v['w'], v['cut']) in done: continue
        r = R40[(R40.PD == v['PD']) & (R40.w == v['w']) & (R40.cut == v['cut'])].iloc[0]
        t0 = time.time(); sd = seed_of(7, int(v['PD'] / 100), v['w'], v['cut']); rng = np.random.default_rng(sd)
        req = Request(s, v['PD'], np.array(v['Pinc']), np.array(v['Ft']), np.array(v['scale']), np.array(v['Pfr']), None, None, pool_cap=8)
        rho0 = 0.05; thr = r.G_phys + 0.05
        G_rf, P_rf = req.gap(1e9, rho0, rng, k=4)                 # stronger re-solve than the campaign (k=1)
        row = dict(PD=v['PD'], w=v['w'], cut=v['cut'], seed=sd, G_phys=r.G_phys, G_Rfree_campaign=r.G_Rfree, G_Rfree=G_rf,
                   needs_set=bool(G_rf > thr), n_band_active=np.nan, single_best=np.nan, single_close=False, pair_close=False,
                   pairs_checked=0, exact_size='0' if G_rf <= thr else '')
        if G_rf > thr:
            lo0, hi0 = req.band(rho0)
            act = [j for j in range(s.n) if min(P_rf[j] - lo0[j], hi0[j] - P_rf[j]) < 1e-3 and (hi0[j] < s.hi[j] - 1e-6 or lo0[j] > s.lo[j] + 1e-6)]
            row['n_band_active'] = len(act)
            light = copy.copy(req); light.pool_cap = 2; light.fallbacks = 0
            def G_with(S, warm=()):                  # incumbent, one random start, P_Rfree and warm starts
                rv = np.full(s.n, rho0); rv[list(S)] = 1.0
                light.pool = [P_rf] + [w_ for w_ in warm if w_ is not None]
                return light.gap(1e9, rv, rng, k=1)
            singles = {j: G_with([j]) for j in range(s.n)}            # every unit, not only band-active ones
            row['single_best'] = min(g for g, _ in singles.values()); row['single_close'] = bool(row['single_best'] <= thr)
            if row['single_close']:
                row['exact_size'] = '1'
            else:
                for S in itertools.combinations(act, 2):
                    row['pairs_checked'] += 1
                    if G_with(S, warm=[singles[S[0]][1], singles[S[1]][1]])[0] <= thr:
                        row['pair_close'] = True; row['exact_size'] = '2'; break
                if not row['pair_close']: row['exact_size'] = '>=3'
        row['runtime_s'] = time.time() - t0
        row['solver_fallbacks'] = req.fallbacks + (light.fallbacks if G_rf > thr else 0)
        append(out, [row])
        print(f'[sets40 {tag}] PD={v["PD"]} w={v["w"]} cut={v["cut"]} size={row["exact_size"]} t={row["runtime_s"]:.0f}s', flush=True)
