"""Stylized operational subset on the 40-unit system (RQ9).

No published loss matrix or ramp table exists for the 40-unit valve-point benchmark (the benchmark
repository states that network losses are neglected). This experiment therefore uses two explicit,
documented modelling assumptions, written to the output, and must be reported as a stylized
sensitivity, not as benchmark data:

* Ramp rates. Each unit ramps at kappa_r * (Pmax_i - Pmin_i) MW/h, with kappa_r the median of the
  ramp-rate-to-span ratio of the 10-unit system (0.297 per hour).
* Losses. B40 = kappa_B * kron(I_4, B10): the 10-unit Kron matrix replicated over four blocks of
  ten units, scaled so that the losses of the lossless w = 0.5 incumbent at 10,000 MW equal the median
  incumbent loss share of the 10-unit study (3.03% of demand).

Requests: demand levels 8,000, 10,000 and 12,000 MW, weights 0.25 and 0.75, a 5% emission cut.
Reported per request: reference gap with and without losses (uniform band rho0, R0 = 200 MW),
physical gap, ramp-band gap at tau0 = 15 min, relief shares of budget and band, minimal response
time tau*, minimal budget R* at tau = 60 min, and the unit set S* under the ramp band with losses
(every single unit and every pair of band-active units, as in experiment X7). Runtimes are recorded
to estimate the cost of a full campaign.
Output: results/raw/supplementary/ops40.csv
"""
import argparse, copy, itertools, json, os, time
import numpy as np, pandas as pd
from eeldattain.paths import COEFFS, DEMAND, RAW_SUPP
from eeldattain.runio import guard_config, append_rows
from eeldops.data_loader import load_eeld10
from eeldops.model import OpsSystem
from eeldops.experiments import build_request, band_active, bisect, seed_of

LEVELS = (8000.0, 10000.0, 12000.0)
WEIGHTS = (0.25, 0.75)
CUT = 0.05
CFG = dict(K=40, kb=1, R0=200.0, rho0=0.05, tau0=15.0, tau_max=300.0, R_hi=3000.0, bis=6, eps=0.05,
           loss_share_target=0.0303, kappa_basis='median UR/DR over span, 10-unit system')


def stylized_system():
    """Stylized 40-unit loss model under the emission convention of the primary 40-unit campaign."""
    from eeldattain.runio import campaign_escale
    es = campaign_escale('40')
    coef10, B10, ramp10, _ = load_eeld10()
    c40 = pd.read_csv(COEFFS['40'])
    span10 = (coef10.Pmax - coef10.Pmin).to_numpy()
    rate10 = np.minimum(ramp10.UR_MW_per_h, ramp10.DR_MW_per_h).to_numpy()
    kappa_r = float(np.median(rate10 / span10))
    span40 = (c40.Pmax - c40.Pmin).to_numpy()
    ramp40 = pd.DataFrame(dict(UR_MW_per_h=kappa_r * span40, DR_MW_per_h=kappa_r * span40))
    Bshape = np.kron(np.eye(4), B10)
    sL = OpsSystem(c40, escale=es)
    rng = np.random.default_rng(seed_of(9, 100))
    A = sL.anchors(10000.0, rng, K=CFG['K'])
    P = sL.incumbent(10000.0, 0.5, A, K=CFG['K'])
    kappa_B = CFG['loss_share_target'] * 10000.0 / float(P @ Bshape @ P)
    sB = OpsSystem(c40, kappa_B * Bshape, ramp40, escale=es)
    sL = OpsSystem(c40, None, ramp40, escale=es)
    return sL, sB, dict(kappa_r=kappa_r, kappa_B=kappa_B, escale=es)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(RAW_SUPP))
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--full', action='store_true', help='all six demand levels and cuts of 5%% and 10%% (24 requests)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    levels = tuple(float(x) for x in pd.read_csv(DEMAND['40']).PD) if a.full else LEVELS
    cuts = (0.05, 0.10) if a.full else (CUT,)
    name = ('ops40full' if a.full else 'ops40') + f'_p{a.part}'
    cfg_rec = dict(CFG=CFG, levels=levels, weights=WEIGHTS, part=a.part, parts=a.parts)
    cfg_rec.update(cuts=cuts) if a.full else cfg_rec.update(cut=CUT)          # the 6-request record used 'cut'
    guard_config(a.out, name, cfg_rec)
    out = f'{a.out}/{name}.csv'
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out); done = set(zip(d.PD, d.w, d.cut))
    sL, sB, assumed = stylized_system()
    print('ASSUMPTIONS', assumed, flush=True)
    jobs = [(PD, w, c) for PD in levels for w in WEIGHTS for c in cuts][a.part::a.parts]
    cache = {}
    for PD, w, cut in jobs:
        if (PD, w, cut) in done:
            continue
        t0 = time.time(); sd = seed_of(9, int(PD / 100), w, cut); rng = np.random.default_rng(sd)
        row = dict(PD=PD, w=w, cut=cut, seed=sd, **assumed)
        for name, s in (('noloss', sL), ('loss', sB)):
            if (name, PD) not in cache:
                ra = np.random.default_rng(seed_of(9, int(PD / 100)))
                A = s.anchors(PD, ra, K=CFG['K'])
                cache[(name, PD)] = (A, {ww: s.incumbent(PD, ww, A, K=CFG['K']) for ww in WEIGHTS})
            A, INC = cache[(name, PD)]
            Pinc = INC[w]
            req, Pfr, Ft, reach, eta = build_request(s, PD, A, Pinc, cut, pool_cap=6)
            kb = CFG['kb']; R0 = CFG['R0']
            Gp, _ = req.gap(1e9, s.span, rng, k=kb)
            G0u, _ = req.gap(R0, s.band_uniform(CFG['rho0']), rng, k=kb)
            G0r, P0r = req.gap(R0, s.band_ramp(CFG['tau0']), rng, k=kb)
            G_Rfree, P_Rfree = req.gap(1e9, s.band_ramp(CFG['tau0']), rng, k=kb)
            G_bandfree, _ = req.gap(R0, s.span, rng, k=kb)
            thr = Gp + CFG['eps']
            r = dict(reachable=reach, eta_E=eta, loss_inc_pct=100 * s.loss(Pinc) / PD, G_phys=Gp, G_ref_uniform=G0u,
                     G_ref_ramp=G0r, G_Rfree_ramp=G_Rfree, G_bandfree=G_bandfree, tau_star=np.nan, R_star=np.nan,
                     relief_budget=np.nan, relief_band=np.nan)
            if G0r > thr:
                tot = G0r - Gp
                r['relief_budget'] = (G0r - G_Rfree) / tot; r['relief_band'] = (G0r - G_bandfree) / tot
                r['tau_star'] = bisect(lambda x: req.gap(1e9, s.band_ramp(x), rng, k=kb)[0], CFG['tau0'], CFG['tau_max'], thr, CFG['bis'])
                r['R_star'] = bisect(lambda x: req.gap(x, s.band_ramp(60.0), rng, k=kb)[0], 0.0, CFG['R_hi'], thr, CFG['bis'])
            if name == 'loss':
                r['S_class'] = ''
                if G0r > thr and G_Rfree > thr:
                    b0 = s.band_ramp(CFG['tau0']); act = band_active(req, P_Rfree, b0); r['n_band_active'] = len(act)
                    light = copy.copy(req); light.pool_cap = 2; light.fallbacks = 0

                    def G_with(S, warm=()):
                        b = b0.copy(); b[list(S)] = s.span[list(S)]
                        light.pool = [P_Rfree] + [x for x in warm if x is not None]; light.n_anchor = len(light.pool)
                        return light.gap(1e9, b, rng, k=1)
                    singles = {j: G_with([j]) for j in range(s.n)}
                    r['single_best'] = min(g for g, _ in singles.values())
                    if r['single_best'] <= thr:
                        r['S_class'] = '1'
                    else:
                        r['S_class'] = '>=3'; r['pairs_checked'] = 0
                        for S in itertools.combinations(act, 2):
                            r['pairs_checked'] += 1
                            if G_with(S, warm=[singles[S[0]][1], singles[S[1]][1]])[0] <= thr:
                                r['S_class'] = '2'; break
                    r['fallbacks_light'] = light.fallbacks
                elif G0r > thr:
                    r['S_class'] = '0'
            r['solver_fallbacks'] = req.fallbacks
            row.update({f'{k}_{name}': v for k, v in r.items()})
            with open(out.replace('.csv', '_vectors.jsonl'), 'a') as fh:          # everything needed to re-solve a cell
                fh.write(json.dumps(dict(PD=PD, w=w, cut=cut, model=name, Pinc=list(map(float, Pinc)), Pfr=list(map(float, Pfr)),
                                         Ft=list(map(float, Ft)), scale=list(map(float, req.sc)))) + '\n')
        row['runtime_s'] = time.time() - t0
        append_rows(out, [row])
        print(f'[ops40] PD={PD} w={w} cut={cut}: G_ref loss={row["G_ref_uniform_loss"]:.3f} noloss={row["G_ref_uniform_noloss"]:.3f} '
              f'tau*={row["tau_star_loss"]:.1f} S={row["S_class_loss"]} t={row["runtime_s"]:.0f}s', flush=True)
    print('DONE', time.strftime('%Y-%m-%dT%H:%M:%S'))


if __name__ == '__main__':
    main()
