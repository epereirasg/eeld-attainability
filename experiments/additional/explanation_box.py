"""Values of the operator explanation (Table 'Explanation of the request of Fig. 3').

Request: 10-unit system, hour 1, w = 0.5, 5% cut (the request of Fig. 3). Values from the stored campaign
row, the response-time counterfactual tau* recomputed on the lossless model with ramp-derived bands
b_i(tau) = min(r_up, r_dn) tau / 60 (bisection over [15, 300] min, eight steps, unrestricted budget), the
bounded emission reach at the reference allowance and the closing unit sets of the exhaustive search.
Output: results/raw/additional/explanation_box.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from eeldops.data_loader import load_eeld10

KEY = (1, 0.5, 0.05)


def main():
    s = System(str(COEFFS['10'])); Rq, vk = load_requests(); row = Rq[Rq.key == KEY].iloc[0]
    req = make_request(s, vk[KEY]); rng = np.random.default_rng(5)
    _, _, ramp, _ = load_eeld10(); r = np.minimum(ramp.UR_MW_per_h.values, ramp.DR_MW_per_h.values)
    thr = row.G_phys + EPS
    G_tau = lambda t: req.gap(BIG, np.minimum(1.0, r * t / 60 / s.span), rng)[0]
    lo, hi = 15.0, 300.0
    for _ in range(8):
        m = 0.5 * (lo + hi)
        if G_tau(m) <= thr: hi = m
        else: lo = m
    Eb, _ = emin_bounded(req, 50.0, 0.05)
    U = pd.concat([pd.read_csv(f) for f in sorted((RAW_ADD / 'unit_sets').glob('sets_p*.csv'))])
    u = U[(U.hour == KEY[0]) & (U.w == KEY[1]) & (U.cut == KEY[2]) & (U.G <= thr)]
    kmin = int(u.k.min())
    # what the gap counterfactuals deliver in the units of the request, and the compliance counterfactuals
    Einc = s.E(req.Pinc); Et = (1.0 - KEY[2]) * Einc          # the cap, which the target meets with equality here
    deliver = {}
    for nm, (R, rho) in (('rho_star', (BIG, row.rho_star)), ('R_star', (row.R_star, 1.0))):
        G, P = req.gap(R, rho, rng)
        deliver[nm] = dict(G=G, share_of_cut=(Einc - s.E(P)) / (Einc - Et), above_cap_pct=100 * (s.E(P) / Et - 1),
                           cost_vs_target_pct=100 * (s.C(P) / float(vk[KEY]['Ft'][0]) - 1))
    Cf = RAW_ADD / 'compliance'
    comp = {}
    if Cf.exists():
        Cq = pd.concat([pd.read_csv(f, dtype={'S_sets_E': str}) for f in sorted(Cf.glob('compliance_p*.csv'))])
        c = Cq[(Cq.hour == KEY[0]) & (Cq.w == KEY[1]) & (Cq.cut == KEY[2])]
        for e, g in c.groupby('eps_E'):
            g = g.iloc[0]
            comp[str(e)] = dict(need=bool(g.need_E), rho_star_E=g.get('rho_star_E'), R_star_E=g.get('R_star_E'),
                                tau_star_E=g.get('tau_star_E'), S_size_E=g.get('S_size_E'), S_sets_E=g.get('S_sets_E'))
    out = dict(request=KEY, reachable=bool(row.reachable), cost_increase_pct=row.dC_target_pct,
               mw_shifted_full=row.mw_full / 2, G_ref=row.G_ref, G_phys=row.G_phys, G_Rfree=row.G_Rfree,
               eta_b_ref=eta_bounded(req, Eb, Et), rho_star=row.rho_star, R_star=row.R_star,
               tau_star=hi, minimal_sets=sorted(u[u.k == kmin].S.tolist()), deliver=deliver, compliance=comp)
    json.dump(out, open(RAW_ADD / 'explanation_box.json', 'w'), indent=1, default=lambda x: None if x != x else float(x))
    print(out)


if __name__ == '__main__':
    main()
