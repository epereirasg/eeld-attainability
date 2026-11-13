"""Number macros of the additional experiments.

Reads results/raw/additional/ and writes results/macros/numbers_additional.tex and
results/processed/numbers_additional.json, plus processed CSV files used by Figs. 5 and 6.
Every quantity is computed from the stored main campaign (results/raw/attainability) and the additional
outputs. Requests are the 204 distinct 10-unit requests unless stated otherwise.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROC, MAC, load_case, boot_ci, paths

REV = str(paths.RAW / 'additional')
N = {}
EPS = 0.05
REAL = [(R, rho) for R in (25, 50, 100) for rho in (0.05, 0.10, 0.20)]


def fmt(x, d=2):
    return 'nan' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def cat(pattern, **kw):
    fs = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f, **kw) for f in fs], ignore_index=True) if fs else None


def stored():
    R, G = load_case('10')
    Ru = R[R.unique].copy()
    return Ru, G[G.unique & (G.kind == 'grid')].copy()


# ------------------------------------------------------------------------------ SCIP global certificate
def scip():
    D = cat(f'{REV}/scip/cells10_p*.csv')
    if D is None:
        return
    key = ['hour', 'w', 'cut', 'kind', 'R', 'rho']
    rt = f'{REV}/scip/retry.csv'
    N['ScipRetried'] = '0'
    if os.path.exists(rt):                      # long re-solves of cells that hit the 10 s limit
        T = pd.read_csv(rt)
        N['ScipRetried'] = str(len(T))
        M = D.merge(T[key + ['G_lb', 'G_ub', 'status', 'time_s']], on=key, how='left', suffixes=('', '_rt'))
        has = M.G_lb_rt.notna()
        M.loc[has, 'G_lb'] = np.fmax(M.loc[has, 'G_lb'], M.loc[has, 'G_lb_rt'])
        M.loc[has, 'G_ub'] = np.fmin(M.loc[has, 'G_ub'], M.loc[has, 'G_ub_rt'])
        M.loc[has, 'status'] = M.loc[has, 'status_rt']
        D = M[D.columns]
    # sanity check of the bounds: a valid lower bound never exceeds a value attained by a feasible dispatch
    N['ScipInvalid'] = str(int((D.G_lb - D.G_stored > 1e-3).sum()))
    D['dev'] = D.G_stored - D.G_lb               # > 0.01: the stored value may exceed the global optimum by more than 0.01
    D['excess'] = D.G_stored - D.G_ub            # > 0.01: SCIP found a point better than the campaign by more than 0.01
    N['ScipCells'] = f'{len(D):,}'.replace(',', '{,}')
    N['ScipReq'] = str(D[['hour', 'w', 'cut']].drop_duplicates().shape[0])
    N['ScipCert'] = f'{int((D.dev <= 0.01).sum()):,}'.replace(',', '{,}')
    pc = 100 * (D.dev <= 0.01).mean()
    N['ScipCertPct'] = fmt(pc, 1) if round(pc, 1) < 100 or pc == 100 else fmt(pc, 2)   # 99.97 must not print as 100.0
    N['ScipFail'] = str(int((D.dev > 0.01).sum()))
    better = D[D.excess > 0.01]
    N['ScipBetter'] = str(len(better))
    N['ScipBetterReq'] = str(better[['hour', 'w', 'cut']].drop_duplicates().shape[0])
    N['ScipBetterMax'] = fmt(better.excess.max(), 3) if len(better) else '0'
    N['ScipBetterKinds'] = ', '.join(sorted(set(better.kind))).replace('_', ' ')
    N['ScipOpen'] = str(int(((D.dev > 0.01) & (D.excess <= 0.01)).sum()))
    N['ScipTight'] = fmt(100 * (D.dev <= 1e-3).mean(), 1)
    N['ScipTimeMed'] = fmt(D.time_s.median(), 1)
    N['ScipTimeNinety'] = fmt(D.time_s.quantile(0.9), 1)
    D.to_csv(f'{PROC}/additional_scip_cells10.csv', index=False)
    fa = paths.RAW_CERT / 'scip_adopted.csv'
    if fa.exists():                              # SCIP values stored in the campaign (experiments/certificate/adopt_scip.py)
        Ad = pd.read_csv(fa); imp = Ad.G_campaign - Ad.G_stored_new
        N['ScipAdoptN'] = str(len(Ad)); N['ScipAdoptReq'] = str(Ad[['hour', 'w', 'cut']].drop_duplicates().shape[0])
        N['ScipAdoptMax'] = fmt(imp.max(), 3) if len(Ad) else '0'; N['ScipAdoptAboveTol'] = str(int((imp > 0.01).sum()))
        N['ScipAdoptKinds'] = ', '.join(sorted(set(Ad.kind))).replace('_', ' ')
    F = sorted(glob.glob(f'{REV}/scip/forty_cells_p*.csv'))
    if F:                                        # every reference and physical cell of the primary 40-unit campaign, 600 s
        Q = pd.concat([pd.read_csv(x) for x in F]); Q['dev'] = Q.G_stored - Q.G_lb; Q['exc'] = Q.G_stored - Q.G_ub
        N['ScipFortyN'] = str(len(Q)); N['ScipFortyCert'] = str(int((Q.dev <= 0.01).sum()))
        N['ScipFortyOpenN'] = str(int(((Q.dev > 0.01) & (Q.exc <= 0.01)).sum()))
        N['ScipFortyBetter'] = str(int((Q.exc > 0.01).sum())); N['ScipFortyBetterMax'] = fmt(max(0.0, Q.exc.max()), 3)
        for k, nm in (('reference', 'Ref'), ('physical', 'Phys')):
            q = Q[Q.kind == k]
            N[f'ScipForty{nm}N'] = str(len(q)); N[f'ScipForty{nm}Cert'] = str(int((q.dev <= 0.01).sum()))
            N[f'ScipForty{nm}GapMax'] = fmt(q.dev.max(), 3); N[f'ScipForty{nm}GapMed'] = fmt(q.dev.median(), 3)
        N['ScipFortyTimeMed'] = fmt(Q.time_s.median(), 0)
        Q.to_csv(f'{PROC}/additional_scip_forty_cells.csv', index=False)
    C = f'{REV}/scip/counterfactuals.csv'
    if os.path.exists(C):
        Q = pd.read_csv(C)
        # one bisection step below the stored value the gap is still open, or the stored value is the first
        # bisection step above rho_0 (then rho* = rho_0 itself, one step below the stored value)
        first = (Q.cf == 'rho') & (Q.value - 0.95 / 256 <= 0.05 + 1e-9)
        minimal = (Q.lb_below > Q.thr - 0.01) | first
        N['ScipCfFirstStep'] = str(int((first & ~(Q.lb_below > Q.thr - 0.01)).sum()))
        valid = Q.ub_at <= Q.thr + 0.01            # at the stored value a dispatch within the tolerance exists
        N['ScipCfN'] = str(len(Q)); N['ScipCfReq'] = str(Q[['hour', 'w', 'cut']].drop_duplicates().shape[0])
        N['ScipCfMinimal'] = str(int(minimal.sum())); N['ScipCfValid'] = str(int(valid.sum()))
        N['ScipCfBoth'] = str(int((minimal & valid).sum()))
        Q.assign(minimal=minimal, valid=valid).to_csv(f'{PROC}/additional_scip_counterfactuals.csv', index=False)
    O = f'{REV}/scip/ops_cells.csv'
    if os.path.exists(O):
        Q = pd.read_csv(O); dev = Q.G_stored - Q.G_lb; exc = Q.G_stored - Q.G_ub
        N['ScipOpsN'] = str(len(Q)); N['ScipOpsCert'] = str(int((dev <= 0.01).sum()))
        N['ScipOpsBetter'] = str(int((exc > 0.01).sum())); N['ScipOpsBetterMax'] = fmt(max(0.0, exc.max()), 3)
        N['ScipOpsOpen'] = str(int(((dev > 0.01) & (exc <= 0.01)).sum()))
        N['ScipOpsInvalid'] = str(int((Q.G_lb - Q.G_stored > 1e-3).sum()))
    print('SCIP', len(D), 'cells; uncertified', int((D.dev > 0.01).sum()), 'better', len(better), 'open', N['ScipOpen'])


# ------------------------------------------------------------------------------ exhaustive unit sets
def unit_sets(Ru):
    U = cat(f'{REV}/unit_sets/sets_p*.csv')
    if U is None:
        return
    key = ['hour', 'w', 'cut']
    done = U.groupby(key).size()
    done = done.index
    U = U.set_index(key).loc[done].reset_index()
    M = Ru.set_index(key).loc[done].reset_index()
    out = []
    for eps in (0.02, 0.05, 0.10):
        for r in M.itertuples():
            thr = r.G_phys + eps
            if r.G_ref <= thr:
                continue                                   # no counterfactual needed at this tolerance
            if r.G_Rfree <= thr:
                out.append(dict(hour=r.hour, w=r.w, cut=r.cut, eps=eps, size=0, n_min=1, sets='', stored=r.S_size))
                continue
            u = U[(U.hour == r.hour) & (U.w == r.w) & (U.cut == r.cut)]
            size, sets = 4, []
            for k in (1, 2, 3):
                ok = u[(u.k == k) & (u.G <= thr)]
                if len(ok):
                    size, sets = k, list(ok.S); break
            out.append(dict(hour=r.hour, w=r.w, cut=r.cut, eps=eps, size=size, n_min=len(sets), sets=';'.join(sets),
                            stored=r.S_size if eps == 0.05 else np.nan,
                            band_active_only=all(bool(x) for x in u[u.S.isin(sets)].band_active) if sets else np.nan))
    S = pd.DataFrame(out); S.to_csv(f'{PROC}/additional_unit_sets.csv', index=False)
    N['RevSetReq'] = str(len(M))
    for eps, T in ((0.02, 'Lo'), (0.05, ''), (0.10, 'Hi')):
        s = S[(S.eps == eps) & (S['size'] > 0)]
        N[f'RevSetNeed{T}'] = str(len(s))
        N[f'RevSetLeTwo{T}'] = fmt(100 * (s['size'] <= 2).mean(), 0)
        lo, hi = boot_ci(((s['size'] <= 2).astype(float)).values)
        N[f'RevSetLeTwoCiLo{T}'] = fmt(100 * lo, 0); N[f'RevSetLeTwoCiHi{T}'] = fmt(100 * hi, 0)
        for k, nm in ((1, 'One'), (2, 'Two'), (3, 'Three'), (4, 'Four')):
            N[f'RevSet{nm}{T}'] = str(int((s['size'] == k).sum()))
        N[f'RevSetBudgetOnly{T}'] = str(int(((S.eps == eps) & (S['size'] == 0)).sum()))
    s = S[(S.eps == 0.05) & (S['size'] > 0)]
    st = s.stored
    N['RevSetAgree'] = fmt(100 * (s['size'] == st).mean(), 0)
    N['RevSetSmaller'] = str(int((s['size'] < st).sum()))
    N['RevSetLarger'] = str(int((s['size'] > st).sum()))
    cl = s[s['size'] <= 3]
    N['RevSetOutsideBA'] = str(int((cl.band_active_only == False).sum()))  # noqa: E712
    N['RevDivMed'] = fmt(cl.n_min.median(), 0)
    N['RevDivMulti'] = fmt(100 * (cl.n_min >= 2).mean(), 0)
    N['RevDivMax'] = str(int(cl.n_min.max()))
    units = [u for x in cl.sets for st_ in x.split(';') for u in st_.split('-') if u]
    req_has = {str(i): np.mean([any(str(i) in st_.split('-') for st_ in x.split(';')) for x in cl.sets]) for i in range(1, 11)}
    top = max(req_has, key=req_has.get)
    N['RevTopUnit'] = top; N['RevTopUnitShare'] = fmt(100 * req_has[top], 0)
    pd.Series(req_has).to_csv(f'{PROC}/additional_unit_frequency.csv')
    print('unit sets', len(M), 'requests;', {k: v for k, v in N.items() if k.startswith('RevSet') or k.startswith('RevDiv')})


# ------------------------------------------------------------------------------ dense grid, reach, joint CF, sensitivity
def grid(Ru, Gst):
    Gd = cat(f'{REV}/allowance_grid/grid_p*.csv'); Rr = cat(f'{REV}/allowance_grid/requests_p*.csv')
    if Gd is None:
        return
    key = ['hour', 'w', 'cut']
    Gd['R'] = Gd.R.astype(float)
    # agreement with the stored campaign on the common cells
    C = Gd.merge(Gst[key + ['R', 'rho', 'G']], on=key + ['R', 'rho'], suffixes=('', '_st'))
    N['RevGridCommon'] = str(len(C))
    N['RevGridAgree'] = fmt(100 * ((C.G - C.G_st).abs() <= 0.01).mean(), 1)
    N['RevGridMaxDiff'] = fmt((C.G - C.G_st).abs().max(), 3)
    N['RevGridOff'] = str(int(((C.G - C.G_st).abs() > 0.01).sum()))
    N['RevGridNewBetter'] = str(int((C.G < C.G_st - 0.01).sum()))
    # best known value per cell (both are feasible upper bounds of the same minimum)
    Gd = Gd.merge(Gst[key + ['R', 'rho', 'G']].rename(columns={'G': 'G_st'}), on=key + ['R', 'rho'], how='left')
    Gd['G_best'] = np.fmin(Gd.G, Gd.G_st)
    Gd = Gd.merge(Ru[key + ['G_phys', 'G_ref', 'S_size']].rename(columns={'G_phys': 'G_phys_campaign'}), on=key)
    Gd['G_allow'] = (Gd.G_best - Gd.G_phys_campaign).clip(lower=0)
    Gd.to_csv(f'{PROC}/additional_dense_grid.csv', index=False)
    n = Gd[key].drop_duplicates().shape[0]
    N['RevGridReq'] = str(n)
    # bounded emission reach
    reg = Gd[[(R, r) in REAL for R, r in zip(Gd.R, Gd.rho)]]
    per = reg.groupby(key).agg(eta_b=('eta_b', 'mean'), eta_opt=('eta_opt', 'mean'))
    per = per.join(Ru.set_index(key)[['eta_E']])
    N['RevEtaBMed'] = fmt(100 * per.eta_b.median(), 0)
    N['RevEtaBQa'] = fmt(100 * per.eta_b.quantile(0.25), 0); N['RevEtaBQc'] = fmt(100 * per.eta_b.quantile(0.75), 0)
    N['RevEtaPhysMed'] = fmt(100 * per.eta_E.median(), 0)
    N['RevEtaBLossShare'] = fmt(100 * (per.eta_E - per.eta_b >= 0.10).mean(), 0)
    N['RevEtaOptMed'] = fmt(100 * per.eta_opt.median(), 0)
    ref = Gd[(Gd.R == 50) & (Gd.rho == 0.05)]
    N['RevEtaBRefMed'] = fmt(100 * ref.eta_b.median(), 0)
    N['RevEtaOptRefMed'] = fmt(100 * ref.eta_opt.median(), 0)
    full = Gd[np.isinf(Gd.R) & (Gd.rho == 1.0)]
    N['RevEtaCheck'] = fmt((full.eta_b.values - Ru.set_index(key).loc[[tuple(x) for x in full[key].values]].eta_E.values).__abs__().max(), 4)
    per.to_csv(f'{PROC}/additional_bounded_reach.csv')
    # joint counterfactual on the dense grid, requests that need a counterfactual (stored rule, eps = 0.05)
    need = Ru[Ru.G_ref > Ru.G_phys + EPS][key]
    J = Gd.merge(need, on=key)
    J['ok'] = J.G_best <= J.G_phys_campaign + EPS
    rows = []
    for k, g in J.groupby(key):
        ok = g[g.ok]
        pts = sorted(zip(ok.R, ok.rho))
        mins = [p for p in pts if not any((q[0] <= p[0] and q[1] <= p[1] and q != p) for q in pts)]
        real = any(R <= 100 and r <= 0.20 for R, r in mins)
        fin = any(np.isfinite(R) and r <= 0.20 for R, r in mins)
        rows.append(dict(hour=k[0], w=k[1], cut=k[2], n_min=len(mins), mins=';'.join(f'{R:g}/{r:g}' for R, r in mins),
                         in_realistic=real, in_finite_grid=fin))
    Jd = pd.DataFrame(rows); Jd.to_csv(f'{PROC}/additional_joint_cf.csv', index=False)
    N['RevJointN'] = str(len(Jd))
    N['RevJointReal'] = fmt(100 * Jd.in_realistic.mean(), 0)
    N['RevJointFinite'] = fmt(100 * Jd.in_finite_grid.mean(), 0)
    N['RevJointMultiPair'] = fmt(100 * (Jd.n_min >= 2).mean(), 0)
    # share of requests attained at each cell (all requests that need a counterfactual)
    att = J.groupby(['R', 'rho']).ok.mean().reset_index()
    att.to_csv(f'{PROC}/additional_attained_share.csv', index=False)
    for (R, rho), nm in (((100, 0.20), 'RealCorner'), ((np.inf, 0.20), 'RfreeTwenty'), ((250, 1.0), 'TwoFiftyFull')):
        v = att[(att.R == R) & (att.rho == rho)].ok
        N[f'RevAttain{nm}'] = fmt(100 * v.iloc[0], 0) if len(v) else 'nan'
    # sensitivity to eps, distance and budget norm
    if Rr is not None:
        Rr = Rr.merge(Ru[key + ['G_ref', 'G_phys', 'rho_star', 'R_star']].rename(columns={'G_ref': 'G_ref_c', 'G_phys': 'G_phys_c'}), on=key)
        for eps, T in ((0.02, 'Lo'), (0.10, 'Hi')):
            need_e = Rr.G_ref_c > Rr.G_phys_c + eps
            N[f'RevNeed{T}'] = str(int(need_e.sum()))
            N[f'RevRhoStarMed{T}'] = fmt(Rr[f'rho_star_{eps}'].median(), 3)
            N[f'RevRStarMed{T}'] = fmt(Rr[f'R_star_{eps}'].median(), 0)
        # distance: Chebyshev, budget: L2 with radius R/sqrt(2)
        for pre, nm in (('Ginf', 'Inf'), ('Gl2', 'Ltwo')):
            if pre == 'Ginf':
                gp = Rr.Ginf_phys
            else:
                gp = Rr.G_phys_c
            cols = [f'{pre}_{R}_{rho}' for R, rho in REAL]
            m = Rr[cols].sub(gp, axis=0).clip(lower=0).mean(axis=1)
            N[f'RevAllowShare{nm}'] = fmt(100 * (m >= 0.10).mean(), 1)
            lo, hi = boot_ci((m >= 0.10).astype(float).values)
            N[f'RevAllowShareLo{nm}'] = fmt(100 * lo, 1); N[f'RevAllowShareHi{nm}'] = fmt(100 * hi, 1)
        st = Gst.merge(Ru[key + ['G_phys']].rename(columns={'G_phys': 'gp'}), on=key)
        reg = st[[(R, r) in REAL for R, r in zip(st.R, st.rho)]]
        m2 = (reg.G - reg.gp).clip(lower=0).groupby([reg.hour, reg.w, reg.cut]).mean()
        mInf = Rr.set_index(key)[[f'Ginf_{R}_{rho}' for R, rho in REAL]].sub(Rr.set_index(key).Ginf_phys, axis=0).clip(lower=0).mean(axis=1)
        j = pd.concat([m2.rename('l2'), mInf.rename('inf')], axis=1).dropna()
        N['RevInfSpear'] = fmt(spearmanr(j.l2, j['inf'])[0])
        N['RevInfRefSpear'] = fmt(spearmanr(Rr.G_ref_c, Rr.Ginf_ref)[0])
        # runtimes
        N['RevTGapMed'] = fmt(Gd.t_gap.median(), 2); N['RevTGapNinety'] = fmt(Gd.t_gap.quantile(0.9), 2)
        N['RevTEminMed'] = fmt(1000 * Gd.t_emin.median(), 0)
        tr = Rr[[c for c in Rr.columns if c.startswith('t_rho_')]].stack(); tR = Rr[[c for c in Rr.columns if c.startswith('t_R_')]].stack()
        N['RevTRhoMed'] = fmt(tr.median(), 0); N['RevTRStarMed'] = fmt(tR.median(), 0)
        N['RevTRhoMax'] = fmt(tr.max(), 0); N['RevTRStarMax'] = fmt(tR.max(), 0)
    print('grid', {k: v for k, v in N.items() if k.startswith('RevGrid') or k.startswith('RevEta') or k.startswith('RevJoint') or k.startswith('RevAttain')})


# ------------------------------------------------------------------------------ model variants
def variants(Ru, Gst):
    key = ['hour', 'w', 'cut']
    odd = Ru[Ru.hour % 2 == 1]
    # main campaign restricted to the odd hours, as a reference for the variants
    def summary(R, G, T):
        R = R[R.duplicate_of_w.isna()] if 'duplicate_of_w' in R else R
        G = G.merge(R[key + ['inc']] if 'inc' in R else R[key], on=key)
        hw = R.groupby('hour').agg(W=('front_width', 'first'), eta=('eta_E', 'mean'))
        N[f'Var{T}N'] = str(len(R))
        N[f'Var{T}Reach'] = fmt(100 * R.reachable.mean(), 0)
        N[f'Var{T}Eta'] = fmt(R.eta_E.mean())
        N[f'Var{T}Spear'] = fmt(spearmanr(hw.W, hw.eta)[0]) if len(hw) > 3 else 'nan'
        reg = G[[(Rv, r) in REAL for Rv, r in zip(G.R, G.rho)]].copy()
        reg['ga'] = (reg.G - reg.G_phys).clip(lower=0)
        per = reg.groupby(key).ga.mean()
        N[f'Var{T}Allow'] = fmt(100 * (per >= 0.10).mean(), 0)
        lo, hi = boot_ci((per >= 0.10).astype(float).values)
        N[f'Var{T}AllowLo'] = fmt(100 * lo, 0); N[f'Var{T}AllowHi'] = fmt(100 * hi, 0)
        pg = G.pivot_table(index='R', columns='rho', values='G', aggfunc='mean')
        N[f'Var{T}Sat'] = fmt(max(pg.loc[100, r] - pg.loc[250, r] for r in pg.columns if r <= 0.10), 3)
        N[f'Var{T}BandGain'] = fmt(pg.loc[250, 0.10] - pg.loc[250, 0.20], 3)
        need = R[R.G_ref > R.G_phys + EPS]
        N[f'Var{T}Need'] = str(len(need))
        N[f'Var{T}Rho'] = fmt(need.rho_star.median(), 3); N[f'Var{T}RStar'] = fmt(need.R_star.median(), 0)
        if 'S_size' in need:
            s = need[need.S_size > 0] if T != 'Main' else need.dropna(subset=['S_size'])
            N[f'Var{T}SetN'] = str(len(s)); N[f'Var{T}SetLeTwo'] = fmt(100 * (s.S_size <= 2).mean(), 0)
        if 't_request' in R:
            N[f'Var{T}TReq'] = fmt(R.t_request.median(), 0); N[f'Var{T}TReqMax'] = fmt(R.t_request.max(), 0)
    Gm = Gst.merge(odd[key + ['G_phys']].rename(columns={'G_phys': 'gp'}), on=key)
    Gm['G_phys'] = Gm.gp
    summary(odd, Gm, 'Main')
    for v, T in (('leastcost', 'Lc'), ('perturbed', 'Pt'), ('convex', 'Cv'), ('reserve', 'Rs')):
        R = cat(f'{REV}/variants/requests_{v}.csv'); G = cat(f'{REV}/variants/grid_{v}.csv')
        if R is None or not len(R):
            continue
        summary(R, G, T)
        print('variant', v, {k: x for k, x in N.items() if k.startswith(f'Var{T}')})
    Re = cat(f'{REV}/variants/requests_leastcost_even.csv')
    if Re is not None and len(Re):             # least-cost incumbent on all 24 hours, against the main campaign on all hours
        R = pd.concat([cat(f'{REV}/variants/requests_leastcost.csv'), Re], ignore_index=True)
        G = pd.concat([cat(f'{REV}/variants/grid_leastcost.csv'), cat(f'{REV}/variants/grid_leastcost_even.csv')], ignore_index=True)
        summary(R, G, 'LcAll')
        Ga = Gst.merge(Ru[key + ['G_phys']].rename(columns={'G_phys': 'gp'}), on=key); Ga['G_phys'] = Ga.gp
        summary(Ru, Ga, 'MainAll')
        print('variant leastcost 24 h', {k: x for k, x in N.items() if k.startswith('VarLcAll') or k.startswith('VarMainAll')})
    done = [T for T in ('Lc', 'Pt', 'Cv', 'Rs') if f'Var{T}N' in N]
    if done:                                   # ranges over the model variants, for the text
        for key_, d in (('Allow', 0), ('SetLeTwo', 0), ('Sat', 3)):
            vals = [float(N[f'Var{T}{key_}']) for T in done]
            N[f'VarRange{key_}Min'] = fmt(min(vals), d); N[f'VarRange{key_}Max'] = fmt(max(vals), d)
        lows = [float(N[f'Var{T}AllowLo']) for T in done]
        N['VarRangeAllowLoMin'] = fmt(min(lows), 0)


def rq1(Ru):
    """Algebraic link behind RQ1: eta_E = min{1, kappa W_h E(P^C) / (c E_inc)}, kappa = incumbent position."""
    R = Ru.copy()
    EC = R.E_min / (1 - R.front_width)
    R['kappa'] = (R.E_inc - R.E_min) / (EC - R.E_min)
    N['RevIdentityErr'] = f"{(np.minimum(1, (R.E_inc - R.E_min) / (R.cut * R.E_inc)) - R.eta_E).abs().max():.1e}"
    for w, nm in ((0.25, 'Lo'), (0.5, 'Mid'), (0.75, 'Hi')):
        N[f'RevKappa{nm}'] = fmt(R[R.w == w].kappa.median())
    N['RevKappaMed'] = fmt(R.kappa.median())
    hw = R.groupby('hour').agg(W=('front_width', 'first'), k=('kappa', 'mean'), eta=('eta_E', 'mean'))
    N['RevSpearWKappa'] = fmt(spearmanr(hw.W, hw.k)[0])
    med = R.groupby('w').kappa.median()
    R['eta_fix'] = np.minimum(1, R.w.map(med) * R.front_width * EC / (R.cut * R.E_inc))
    N['RevSpearFixedKappa'] = fmt(spearmanr(hw.W, R.groupby('hour').eta_fix.mean())[0])


def forty():
    """40-unit allowance share on a region matched to the 10-unit realistic region relative to peak demand.

    The 10-unit realistic budgets 25 to 100 MW are 1.1% to 4.5% of the 2220 MW peak. Scaled to the 12000 MW
    peak of the 40-unit levels they become 135 to 540 MW, and the nearest grid budgets are 100, 250 and 500 MW.
    """
    R, G = load_case('40')
    Ru = R[R.unique]; G = G[G.unique & (G.kind == 'grid')].copy()
    G['ga'] = (G.G - G.G_phys).clip(lower=0)
    reg = G[G.R.isin([100, 250, 500]) & G.rho.isin([0.05, 0.10, 0.20])]
    per = reg.groupby(['hour', 'w', 'cut']).ga.mean()
    N['RevFortyMatchShare'] = fmt(100 * (per >= 0.10).mean(), 1)
    lo, hi = boot_ci((per >= 0.10).astype(float).values)
    N['RevFortyMatchLo'] = fmt(100 * lo, 1); N['RevFortyMatchHi'] = fmt(100 * hi, 1)
    N['RevFortyMatchN'] = str(len(per))


def rq2_parts():
    """All three parts of the RQ2 rule (docs/evaluation_criteria.md), for every system and model.

    (a) share of requests with a mean allowance gap of at least 0.10 over the realistic region (reported elsewhere),
    (b) the largest mean change in G* from 100 to 250 MW over the bands rho <= 0.10 (at most 0.02),
    (c) the share of counterfactual requests in which relaxing the budget alone and relaxing the band alone each
        remove less than 70% of the gap (more than half of the requests).
    """
    for case, T in (('10', 'Ten'), ('40', 'Forty')):
        R, G = load_case(case)
        Gu = G[G.unique & (G.kind == 'grid')]
        pg = Gu.pivot_table(index='R', columns='rho', values='G', aggfunc='mean')
        N[f'RqSat{T}'] = fmt(max(pg.loc[100, r] - pg.loc[250, r] for r in pg.columns if r <= 0.10 + 1e-9), 3)
        Ru = R[R.unique]; need = Ru[Ru.G_ref > Ru.G_phys + EPS]; tot = need.G_ref - need.G_phys
        j = ((need.G_ref - need.G_Rfree) / tot < 0.7) & ((need.G_ref - need.G_rhofree) / tot < 0.7)
        N[f'RqJoint{T}'] = fmt(100 * j.mean(), 0); N[f'RqJointN{T}'] = str(len(need))
    # loss model, uniform bands (RQ8)
    O = pd.read_csv(f'{paths.RAW_OPS}/ops_requests_a.csv', dtype={'S_units': str})
    OG = pd.read_csv(f'{paths.RAW_OPS}/ops_grid_a.csv')
    OG = OG.merge(O[['hour', 'w', 'cut', 'duplicate_of_w']].rename(columns={'duplicate_of_w': 'dup'}), on=['hour', 'w', 'cut'])
    OG = OG[OG.dup.isna() & (OG.band == 'uniform')]
    pg = OG.pivot_table(index='R', columns='level', values='G', aggfunc='mean')
    N['RqSatOpsUni'] = fmt(max(pg.loc[100, r] - pg.loc[250, r] for r in (0.05, 0.10)), 3)
    f = f'{REV}/ops_uniform_relief.csv'
    if os.path.exists(f):
        U = pd.read_csv(f); tot = U.G_ref_uniform - U.G_phys
        j = ((U.G_ref_uniform - U.G_Rfree_uniform) / tot < 0.7) & ((U.G_ref_uniform - U.G_bandfree) / tot < 0.7)
        N['RqJointOpsUni'] = fmt(100 * j.mean(), 0); N['RqJointNOpsUni'] = str(len(U))
        N['RqOpsUniResolveMax'] = fmt((U.G_ref_uniform_resolve - U.G_ref_uniform).abs().max(), 4)
    # 40-unit stylized loss model, ramp bands (RQ9, 40-unit): no budget grid, so part (b) cannot be evaluated
    XF = cat(f'{paths.RAW_SUPP}/ops40full_p*.csv', dtype={'S_class_loss': str})
    nd = XF[XF.G_ref_ramp_loss > XF.G_phys_loss + EPS]
    j = (nd.relief_budget_loss < 0.7) & (nd.relief_band_loss < 0.7)
    N['RqJointOpsForty'] = fmt(100 * j.mean(), 0); N['RqJointNOpsForty'] = str(len(nd))


def rq3_seeds():
    """Seed reproducibility of rho* and R* (RQ3 clause: within 0.004 and 8 MW)."""
    D = cat(f'{REV}/rq3_seeds_p*.csv')
    if D is None:
        return
    dr = (D.rho_star_new - D.rho_star).abs(); dR = (D.R_star_new - D.R_star).abs()
    ok = (dr <= 0.004 + 1e-9) & (dR <= 8 + 1e-9)
    N['SeedCfN'] = str(len(D)); N['SeedCfOk'] = str(int(ok.sum())); N['SeedCfOkPct'] = fmt(100 * ok.mean(), 1)
    N['SeedCfRhoMax'] = fmt(dr.max(), 4); N['SeedCfRMax'] = fmt(dR.max(), 1)
    N['SeedCfRhoExact'] = str(int((dr < 1e-9).sum())); N['SeedCfRExact'] = str(int((dR < 1e-9).sum()))
    N['SeedCfGphysMax'] = fmt((D.G_phys_new - D.G_phys).abs().max(), 4)


def cf_bounds():
    """Bisection decisions with propagated SCIP bounds (no tolerance): the stored counterfactual is certified sufficient
    when U(at) - L(phys) <= eps, and the step below is certified insufficient when L(below) - U(phys) > eps."""
    C = cat(f'{REV}/scip/counterfactuals.csv'); D = cat(f'{REV}/scip/cells10_p*.csv')
    if C is None or D is None:
        return
    P = D[D.kind == 'physical'].copy()
    P['G_ub'] = np.fmin(P.G_ub, P.G_stored)          # the stored value is attained by a feasible dispatch
    P = P[['hour', 'w', 'cut', 'G_lb', 'G_ub']].rename(columns={'G_lb': 'Lp', 'G_ub': 'Up'})
    M = C.merge(P, on=['hour', 'w', 'cut'])
    suff = M.ub_at - M.Lp <= EPS; ins = M.lb_below - M.Up > EPS
    first = (M.cf == 'rho') & (M.value - 0.95 / 256 <= 0.05 + 1e-9)
    N['CfbN'] = str(len(M)); N['CfbSuff'] = str(int(suff.sum())); N['CfbMinimal'] = str(int((suff & ins).sum()))
    N['CfbFirst'] = str(int((~ins & first).sum())); N['CfbOpen'] = str(int((~ins & ~first).sum()))
    # rho0 itself certified sufficient: the band counterfactual is rho0 (set in analysis/attainability.py), minimal by definition
    suff0 = (M.cf == 'rho') & first & (M.ub_below - M.Lp <= EPS)
    N['CfbRhoZeroCert'] = str(int(suff0.sum())); N['CfbMinimalAll'] = str(int((suff & ins).sum() + suff0.sum()))


def direct_cf():
    """Band and budget counterfactuals by direct optimization (experiments/additional/direct_cf.py) against bisection."""
    f = f'{REV}/direct_cf.csv'
    if not os.path.exists(f):
        return
    X = pd.read_csv(f)
    N['DirN'] = str(len(X)); N['DirFail'] = str(int(X.R_direct.isna().sum() + X.rho_direct.isna().sum()))
    X = X.dropna()
    dR = X.R_direct - X.R_star; dr = X.rho_direct - X.rho_star
    N['DirRMedDiff'] = fmt(dR.median(), 1); N['DirRWithin'] = fmt(100 * (dR.abs() <= 1000 / 256 + 1e-9).mean(), 0)
    N['DirRhoMedDiff'] = fmt(dr.median(), 4); N['DirRhoWithin'] = fmt(100 * (dr.abs() <= 0.95 / 256 + 1e-9).mean(), 0)
    N['DirRAbove'] = str(int((dR > 1000 / 256).sum())); N['DirRhoAbove'] = str(int((dr > 0.95 / 256).sum()))
    N['DirTR'] = fmt(X.t_R.median(), 1); N['DirTRho'] = fmt(X.t_rho.median(), 1)


def runtimes():
    """Wall-clock time of the stored operational pipeline per request (gaps, counterfactuals and unit sets)."""
    O = cat(f'{paths.RAW_OPS}/ops_requests_a.csv')
    N['RevOpsTReqMed'] = fmt(O.runtime_s.median(), 0); N['RevOpsTReqMax'] = fmt(O.runtime_s.max(), 0)


def main():
    Ru, Gst = stored()
    runtimes(); forty(); rq2_parts(); rq3_seeds(); cf_bounds(); direct_cf()
    rq1(Ru); scip(); unit_sets(Ru); grid(Ru, Gst); variants(Ru, Gst)
    with open(f'{MAC}/numbers_additional.tex', 'w') as f:
        for k, v in sorted(N.items()):
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(N, open(f'{PROC}/numbers_additional.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
