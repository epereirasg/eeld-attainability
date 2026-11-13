"""RQ1, RQ2, RQ3, RQ7 and the 40-unit replication: number macros and processed files."""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr
from common import *

N = {}          # macro name -> formatted value


def fmt(x, d=2):
    return f'{x:.{d}f}'


def realistic(G, case):
    if case == '10':
        return G[(G.kind == 'grid') & G.R.isin([25, 50, 100]) & G.rho.isin([0.05, 0.10, 0.20])]
    # 40-unit region matched to the 10-unit one relative to peak demand (1.1% to 4.5% of 12000 MW, nearest grid budgets)
    return G[(G.kind == 'grid') & G.R.isin([100, 250, 500]) & G.rho.isin([0.05, 0.10, 0.20])]


def analyse(case):
    R, G = load_case(case)
    T = 'Ten' if case == '10' else 'Forty'
    Ru = R[R.unique].copy(); Gu = G[G.unique & (G.kind == 'grid')].copy()
    Gu['G_allow'] = (Gu.G - Gu.G_phys).clip(lower=0)
    N[f'NumReq{T}'] = str(len(R)); N[f'NumReq{T}Unique'] = str(len(Ru))
    # RQ1 physical reach
    hw = Ru.groupby('hour').agg(W=('front_width', 'first'), eta=('eta_E', 'mean'), PD=('PD', 'first'))
    rho_s, _ = spearmanr(hw.W, hw.eta)
    N[f'Spearman{T}'] = fmt(rho_s)
    pk = hw.PD.idxmax()
    N[f'WidthPeak{T}'] = fmt(100 * hw.W.loc[pk], 1); N[f'PeakPD{T}'] = f'{hw.PD.loc[pk]:.0f}'
    up = Ru[Ru.hour == pk]; EC = up.E_min / (1 - up.front_width)
    N[f'PeakKappa{T}'] = fmt(((up.E_inc - up.E_min) / (EC - up.E_min)).median(), 2)
    N[f'PeakIncDistinct{T}'] = str(up.drop_duplicates('E_inc').shape[0])
    if 'front_width_stored' in Ru:                 # stored anchors, kept as a sensitivity check
        ws = Ru.groupby('hour').front_width_stored.first()
        N[f'Spearman{T}Stored'] = fmt(spearmanr(ws, hw.eta)[0])
        N[f'WidthPeak{T}Stored'] = fmt(100 * ws.loc[hw.PD.idxmax()], 1); N[f'WidthPeak{T}'] = fmt(100 * hw.W.loc[hw.PD.idxmax()], 1)
    N[f'WidthMin{T}'] = fmt(100 * hw.W.min(), 1); N[f'WidthMax{T}'] = fmt(100 * hw.W.max(), 1)
    N[f'WidthMinPD{T}'] = f'{hw.loc[hw.W.idxmin(), "PD"]:.0f}'; N[f'WidthMaxPD{T}'] = f'{hw.loc[hw.W.idxmax(), "PD"]:.0f}'
    for c, nm in [(0.025, 'A'), (0.05, 'B'), (0.10, 'C')]:
        x = Ru[Ru.cut == c]
        N[f'Reach{nm}{T}'] = fmt(100 * x.reachable.mean(), 0); N[f'Eta{nm}{T}'] = fmt(x.eta_E.mean())
        N[f'CostTar{nm}{T}'] = fmt(x[x.reachable].dC_target_pct.median())
        N[f'MwFull{nm}{T}'] = fmt(x[x.reachable].mw_full.median(), 0)
    # RQ2 allowance gap
    reg = realistic(G[G.unique], case).copy(); reg['G_allow'] = (reg.G - reg.G_phys).clip(lower=0)
    per = reg.groupby(['hour', 'w', 'cut']).G_allow.mean()
    share = (per >= 0.10).mean(); lo, hi = boot_ci((per >= 0.10).astype(float).values)
    N[f'AllowShare{T}'] = fmt(100 * share, 1); N[f'AllowShareLo{T}'] = fmt(100 * lo, 1); N[f'AllowShareHi{T}'] = fmt(100 * hi, 1)
    N[f'AllowMean{T}'] = fmt(per.mean()); N[f'AllowMedian{T}'] = fmt(per.median())
    pg = Gu.pivot_table(index='R', columns='rho', values='G', aggfunc='mean')
    Rs = sorted(Gu.R.unique()); r_hi, r_mid = Rs[-1], Rs[-2]
    sat = max(pg.loc[r_mid, r] - pg.loc[r_hi, r] for r in pg.columns if r <= 0.10)
    N[f'Saturation{T}'] = fmt(sat, 3); N[f'SatRmid{T}'] = f'{r_mid:.0f}'; N[f'SatRhi{T}'] = f'{r_hi:.0f}'
    need = Ru[Ru.G_ref > Ru.G_phys + 0.05].copy()
    tot = need.G_ref - need.G_phys
    fR = (need.G_ref - need.G_Rfree) / tot; fr = (need.G_ref - need.G_rhofree) / tot
    N[f'NeedCF{T}'] = str(len(need))
    N[f'ReliefR{T}'] = fmt(100 * fR.mean(), 0); N[f'ReliefRho{T}'] = fmt(100 * fr.mean(), 0)
    N[f'Joint{T}'] = fmt(100 * ((fR < 0.7) & (fr < 0.7)).mean(), 0)
    # RQ3 counterfactuals. Bisection starts above rho0 and never tests rho0 itself, so when the budget-free gap at rho0
    # already meets the threshold it returns rho0 plus one step. Those requests need no wider band, so rho* = rho0.
    at0 = need.G_Rfree <= need.G_phys + 0.05
    need.loc[at0, 'rho_star'] = need.loc[at0, 'rho0']
    N[f'RhoStarAtZero{T}'] = str(int(at0.sum()))
    q = need.rho_star.quantile([0.25, 0.5, 0.75]); N[f'RhoStarMed{T}'] = fmt(q[0.5], 3); N[f'RhoStarQa{T}'] = fmt(q[0.25], 3); N[f'RhoStarQc{T}'] = fmt(q[0.75], 3)
    q = need.R_star.quantile([0.25, 0.5, 0.75]); N[f'RStarMed{T}'] = fmt(q[0.5], 0); N[f'RStarQa{T}'] = fmt(q[0.25], 0); N[f'RStarQc{T}'] = fmt(q[0.75], 0)
    N[f'RhoStarNaN{T}'] = str(int(need.rho_star.isna().sum())); N[f'RStarNaN{T}'] = str(int(need.R_star.isna().sum()))
    sets = need.dropna(subset=['S_size']) if 'S_size' in need else need.iloc[0:0]
    N[f'NeedSet{T}'] = str(len(sets))
    if len(sets):
        cnt = sets.S_size.value_counts().sort_index()
        smax = int(sets.S_size.max())
        for k in range(1, smax + 1):
            N[f'SetSize{"ABCDE"[k-1]}{T}'] = str(int(cnt.get(k, 0)))
        N[f'SetLeTwo{T}'] = fmt(100 * (sets.S_size <= 2).mean(), 0)
        N[f'SetOpen{T}'] = str(int((~sets.S_closed.astype(bool)).sum()))
        if case == '10':
            # run.py validates requests at positions 0, 4 and 8 of each hour, i.e. (w, c) in
            # {(0.25, 2.5%), (0.5, 5%), (0.75, 10%)}; a NaN check there means no set of size <= 2 exists.
            val = {(0.25, 0.025), (0.5, 0.05), (0.75, 0.10)}
            v = sets[[(a, b) in val for a, b in zip(sets.w, sets.cut)]]
            agree = ((v.S_exact_check == v.S_size) | ((v.S_size > 2) & v.S_exact_check.isna())).mean() if len(v) else np.nan
            N['SetMissed'] = str(int(((v.S_exact_check < v.S_size)).sum()))
            N['SetValidated'] = str(len(v)); N['SetAgree'] = fmt(100 * agree, 0)
            lists = sets.S_units.apply(lambda u: [x for x in str(u).split('-') if x])
            top = lists.explode().dropna().value_counts()
            N['TopUnitTen'] = str(top.index[0]); N['TopUnitShareTen'] = fmt(100 * lists.apply(lambda l: top.index[0] in l).mean(), 0)
    # RQ7 recommendation size and efficiency
    N[f'RefMw{T}'] = fmt(Ru.ref_mw.median(), 1); N[f'RefUnits{T}'] = fmt(Ru.ref_units_moved.median(), 0)
    N[f'EffLossMed{T}'] = fmt(Ru.ref_eff_loss_pct.median(), 2); N[f'EffLossQc{T}'] = fmt(Ru.ref_eff_loss_pct.quantile(0.9), 2)
    rr = need[need.reachable & need.R_star.notna() & (need.mw_full > 1)]
    N[f'RStarRatio{T}'] = fmt(100 * (rr.R_star / rr.mw_full).median(), 0); N[f'RStarRatioN{T}'] = str(len(rr))
    # cloud-scale sensitivity
    if 'G_ref_cloud' in Ru:
        d1 = (Ru.G_ref - Ru.G_ref_cloud).abs(); d2 = (Ru.G_alt - Ru.G_alt_cloud).abs()
        N[f'CloudRefMad{T}'] = fmt(d1.median(), 3); N[f'CloudAltMad{T}'] = fmt(d2.median(), 3)
        N[f'CloudRefSp{T}'] = fmt(spearmanr(Ru.G_ref, Ru.G_ref_cloud)[0]); N[f'CloudAltSp{T}'] = fmt(spearmanr(Ru.G_alt, Ru.G_alt_cloud)[0])
    return R, G, Ru, Gu, hw, need


def concept_request():
    """Bounded optimum of the request drawn in Fig. 3 (macros ConceptHour, ConceptW, ConceptG).

    analysis/figures_seaborn.py draws the same request with the same anchors and random stream.
    """
    from eeldattain.model import System, Request, anchors, balance, DATA
    V = [json.loads(l) for f in sorted(glob.glob(f'{ATT}/vectors_e10*.jsonl')) for l in open(f)]
    R10, _ = load_case('10')
    pick = R10[(R10.unique) & (R10.cut == 0.05) & (R10.reachable) & (R10.G_ref > 0.25) & (R10.G_ref < 0.6)].sort_values('hour').iloc[0]
    v = [x for x in V if x['hour'] == pick.hour and x['w'] == pick.w and x['cut'] == pick.cut][0]
    s = System(f'{DATA}/eeld10_coeffs.csv'); rng = np.random.default_rng(3)
    Pinc = np.array(v['Pinc']); Ft = np.array(v['Ft']); PD = v['PD']
    A = anchors(s, PD, rng, K=40, sweep=25)
    req = Request(s, PD, Pinc, Ft, np.array(v['scale']), np.array(v['Pfr']), A['Pc'], A['Pe'])

    def cloud(R, rho, n):
        lo, hi = req.band(rho); out = []
        while len(out) < n:
            P = balance(rng.uniform(lo, hi), PD, lo, hi)
            if P is None: continue
            dd = np.abs(P - Pinc).sum(); P = P if dd <= R else Pinc + R / dd * (P - Pinc); out.append(s.F(P))
        return np.array(out)
    cloud(1e9, 1.0, 1500); cloud(pick.R0, pick.rho0, 1500)   # drawn for Fig. 3, kept so that the random stream matches
    Gb, _ = req.gap(pick.R0, pick.rho0, rng)
    N['ConceptHour'] = str(int(pick.hour)); N['ConceptW'] = f'{pick.w:g}'; N['ConceptG'] = fmt(Gb)


if __name__ == '__main__':
    for d in (TAB, MAC, PROC):
        os.makedirs(d, exist_ok=True)
    R10, G10, Ru10, Gu10, hw10, need10 = analyse('10')
    Ru10.to_csv(f'{PROC}/eeld10_requests_unique.csv', index=False); need10.to_csv(f'{PROC}/eeld10_counterfactuals.csv', index=False)
    hw10.to_csv(f'{PROC}/eeld10_hourly.csv')
    if glob.glob(f'{ATT}/requests_e40*.csv'):
        R4, G4, Ru4, Gu4, hw4, need4 = analyse('40')
        Ru4.to_csv(f'{PROC}/eeld40_requests_unique.csv', index=False); need4.to_csv(f'{PROC}/eeld40_counterfactuals.csv', index=False)
    if '--concept' in sys.argv:
        concept_request()
    with open(f'{MAC}/numbers_attain.tex', 'w') as f:
        for k, v in sorted(N.items()):
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(N, open(f'{PROC}/numbers_attain.json', 'w'), indent=1)
    print(json.dumps(N, indent=1))
