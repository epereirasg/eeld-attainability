"""Operational extension (experiments X1 to X4 and X7): statistics, processed files and number macros (numbers_ops.tex)."""
import glob
import json
import numpy as np, pandas as pd
from common import *
from eeldops import stats as S

OPSD = str(paths.RAW_OPS)
N = {}
fmt = lambda x, d=2: 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{d}f}'
EPS = 0.05


def pf(p):
    if p >= 0.001: return f'{p:.2g}'
    m, e = f'{p:.1e}'.split('e'); return f'{m}\\times10^{{{int(e)}}}'


def load(pattern, dtype=None):
    fs = sorted(glob.glob(f'{OPSD}/{pattern}'))
    return pd.concat([pd.read_csv(f, dtype=dtype) for f in fs], ignore_index=True) if fs else None


def x1_x2():
    R = load('ops_requests_*.csv', dtype={'S_units': str}); G = load('ops_grid_*.csv')
    R['unique'] = R.duplicate_of_w.isna(); Ru = R[R.unique].copy()
    G = G.merge(R[['hour', 'w', 'cut', 'unique']], on=['hour', 'w', 'cut']); G = G[G.unique & (G.R < 1e8)].copy()
    G['G_allow'] = (G.G - G.G_phys).clip(lower=0)
    N['OpsNumReq'] = str(len(R)); N['OpsNumReqUnique'] = str(len(Ru)); N['OpsHours'] = str(R.hour.nunique())
    N['OpsLossPctMed'] = fmt(Ru.loss_inc_pct.median(), 2); N['OpsLossPctMin'] = fmt(Ru.loss_inc_pct.min(), 2); N['OpsLossPctMax'] = fmt(Ru.loss_inc_pct.max(), 2)
    N['OpsLossMwMed'] = fmt(Ru.loss_inc.median(), 1)
    # RQ1 with losses (RQ8)
    hw = Ru.groupby('hour').agg(W=('front_width', 'first'), eta=('eta_E', 'mean'), PD=('PD', 'first'))
    r, p = S.spearman(hw.W, hw.eta); N['OpsSpearman'] = fmt(r); N['OpsSpearmanP'] = pf(p)
    N['OpsWidthMin'] = fmt(100 * hw.W.min(), 1); N['OpsWidthMax'] = fmt(100 * hw.W.max(), 1)
    for c, nm in [(0.05, 'B'), (0.10, 'C')]:
        x = Ru[Ru.cut == c]; N[f'OpsReach{nm}'] = fmt(100 * x.reachable.mean(), 0); N[f'OpsEta{nm}'] = fmt(x.eta_E.mean())
    # paired comparison with the lossless campaign (same hour, weight, cut, same R0 and rho0)
    L, LG = load_case('10'); L = L[L.cut.isin([0.05, 0.10])]
    M = Ru.merge(L[['hour', 'w', 'cut', 'G_ref', 'G_phys', 'eta_E', 'unique']], on=['hour', 'w', 'cut'], suffixes=('', '_nl'))
    M = M[M.unique_nl]
    pr = S.paired(M.G_ref_uniform, M.G_ref)
    N['OpsPairN'] = str(pr['n']); N['OpsGrefLoss'] = fmt(pr['mean_a']); N['OpsGrefLossSD'] = fmt(pr['sd_a'])
    N['OpsGrefNoLoss'] = fmt(pr['mean_b']); N['OpsGrefNoLossSD'] = fmt(pr['sd_b'])
    N['OpsGrefDiff'] = fmt(pr['mean_diff'], 3); N['OpsGrefDiffLo'] = fmt(pr['ci_lo'], 3); N['OpsGrefDiffHi'] = fmt(pr['ci_hi'], 3)
    N['OpsGrefHL'] = fmt(pr['HL'], 3); N['OpsGrefRbc'] = fmt(pr['rbc']); N['OpsGrefP'] = pf(pr["p"])
    pe = S.paired(M.eta_E, M.eta_E_nl); N['OpsEtaDiff'] = fmt(pe['mean_diff'], 3); N['OpsEtaP'] = pf(pe["p"])
    N['OpsEtaLo'] = fmt(pe['ci_lo'], 4); N['OpsEtaHi'] = fmt(pe['ci_hi'], 4); N['OpsEtaRbc'] = fmt(pe['rbc'])
    N['OpsAllowLossMean'] = fmt((M.G_ref_uniform - M.G_phys).clip(lower=0).mean()); N['OpsAllowNoLossMean'] = fmt((M.G_ref - M.G_phys_nl).clip(lower=0).mean())
    N['OpsGrefRankR'] = fmt(S.spearman(M.G_ref_uniform, M.G_ref)[0])
    # RQ2 with losses, uniform and ramp bands (RQ8, RQ9)
    tests = []; mat = {}
    for band, lev, nm in [('uniform', [0.05, 0.10, 0.20], 'Uni'), ('ramp', [10, 15, 30], 'Ramp')]:
        reg = G[(G.band == band) & G.R.isin([25, 50, 100]) & G.level.isin(lev)]
        per = reg.groupby(['hour', 'w', 'cut']).G_allow.mean(); ind = (per >= 0.10).astype(float)
        mat[nm] = set(per[per >= 0.10].index)
        lo, hi = S.boot_ci(ind.values)
        N[f'OpsAllowShare{nm}'] = fmt(100 * ind.mean(), 1); N[f'OpsAllowShareLo{nm}'] = fmt(100 * lo, 1); N[f'OpsAllowShareHi{nm}'] = fmt(100 * hi, 1)
        m, sd = S.mean_sd(per.values); N[f'OpsAllowMean{nm}'] = fmt(m); N[f'OpsAllowSD{nm}'] = fmt(sd)
    N['OpsSameMaterial'] = str(len(mat['Uni'] & mat['Ramp'])); N['OpsSwapMaterial'] = str(len(mat['Uni'] ^ mat['Ramp']))
    # ramp bands against the uniform surrogate at the reference allowance
    pr = S.paired(Ru.G_ref_ramp, Ru.G_ref_uniform)
    N['OpsRampVsUniDiff'] = fmt(pr['mean_diff'], 3); N['OpsRampVsUniLo'] = fmt(pr['ci_lo'], 3); N['OpsRampVsUniHi'] = fmt(pr['ci_hi'], 3)
    N['OpsRampVsUniP'] = pf(pr["p"]); N['OpsRampVsUniRbc'] = fmt(pr['rbc']); N['OpsGrefRamp'] = fmt(pr['mean_a']); N['OpsGrefRampSD'] = fmt(pr['sd_a'])
    # saturation of the budget under ramp bands
    pg = G[G.band == 'ramp'].pivot_table(index='R', columns='level', values='G', aggfunc='mean')
    N['OpsRampSat'] = fmt(max(pg.loc[100, t] - pg.loc[250, t] for t in (10, 15, 30)), 3)
    # counterfactuals under ramp bands
    need = Ru[Ru.G_ref_ramp > Ru.G_phys + EPS].copy(); tot = need.G_ref_ramp - need.G_phys
    N['OpsNeedCF'] = str(len(need))
    fR = (need.G_ref_ramp - need.G_Rfree_ramp) / tot; fb = (need.G_ref_ramp - need.G_bandfree) / tot
    N['OpsReliefR'] = fmt(100 * fR.mean(), 0); N['OpsReliefBand'] = fmt(100 * fb.mean(), 0)
    N['OpsJoint'] = fmt(100 * ((fR < 0.7) & (fb < 0.7)).mean(), 0)
    for col, nm, d in [('tau_star', 'TauStar', 0), ('R_star', 'RStar', 0), ('rho_star', 'RhoStar', 3)]:
        sub = need if col != 'rho_star' else Ru[Ru.G_ref_uniform > Ru.G_phys + EPS]
        q = sub[col].quantile([0.25, 0.5, 0.75])
        N[f'Ops{nm}Med'] = fmt(q[0.5], d); N[f'Ops{nm}Qa'] = fmt(q[0.25], d); N[f'Ops{nm}Qc'] = fmt(q[0.75], d)
        N[f'Ops{nm}NaN'] = str(int(sub[col].isna().sum())); N[f'Ops{nm}N'] = str(int(sub[col].notna().sum()))
    N['OpsTauStarLeSixty'] = fmt(100 * (need.tau_star <= 60).mean(), 0)
    sets = need.dropna(subset=['S_size']); sz = sets.S_size.astype(int)
    N['OpsNeedSet'] = str(len(sets))
    for k, nm in ((1, 'A'), (2, 'B'), (3, 'C')): N[f'OpsSetSize{nm}'] = str(int((sz == k).sum()))
    N['OpsSetSizeD'] = str(int((sz > 3).sum())); N['OpsSetLeTwo'] = fmt(100 * (sz <= 2).mean(), 0)
    lo, hi = S.boot_ci((sz <= 2).astype(float).values); N['OpsSetLeTwoLo'] = fmt(100 * lo, 0); N['OpsSetLeTwoHi'] = fmt(100 * hi, 0)
    v = sets[sets.S_validated.astype(bool)]
    agree = ((v.S_size <= 2) & (v.S_exact_check == v.S_size)) | ((v.S_size > 2) & v.S_exact_check.isna())
    N['OpsSetValN'] = str(len(v)); N['OpsSetValAgree'] = fmt(100 * agree.mean(), 0) if len(v) else 'n/a'
    units = pd.Series([u for s in sets.S_units.fillna('') for u in s.split('-') if u]).value_counts()
    N['OpsTopUnit'] = units.index[0] if len(units) else 'n/a'; N['OpsTopUnitShare'] = fmt(100 * units.iloc[0] / max(len(sets[sets.S_size <= 3]), 1), 0) if len(units) else 'n/a'
    N['OpsRefMw'] = fmt(Ru.ref_mw.median(), 0); N['OpsRefUnits'] = fmt(Ru.ref_units_moved.median(), 0)
    N['OpsEffLoss'] = fmt(Ru.ref_eff_loss_pct.median(), 2)
    # mean per-unit band at the reference settings, computed from the input data (no typed constants)
    from eeldops.data_loader import load_eeld10
    from eeldops.experiments import OPS
    coef, _, ramp, _ = load_eeld10()
    rate = np.minimum(ramp.UR_MW_per_h, ramp.DR_MW_per_h).to_numpy()
    N['OpsRampBandMwTauZero'] = fmt(float(np.mean(rate)) * OPS['tau0'] / 60, 1)
    N['OpsUniBandMwRhoZero'] = fmt(OPS['rho0'] * float(np.mean(coef.Pmax - coef.Pmin)), 1)
    return R, Ru, G, M, need


def x3():
    W = load('window_*.csv')
    if W is None: return None
    W = W.dropna(subset=['G_window'])
    N['WinN'] = str(len(W)); N['WinHours'] = str(W.hour.nunique())
    for col, nm in [('G_fixed', 'Fixed'), ('G_window', 'Adj')]:
        pr = S.paired(W[col], W.G_single); d = W[col] - W.G_single
        N[f'Win{nm}Diff'] = fmt(pr['mean_diff'], 3); N[f'Win{nm}Lo'] = fmt(pr['ci_lo'], 3); N[f'Win{nm}Hi'] = fmt(pr['ci_hi'], 3)
        N[f'Win{nm}HL'] = fmt(pr['HL'], 3); N[f'Win{nm}P'] = pf(pr["p"]); N[f'Win{nm}Rbc'] = fmt(pr['rbc'])
        N[f'Win{nm}Small'] = fmt(100 * (d.abs() < 0.05).mean(), 0); N[f'Win{nm}MedAbs'] = fmt(d.abs().median(), 3)
        N[f'Win{nm}Max'] = fmt(d.max(), 3); N[f'Win{nm}Min'] = fmt(d.min(), 3)
    N['WinSingleMean'] = fmt(W.G_single.mean()); N['WinFixedMean'] = fmt(W.G_fixed.mean()); N['WinAdjMean'] = fmt(W.G_window.mean())
    N['WinPhysFixedDiff'] = fmt((W.G_phys_fixed - W.G_phys_single).mean(), 3)
    return W


def x4():
    D = load('seeds_*.csv')
    if D is None: return None
    sp = D.groupby(['hour', 'cut', 'R', 'tau']).G.agg(lambda g: g.max() - g.min())
    N['SeedCells'] = str(len(sp)); N['SeedReq'] = str(D.groupby(['hour', 'cut']).ngroups)
    N['SeedMaxDiff'] = fmt(sp.max(), 4); N['SeedMedDiff'] = fmt(sp.median(), 4); N['SeedShareSmall'] = fmt(100 * (sp <= 0.01).mean(), 1)
    return D


def x7():
    X = load('sets40_*.csv', dtype={'exact_size': str})
    if X is None: return None
    N['ExactN'] = str(len(X)); N['ExactNeed'] = str(int(X.needs_set.sum()))
    nd = X[X.needs_set]
    for k, nm in [('1', 'One'), ('2', 'Two'), ('>=3', 'Three')]:
        N[f'Exact{nm}'] = str(int((nd.exact_size == k).sum()))
    N['ExactLeTwo'] = fmt(100 * nd.exact_size.isin(['1', '2']).mean(), 0) if len(nd) else 'n/a'
    N['ExactPairsMean'] = fmt(nd.pairs_checked.mean(), 0); N['ExactActiveMean'] = fmt(nd.n_band_active.mean(), 0)
    N['ExactRfreeChanged'] = str(int(((X.G_Rfree - X.G_Rfree_campaign).abs() > 0.01).sum()))
    N['ExactRfreeMaxDrop'] = fmt((X.G_Rfree_campaign - X.G_Rfree).max(), 3)
    return X


def main():
    x1_x2(); x3(); x4(); x7()
    with open(f'{MAC}/numbers_ops.tex', 'w') as f:
        for k, v in sorted(N.items()): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(N, open(f'{PROC}/numbers_ops.json', 'w'), indent=1)
    for k, v in sorted(N.items()): print(f'{k:28s} {v}')


if __name__ == '__main__':
    main()
