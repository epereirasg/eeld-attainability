"""Supplementary experiments (window dissection, oracle-limited comparison, 40-unit operational replication,
exhaustive 40-unit pairs, objective scales) and the anchor checks: number macros, tables and processed files.

Reads results/raw/supplementary/ and writes results/processed/numbers_supplementary.json, results/macros/numbers_supplementary.tex
and results/tables/tab_oracle.tex. Sections whose raw files are absent are skipped and reported as such.
"""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from common import *
from eeldattain.paths import RAW_SUPP, RAW_OPS

REV = str(RAW_SUPP)
N = {}
fmt = lambda x, d=2: 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


WORDS = {'7': 'Seven', '25': 'TwentyFive', '50': 'Fifty', '75': 'SeventyFive'}


def texname(k):
    """LaTeX control sequences cannot contain digits, so digit groups become words (X7 -> XSeven)."""
    import re
    return re.sub(r'\d+', lambda m: WORDS[m.group(0)], k)


def have(pattern):
    return sorted(glob.glob(f'{REV}/{pattern}'))


def cat(pattern, **kw):
    fs = have(pattern)
    return pd.concat([pd.read_csv(f, **kw) for f in fs], ignore_index=True) if fs else None


# ------------------------------------------------------------------ three-hour window dissection (RQ10)
def window():
    L = cat('window_lock.csv'); G = cat('window_lock_gaps.csv')
    if L is None: return None
    W = pd.read_csv(RAW_OPS / 'window_a.csv')
    e = L[L.construction == 'efficient_window'].copy(); c = L[L.construction == 'closest_to_hourly'].copy()
    lk = set(zip(e[e.locked].hour, e[e.locked].w))
    W['locked'] = [(h, w) in lk for h, w in zip(W.hour, W.w)]
    W['gap_one'] = W.G_fixed > 0.999
    N['WinLockPairs'] = str(int(e.locked.sum())); N['WinPairs'] = str(len(e))
    N['WinLockReq'] = str(int(W.locked.sum())); N['WinReqTotal'] = str(len(W))
    N['WinLockMatch'] = str(int((W.locked == W.gap_one).sum()))
    N['WinLockHours'] = ', '.join(str(h) for h in sorted(e[e.locked].hour.unique()))
    for w in (0.25, 0.5, 0.75):
        N[f'WinLockW{int(100*w)}'] = str(int(e[e.w == w].locked.sum()))
    N['WinLockAllUp'] = 'yes' if set(e[e.locked].locked_direction) == {'up'} else 'no'
    N['WinLockPmaxMin'] = str(int(e[e.locked].units_at_pmax.min())); N['WinLockPmaxMax'] = str(int(e[e.locked].units_at_pmax.max()))
    N['WinLockPDMin'] = f'{e[e.locked].PD.min():.0f}'
    N['WinLockViolShare'] = fmt(100 * (e[e.locked].hourly_incumbents_ramp_violation_mw > 0).mean(), 0)
    N['WinUnlockViolShare'] = fmt(100 * (e[~e.locked].hourly_incumbents_ramp_violation_mw > 0).mean(), 0)
    N['WinLockClosestPairs'] = str(int(c.locked.sum()))
    N['WinLockPersist'] = str(len(lk & set(zip(c[c.locked].hour, c[c.locked].w))))
    if G is not None:
        N['WinClosestGapOne'] = str(int((G.G_fixed > 0.999).sum()))
        N['WinClosestFixedDiff'] = fmt((G.G_fixed - G.G_single).mean(), 3)
    return e


# ------------------------------------------------------------------ oracle-limited comparison (RQ5c, Table 11)
LBL = {'POLY': 'POLY', 'ELLP': 'ELLP', 'CGM': 'CGM', 'GS': 'GS-R', 'COBYQA': 'COBYQA', 'SLSQP_FD': 'SLSQP-FD', 'CMAES': 'CMA-ES'}


def oracle():
    O = cat('oracle_a.csv')
    if O is None: return None
    O = O.groupby(['hour', 'w', 'cut', 'B', 'sigma', 'method'], as_index=False).agg(
        shortfall=('shortfall', 'median'), evals=('evals', 'mean'), err=('err', lambda s: int(s.fillna('').astype(bool).sum())),
        G_ref=('G_ref', 'first'))                                                  # CMA-ES: median over its seeds
    regimes = sorted(O[['B', 'sigma']].drop_duplicates().itertuples(index=False, name=None), key=lambda r: (r[1], r[0]))
    methods = [m for m in ['POLY', 'ELLP', 'CGM', 'GS', 'COBYQA', 'SLSQP_FD', 'CMAES'] if m in set(O.method)]
    N['OracleInst'] = str(O.drop_duplicates(['hour', 'w', 'cut']).shape[0])
    N['OracleErrors'] = str(int(O.err.sum()))
    rows = []
    for B, sg in regimes:
        X = O[(O.B == B) & (O.sigma == sg)]
        for m in methods:
            x = X[X.method == m]
            rows.append(dict(B=B, sigma=sg, method=m, n=len(x), med=x.shortfall.median(), q3=x.shortfall.quantile(0.75),
                             close=100 * (x.shortfall <= 0.01).mean(), evals=x.evals.mean()))
    T = pd.DataFrame(rows); T.to_csv(f'{PROC}/supplementary_oracle_summary.csv', index=False)
    for (B, sg), X in T.groupby(['B', 'sigma']):
        best = X.loc[X.med.idxmin()]; tag = {25: 'TwentyFive', 50: 'Fifty', 100: 'Hundred', 500: 'FiveHundred'}[int(B)] + ('Noisy' if sg > 0 else '')   # LaTeX names have no digits
        N[f'OracleBest{tag}'] = LBL[best.method]; N[f'OracleBestMed{tag}'] = fmt(best.med, 3)
        for m in methods:
            r = X[X.method == m].iloc[0]
            N[f'Oracle{m.replace("_", "")}{tag}'] = fmt(r.med, 3)
    # paired tests in the hardest noise-free regime: best repaired procedure vs best DFO method
    with open(f'{TAB}/tab_oracle.tex', 'w') as f:
        cols = ' & '.join(f'{B}' + (r'$^{\dagger}$' if sg > 0 else '') for B, sg in regimes)
        f.write('\\begin{tabular}{@{}l' + 'r' * len(regimes) + '@{}}\n\\toprule\n')
        f.write(f'Method & \\multicolumn{{{len(regimes)}}}{{c}}{{Median shortfall at budget (calls)}} \\\\\n')
        f.write(f'\\cmidrule(l){{2-{len(regimes) + 1}}}\n & {cols} \\\\\n\\midrule\n')
        for m in methods:
            vals = [T[(T.B == B) & (T.sigma == sg) & (T.method == m)].med.iloc[0] for B, sg in regimes]
            f.write(f'{LBL[m]} & ' + ' & '.join(f'{v:.3f}' for v in vals) + ' \\\\\n')
        f.write('\\midrule\nCalls used (mean, COBYQA) & ' + ' & '.join(
            f"{T[(T.B == B) & (T.sigma == sg) & (T.method == 'COBYQA')].evals.iloc[0]:.0f}" for B, sg in regimes) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    S1 = sorted(glob.glob(f'{REV}/oracle_t1/oracle_*.csv'))
    if S1:          # same design on the instance set generated with one BLAS thread (sensitivity)
        O1 = pd.concat([pd.read_csv(f) for f in S1]).groupby(['hour', 'w', 'cut', 'B', 'sigma', 'method'], as_index=False).shortfall.median()
        agree = 0
        for (B, sg), X in O1.groupby(['B', 'sigma']):
            m1 = X.groupby('method').shortfall.median().idxmin()
            m2 = T[(T.B == B) & (T.sigma == sg)].set_index('method').med.idxmin()
            agree += int(m1 == m2)
        N['OracleSensRegimes'] = str(O1[['B', 'sigma']].drop_duplicates().shape[0]); N['OracleSensBestAgree'] = str(agree)
    return T


# ------------------------------------------------------------------ stylized 40-unit operational subset (RQ9)
def ops40():
    X = cat('ops40_p*.csv', dtype={'S_class_loss': str})
    if X is None: return None
    X = X.sort_values(['PD', 'w'])
    N['OpsFortyN'] = str(len(X)); N['OpsFortyKappaR'] = fmt(X.kappa_r.iloc[0], 3)
    N['OpsFortyLossPct'] = fmt(X.loss_inc_pct_loss.median(), 2)
    N['OpsFortyGrefLoss'] = fmt(X.G_ref_uniform_loss.mean()); N['OpsFortyGrefNoLoss'] = fmt(X.G_ref_uniform_noloss.mean())
    N['OpsFortyGrefDiff'] = fmt((X.G_ref_uniform_loss - X.G_ref_uniform_noloss).mean(), 3)
    N['OpsFortyGrefMaxAbsDiff'] = fmt((X.G_ref_uniform_loss - X.G_ref_uniform_noloss).abs().max(), 3)
    needx = X.G_ref_ramp_loss > X.G_phys_loss + 0.05
    N['OpsFortyNeedCF'] = str(int(needx.sum()))
    N['OpsFortyTauMed'] = fmt(X[needx].tau_star_loss.median(), 0); N['OpsFortyTauNaN'] = str(int(X[needx].tau_star_loss.isna().sum()))
    N['OpsFortyTauMedNoLoss'] = fmt(X.tau_star_noloss.median(), 0)
    N['OpsFortyRStarMed'] = fmt(X.R_star_loss.median(), 0)
    N['OpsFortyReliefBudget'] = fmt(100 * X.relief_budget_loss.mean(), 0); N['OpsFortyReliefBand'] = fmt(100 * X.relief_band_loss.mean(), 0)
    cls = X.S_class_loss.fillna('').astype(str)
    for k, nm in (('1', 'One'), ('2', 'Two'), ('>=3', 'Three'), ('0', 'Zero')):
        N[f'OpsFortyS{nm}'] = str(int((cls == k).sum()))
    N['OpsFortyRuntimeH'] = fmt(X.runtime_s.sum() / 3600, 1)
    N['OpsFortyRuntimeReqMin'] = fmt(X.runtime_s.median() / 60, 0)
    return X


# ------------------------------------------------------------------ full stylized 40-unit operational replication (RQ4, RQ9)
def ops40full():
    """24 requests (6 levels x 2 weights x 2 cuts), loss and lossless models, seed replication and DE certificate."""
    from eeldops import stats as ST
    XF = cat('ops40full_p*.csv', dtype={'S_class_loss': str})
    if XF is None: return None
    XF = XF.sort_values(['PD', 'w', 'cut']).reset_index(drop=True)
    N['OpsFortyKappaR'] = fmt(XF.kappa_r.iloc[0], 3)
    def pf(p):
        if p >= 0.001: return f'{p:.3f}'
        m, e = f'{p:.1e}'.split('e'); return f'{m}\\times10^{{{int(e)}}}'
    need = XF.G_ref_ramp_loss > XF.G_phys_loss + 0.05
    N['OpsFortyFullN'] = str(len(XF)); N['OpsFortyFullNeedCF'] = str(int(need.sum()))
    N['OpsFortyFullLossPctMed'] = fmt(XF.loss_inc_pct_loss.median(), 2)
    N['OpsFortyFullLossPctMin'] = fmt(XF.loss_inc_pct_loss.min(), 2); N['OpsFortyFullLossPctMax'] = fmt(XF.loss_inc_pct_loss.max(), 2)
    def put(tag, pr):
        N[f'OpsFortyFull{tag}N'] = str(pr['n']); N[f'OpsFortyFull{tag}A'] = fmt(pr['mean_a']); N[f'OpsFortyFull{tag}ASD'] = fmt(pr['sd_a'])
        N[f'OpsFortyFull{tag}B'] = fmt(pr['mean_b']); N[f'OpsFortyFull{tag}BSD'] = fmt(pr['sd_b'])
        N[f'OpsFortyFull{tag}Diff'] = fmt(pr['mean_diff'], 3); N[f'OpsFortyFull{tag}Lo'] = fmt(pr['ci_lo'], 3); N[f'OpsFortyFull{tag}Hi'] = fmt(pr['ci_hi'], 3)
        N[f'OpsFortyFull{tag}P'] = pf(pr['p']); N[f'OpsFortyFull{tag}Rbc'] = fmt(pr['rbc']); N[f'OpsFortyFull{tag}HL'] = fmt(pr['HL'], 3)
    # (a) losses vs no losses, reference gap with uniform bands (same request, same stylized pipeline)
    put('Loss', ST.paired(XF.G_ref_uniform_loss, XF.G_ref_uniform_noloss))
    N['OpsFortyFullGrefRankR'] = fmt(spearmanr(XF.G_ref_uniform_loss, XF.G_ref_uniform_noloss)[0])
    # (b) ramp-derived vs uniform bands at the reference allowance, loss model
    put('Ramp', ST.paired(XF.G_ref_ramp_loss, XF.G_ref_uniform_loss))
    # (c) emission reach with and without losses
    put('Eta', ST.paired(XF.eta_E_loss, XF.eta_E_noloss))
    # allowance gap at the reference allowance (ramp bands, losses)
    allow = (XF.G_ref_ramp_loss - XF.G_phys_loss).values
    N['OpsFortyFullAllowMean'] = fmt(allow.mean()); N['OpsFortyFullAllowSD'] = fmt(allow.std(ddof=1))
    sh = (allow >= 0.10).astype(float); lo, hi = boot_ci(sh)
    N['OpsFortyFullAllowShare'] = fmt(100 * sh.mean(), 0); N['OpsFortyFullAllowShareLo'] = fmt(100 * lo, 0); N['OpsFortyFullAllowShareHi'] = fmt(100 * hi, 0)
    alluni = (XF.G_ref_uniform_loss - XF.G_phys_loss).values; shu = (alluni >= 0.10).astype(float)
    N['OpsFortyFullAllowShareUni'] = fmt(100 * shu.mean(), 0)
    # counterfactuals on requests that need one
    t = XF[need].tau_star_loss
    N['OpsFortyFullTauMed'] = fmt(t.median(), 0); N['OpsFortyFullTauQa'] = fmt(t.quantile(0.25), 0); N['OpsFortyFullTauQc'] = fmt(t.quantile(0.75), 0)
    N['OpsFortyFullTauNaN'] = str(int(t.isna().sum())); N['OpsFortyFullTauLeSixty'] = fmt(100 * (t <= 60).mean(), 0)
    r_ = XF[need].R_star_loss; N['OpsFortyFullRStarMed'] = fmt(r_.median(), 0); N['OpsFortyFullRStarNaN'] = str(int(r_.isna().sum()))
    N['OpsFortyFullReliefBudget'] = fmt(100 * XF[need].relief_budget_loss.mean(), 0)
    N['OpsFortyFullReliefBand'] = fmt(100 * XF[need].relief_band_loss.mean(), 0)
    cf = XF[need].S_class_loss.fillna('').astype(str)
    for k, nm in (('0', 'Zero'), ('1', 'One'), ('2', 'Two'), ('>=3', 'Three')):
        N[f'OpsFortyFullS{nm}'] = str(int((cf == k).sum()))
    cs = cf[cf.isin(['1', '2', '>=3'])]          # requests that need a unit counterfactual (class 0: the budget alone closes the gap)
    N['OpsFortyFullNeedSet'] = str(len(cs))
    le2 = cs.isin(['1', '2']).astype(float).values; lo, hi = boot_ci(le2)
    N['OpsFortyFullSLeTwo'] = fmt(100 * le2.mean(), 0); N['OpsFortyFullSLeTwoLo'] = fmt(100 * lo, 0); N['OpsFortyFullSLeTwoHi'] = fmt(100 * hi, 0)
    N['OpsFortyFullCPUh'] = fmt(XF.runtime_s.sum() / 3600, 1)
    N['OpsFortyFullFallbacks'] = str(int(XF.solver_fallbacks_loss.sum() + XF.solver_fallbacks_noloss.sum()))
    XF.to_csv(f'{PROC}/supplementary_ops40full_requests.csv', index=False)
    # seed replication: three seeds, four starts each, every reference cell of both models
    SD = cat('ops40_seeds*.csv')
    cells = ['G_phys', 'G_ref_uniform', 'G_ref_ramp', 'G_Rfree_ramp', 'G_bandfree']
    if SD is not None:
        L = SD.melt(id_vars=['PD', 'w', 'cut', 'model', 'rep'], value_vars=cells, var_name='cell', value_name='G')
        g = L.groupby(['PD', 'w', 'cut', 'model', 'cell']).G
        spread = (g.max() - g.min()); sdv = g.std(ddof=1)
        N['OpsFortySeedCells'] = str(len(spread)); N['OpsFortySeedReps'] = str(SD.rep.nunique())
        N['OpsFortySeedMaxSpread'] = fmt(spread.max(), 3); N['OpsFortySeedMedSD'] = fmt(sdv.median(), 4)
        N['OpsFortySeedShareSmall'] = fmt(100 * (spread <= 0.01).mean(), 0)
        st = XF.melt(id_vars=['PD', 'w', 'cut'], value_vars=[f'{c}_{m}' for c in cells for m in ('loss', 'noloss')], var_name='k', value_name='G_stored')
        st['model'] = st.k.str.rsplit('_', n=1).str[1]; st['cell'] = st.k.str.rsplit('_', n=1).str[0]
        best = g.min().rename('G_seedmin').reset_index().merge(st, on=['PD', 'w', 'cut', 'model', 'cell'])
        imp = best.G_stored - best.G_seedmin
        N['OpsFortySeedMaxImp'] = fmt(imp.max(), 3); N['OpsFortySeedImproved'] = str(int((imp > 0.01).sum()))
        N['OpsFortySeedImprovedLoss'] = str(int(((imp > 0.01) & (best.model == 'loss')).sum()))
        N['OpsFortySeedMaxImpLoss'] = fmt(imp[best.model == 'loss'].max(), 3)
        best.assign(spread=spread.values if len(spread) == len(best) else np.nan).to_csv(f'{PROC}/supplementary_ops40_seeds_summary.csv', index=False)
    # independent DE certificate of every reference cell
    CE = cat('certificate_ops40_p*.csv')
    if CE is not None:
        N['OpsFortyCertCells'] = str(len(CE)); N['OpsFortyCertMaxImp'] = fmt(CE.improvement.max(), 3)
        N['OpsFortyCertFail'] = str(int((CE.improvement > 0.01).sum()))
        N['OpsFortyCertPolished'] = str(int((CE.de_status == 'polished').sum()))
        N['OpsFortyCertFailLoss'] = str(int(((CE.improvement > 0.01) & (CE.model == 'loss')).sum()))
        N['OpsFortyCertMaxImpLoss'] = fmt(CE[CE.model == 'loss'].improvement.max(), 3)
        bad = CE[CE.improvement > 0.01]
        used = ~((CE.model == 'noloss') & (CE.cell == 'G_bandfree'))       # cells behind reported numbers
        imp_used = [CE[used].improvement.max()]
        if SD is not None:
            imp_used.append(imp[~((best.model == 'noloss') & (best.cell == 'G_bandfree'))].max())
        N['OpsFortyMaxImpUsed'] = fmt(max(imp_used), 3)
        N['OpsFortyCertFailCells'] = 'band-free relaxation, lossless model' if set(bad.cell) == {'G_bandfree'} and set(bad.model) == {'noloss'} else 'mixed'
    return XF


# ------------------------------------------------------------------ exhaustive pairs on 40 units (RQ11, Table 12)
def pairs40():
    S = cat('pairs40_p*.csv'); A = cat('pairs40_all_p*.csv')
    if S is not None:                  # pilot on one request per demand level, when stored
        pilot(S, A)
    full_pairs()
    return S if S is not None else True


def pilot(S, A):
    N['PairsReq'] = str(len(S)); N['PairsTested'] = str(int(S.pairs_tested.sum()))
    N['PairsClosing'] = str(int(S.pairs_closing.sum())); N['PairsClosingOneActive'] = str(int(S.closing_one_active.sum()))
    N['PairsClosingNoneActive'] = str(int(S.closing_none_active.sum())); N['PairsClosingBoth'] = str(int(S.closing_both_active.sum()))
    N['PairsMinMargin'] = fmt(S.margin_to_thr.min(), 3)
    N['PairsReproX7'] = str(int(((S.G_Rfree - S.G_Rfree_X7).abs() < 1e-6).sum()))
    N['PairsSingleReproX7'] = str(int(((S.single_best - S.single_best_X7).abs() < 1e-6).sum()))
    if A is not None:
        A['kind'] = np.where(A.i_active & A.j_active, 'both', np.where(A.i_active | A.j_active, 'one', 'none'))
        N['PairsOneActiveN'] = str(int((A.kind == 'one').sum()))


def full_pairs():
    F = sorted(glob.glob(f'{REV}/pairs40_full/pairs40_p*.csv'))
    if F:                              # all |S*| >= 3 requests of X7 tested on every pair
        P = pd.concat([pd.read_csv(f) for f in F]).drop_duplicates(['PD', 'w', 'cut'])
        X7 = pd.read_csv(RAW_OPS / 'sets40_a_p0.csv', dtype={'exact_size': str}); nd = X7[X7.needs_set].copy()
        N['PairsFullReq'] = str(len(P)); N['PairsFullOf'] = str(int((nd.exact_size == '>=3').sum()))
        N['PairsFullFlipped'] = str(int((P.pairs_closing > 0).sum()))
        N['PairsFullFlippedNoneActive'] = str(int(((P.pairs_closing > 0) & (P.closing_both_active == 0) & (P.closing_one_active == 0)).sum()))
        flip = set(zip(P[P.pairs_closing > 0].PD, P[P.pairs_closing > 0].w, P[P.pairs_closing > 0].cut))
        nd['corrected'] = [('2' if (a, b, c) in flip else e) for a, b, c, e in zip(nd.PD, nd.w, nd.cut, nd.exact_size)]
        tested = set(zip(P.PD, P.w, P.cut))
        nd.loc[[(a, b, c) not in tested and e == '>=3' for a, b, c, e in zip(nd.PD, nd.w, nd.cut, nd.exact_size)], 'corrected'] = '>=3 (untested)'
        for k, nm in (('1', 'One'), ('2', 'Two')):
            N[f'PairsFullCorr{nm}'] = str(int((nd.corrected == k).sum()))
        N['PairsFullCorrThree'] = str(int(nd.corrected.str.startswith('>=3').sum()))
        N['PairsFullCorrLeTwo'] = fmt(100 * nd.corrected.isin(['1', '2']).mean(), 0)
        N['PairsFullBorderline'] = str(int(((P.best_pair_G - P.thr).between(0, 0.02)).sum()))
        nd.to_csv(f'{PROC}/supplementary_eeld40_sets_corrected.csv', index=False)


# ------------------------------------------------------------------ counterfactuals under the two objective scales (Table 8)
def scale_cf():
    D = cat('scale_cf_p*.csv', dtype={'S_class': str})
    if D is None: return None
    D = D.drop_duplicates(['hour', 'w', 'cut', 'scale'], keep='last')
    P = D.pivot_table(index=['hour', 'w', 'cut'], columns='scale',
                      values=['needs_cf', 'rho_star', 'R_star', 'G_ref', 'G_phys'], aggfunc='first')
    S = D.pivot_table(index=['hour', 'w', 'cut'], columns='scale', values='S_class', aggfunc='first')
    both = P.dropna(subset=[('needs_cf', 'target'), ('needs_cf', 'cloud')])
    N['ScaleReq'] = str(len(both))
    nt = both[('needs_cf', 'target')].astype(bool); nc = both[('needs_cf', 'cloud')].astype(bool)
    N['ScaleNeedTarget'] = str(int(nt.sum())); N['ScaleNeedCloud'] = str(int(nc.sum())); N['ScaleNeedBoth'] = str(int((nt & nc).sum()))
    N['ScaleNeedAgree'] = fmt(100 * (nt == nc).mean(), 0)
    B = both[nt & nc]
    for col, nm, d in (('rho_star', 'RhoStar', 3), ('R_star', 'RStar', 0)):
        t, c = B[(col, 'target')].astype(float), B[(col, 'cloud')].astype(float); ok = t.notna() & c.notna()
        N[f'Scale{nm}MedTarget'] = fmt(t[ok].median(), d); N[f'Scale{nm}MedCloud'] = fmt(c[ok].median(), d)
        N[f'Scale{nm}MedAbsDiff'] = fmt((t[ok] - c[ok]).abs().median(), d)
        N[f'Scale{nm}Sp'] = fmt(spearmanr(t[ok], c[ok])[0])
        dd = (c[ok] - t[ok]).values
        N[f'Scale{nm}P'] = fmt(wilcoxon(dd).pvalue, 3) if np.any(dd != 0) else '1.000'
        N[f'Scale{nm}N'] = str(int(ok.sum()))
    Sb = S.loc[B.index]
    agree = (Sb['target'].fillna('') == Sb['cloud'].fillna(''))
    N['ScaleSetAgree'] = fmt(100 * agree.mean(), 0)
    le2 = lambda s: s.isin(['0', '1', '2'])
    N['ScaleSetLeTwoTarget'] = fmt(100 * le2(Sb['target'].fillna('')).mean(), 0)
    N['ScaleSetLeTwoCloud'] = fmt(100 * le2(Sb['cloud'].fillna('')).mean(), 0)
    T = D[D.scale == 'target']
    ok = T.needs_cf.astype(bool) & T.campaign_rho_star.notna() & T.rho_star.notna()
    N['ScaleConsistRhoMad'] = fmt((T[ok].rho_star - T[ok].campaign_rho_star).abs().median(), 4)
    N['ScaleConsistGrefMax'] = fmt((T.G_ref - T.campaign_G_ref).abs().max(), 4)
    ct = pd.crosstab(Sb['target'].fillna('none'), Sb['cloud'].fillna('none'))
    ct.to_csv(f'{PROC}/supplementary_scale_set_crosstab.csv')
    return D


# ------------------------------------------------------------------ anchor-cache check
def main():
    os.makedirs(TAB, exist_ok=True); os.makedirs(MAC, exist_ok=True)
    done = {name: f() is not None for name, f in (('window', window), ('oracle', oracle), ('ops40', ops40), ('ops40 full', ops40full),
                                                   ('pairs40', pairs40), ('scale', scale_cf))}
    for k, v in done.items():
        print(f'{k:14s} {"processed" if v else "no raw files, skipped"}')
    with open(f'{MAC}/numbers_supplementary.tex', 'w') as f:
        for k, v in sorted(N.items()): f.write(f'\\newcommand{{\\{texname(k)}}}{{{v}}}\n')
    json.dump(N, open(f'{PROC}/numbers_supplementary.json', 'w'), indent=1)
    for k, v in sorted(N.items()): print(f'{k:28s} {v}')


if __name__ == '__main__':
    main()
