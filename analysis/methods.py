import os
"""RQ5 and RQ6: search methods against each other and against the multistart reference."""
import glob
import json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from common import *

RB = str(paths.RAW_ROBUST); DF = str(paths.RAW_DFO)
PROC_M, BASE_M, DFO_M = ['POLY', 'ELLP', 'CGM'], ['GS', 'FACE', 'NICE_R', 'DAACE_R', 'DiCE'], ['COBYQA', 'CMAES', 'SLSQP_FD']
CFG_ORDER = ['default', 'budget200', 'budget2000', 'allow100_010', 'allow50_005', 'scale_cloud', 'inc_random', 'emission_repo', 'method_seed2']
CFG_LABEL = {'default': 'Default', 'budget200': '200 calls', 'budget2000': '2000 calls', 'allow100_010': r'$R=100$, $\rho=0.10$',
             'allow50_005': r'$R=50$, $\rho=0.05$', 'scale_cloud': 'Cloud scale', 'inc_random': 'Random incumbent',
             'emission_repo': 'Scaled emission', 'method_seed2': 'Method seed 11'}
N = {}


def pf_tex(p):
    if p >= 1e-3: return f'{p:.3f}'
    m, e = f'{p:.1e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def hl(d):
    d = np.asarray(d); w = (d[:, None] + d[None, :]) / 2
    return float(np.median(w[np.triu_indices(len(d))]))


def rbc(d):
    d = np.asarray(d); d = d[d != 0]
    if not len(d): return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cohen_d(a, b, B=10000, seed=0):
    """Paired standardized differences (Lakens 2013). d_av divides the mean paired difference by the mean of the two
    standard deviations and is the effect size reported in the paper. d_z divides it by the standard deviation of the
    differences and grows with the correlation of the paired values, so it is kept for reference only.
    Returns d_av with a percentile bootstrap interval over instances, d_z and the correlation of the pairs.
    Negative values mean that the first method attains a lower gap."""
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    def dav(x, y):
        s = (x.std(ddof=1) + y.std(ddof=1)) / 2
        return np.nan if s == 0 else (x - y).mean() / s
    if len(d) < 2:
        return dict(dav=np.nan, dav_lo=np.nan, dav_hi=np.nan, dz=np.nan, corr=np.nan)
    idx = np.random.default_rng(seed).integers(0, len(d), (B, len(d)))
    v = np.array([dav(a[i], b[i]) for i in idx])
    lo, hi = np.nanquantile(v, [0.025, 0.975])
    sd = d.std(ddof=1)
    return dict(dav=float(dav(a, b)), dav_lo=float(lo), dav_hi=float(hi), dz=float(d.mean() / sd) if sd > 0 else np.nan,
                corr=float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan)


def test(a, b):
    d = a - b
    p = wilcoxon(d, zero_method='zsplit').pvalue if np.any(d != 0) else 1.0
    return dict(n=len(d), med_a=float(np.median(a)), med_b=float(np.median(b)), HL=hl(d), rbc=rbc(d), p=p, a_better=float((d < 0).mean()),
                mean_diff=float(d.mean()), **cohen_d(a, b))


def load():
    # raw_C5 and raw_all5 hold the five configurations that were run for every family on the same instances (one run,
    # instance hashes recorded). They replace the stored rows of those configurations.
    new_rb = sorted(glob.glob(f'{RB}/raw_C5*.csv')); new_df = sorted(glob.glob(f'{DF}/raw_all5*.csv'))
    rb = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{RB}/raw_[ABC]*.csv')) if not os.path.basename(f).startswith('raw_C5')])
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{DF}/raw_all.csv'))])
    if new_rb:
        R5 = pd.concat([pd.read_csv(f) for f in new_rb]); D5 = pd.concat([pd.read_csv(f) for f in new_df]) if new_df else None
        cf5 = set(R5.config)
        rb = pd.concat([rb[~rb.config.isin(cf5)], R5], ignore_index=True)
        if D5 is not None:
            df = pd.concat([df[~df.config.isin(set(D5.config))], D5], ignore_index=True)
        # identical instances (the same incumbent and target for two weights) enter once
        kh = R5[['config', 'hour', 'w', 'cut', 'inst_hash']].drop_duplicates()
        dup = kh[kh.duplicated(['config', 'inst_hash'])][['config', 'hour', 'w', 'cut']].assign(_dup=1)
        N['DupInstDropped'] = str(len(dup))
        rb = rb.merge(dup, on=['config', 'hour', 'w', 'cut'], how='left'); rb = rb[rb._dup.isna()].drop(columns='_dup')
        df = df.merge(dup, on=['config', 'hour', 'w', 'cut'], how='left'); df = df[df._dup.isna()].drop(columns='_dup')
    rb = rb.drop_duplicates(['config', 'hour', 'w', 'cut', 'method'])
    rb['mseed'] = 0
    ref = rb.drop_duplicates(['config', 'hour', 'w', 'cut'])[['config', 'hour', 'w', 'cut', 'G_ref']]
    df = df.merge(ref, on=['config', 'hour', 'w', 'cut'], how='left')
    # CMA-ES: median over its three seeds per instance; stochastic spread reported separately
    cma_spread = df[df.method == 'CMAES'].groupby(['config', 'hour', 'w', 'cut']).G_fallback.agg(lambda x: x.max() - x.min())
    dfm = df.groupby(['config', 'hour', 'w', 'cut', 'method'], as_index=False).agg(G=('G', 'median'), G_fallback=('G_fallback', 'median'),
                                                                                  evals=('evals', 'mean'), runtime=('runtime', 'median'), G_ref=('G_ref', 'first'))
    cols = ['config', 'hour', 'w', 'cut', 'method', 'G', 'G_fallback', 'evals', 'runtime', 'G_ref']
    allm = pd.concat([rb[cols], dfm[cols]], ignore_index=True)
    allm['shortfall'] = allm.G_fallback - allm.G_ref
    return rb, df, allm, cma_spread


def main():
    rb, df, A, cma_spread = load()
    A.to_csv(f'{PROC}/methods_all_runs.csv', index=False)
    inst = A.dropna(subset=['G_ref']).drop_duplicates(['config', 'hour', 'w', 'cut']).groupby('config').size()
    N['NumInstDefault'] = str(int(inst['default'])); N['NumInstRandom'] = str(int(inst['inc_random'])); N['NumInstScaled'] = str(int(inst['emission_repo']))
    N['NumMethodRuns'] = f'{len(rb) + len(df):,}'.replace(',', '{,}')
    N['NegShortfall'] = str(int((A.shortfall < -0.01).sum()))
    # family comparisons per configuration
    rows, FR = [], []
    for cfg in CFG_ORDER:
        P = A[A.config == cfg].pivot_table(index=['hour', 'w', 'cut'], columns='method', values='G_fallback').dropna(axis=1, how='all')
        P = P.dropna()
        fr = friedmanchisquare(*[P[c].values for c in P.columns])   # omnibus test over the compared methods (equal call budget)
        FR.append(dict(config=cfg, k=P.shape[1], n=len(P), chi2=fr.statistic, p=fr.pvalue))
        fam = {'proc': P[PROC_M].min(axis=1), 'base': P[BASE_M].min(axis=1)}
        if all(m in P for m in DFO_M): fam['dfo'] = P[DFO_M].min(axis=1)
        for a, b in [('proc', 'base'), ('dfo', 'proc'), ('dfo', 'base')]:
            if a in fam and b in fam:
                rows.append(dict(config=cfg, A=a, B=b, **test(fam[a].values, fam[b].values)))
        for m in BASE_M:
            for pm in PROC_M:
                rows.append(dict(config=cfg, A=pm, B=m, **test(P[pm].values, P[m].values)))
        if 'COBYQA' in P:
            for pm in PROC_M:
                rows.append(dict(config=cfg, A='COBYQA', B=pm, **test(P['COBYQA'].values, P[pm].values)))
    T = pd.DataFrame(rows)
    T['p_holm'] = np.nan
    for cfg, idx in T.groupby('config').groups.items():
        T.loc[idx, 'p_holm'] = holm(T.loc[idx, 'p'].values)
    T['A_wins'] = (T.p_holm < 0.05) & (T.HL <= -0.05)
    T.to_csv(f'{PROC}/methods_paired_tests.csv', index=False)
    fp = T[(T.A == 'proc') & (T.B == 'base')]; N['ProcBeatsBaseCfg'] = str(int(fp.A_wins.sum()))
    fd = T[(T.A == 'dfo') & (T.B == 'proc')]; N['DfoBeatsProcCfg'] = str(int(fd.A_wins.sum())); N['DfoCfgTotal'] = str(len(fd))
    N['DfoVerdict'] = 'supported' if int(fd.A_wins.sum()) >= 7 and len(fd) == 9 else 'not supported'
    for m in BASE_M:
        x = T[(T.B == m) & T.A.isin(PROC_M)]
        N[f'WinsVs{m.replace("_", "")}'] = f'{int(x.groupby("config").A_wins.all().sum())}'
        N[f'WinsAnyVs{m.replace("_", "")}'] = f'{int(x.groupby("config").A_wins.any().sum())}'
    x = T[(T.A == 'COBYQA') & T.B.isin(PROC_M)]; N['CobyqaBeatsAllProcCfg'] = str(int(x.groupby('config').A_wins.all().sum())); N['CobyqaCfgTotal'] = str(x.config.nunique())
    d = fp.set_index('config'); N['ProcBaseHLDefault'] = f'{d.loc["default","HL"]:.3f}'; N['ProcBaseHLScaled'] = f'{d.loc["emission_repo","HL"]:.3f}'
    N["ProcBaseHLCloud"] = f"{d.loc['scale_cloud','HL']:.3f}"
    N['ProcBasePScaled'] = f'{d.loc["emission_repo","p_holm"]:.2f}'
    FRD = pd.DataFrame(FR); FRD.to_csv(f'{PROC}/methods_friedman.csv', index=False)
    N['FriedmanCfg'] = str(len(FRD)); N['FriedmanMaxP'] = pf_tex(FRD.p.max())
    N['FriedmanKmin'] = str(int(FRD.k.min())); N['FriedmanKmax'] = str(int(FRD.k.max()))
    sig = fp[fp.A_wins]
    N['ProcBaseDavMin'] = f'{sig.dav.abs().min():.2f}'; N['ProcBaseDavMax'] = f'{sig.dav.abs().max():.2f}'
    N['ProcBaseDavDefault'] = f'{d.loc["default","dav"]:.2f}'
    N['ProcBaseDavScaled'] = f'{d.loc["emission_repo","dav"]:.2f}'; N['ProcBaseDavScaledLo'] = f'{d.loc["emission_repo","dav_lo"]:.2f}'; N['ProcBaseDavScaledHi'] = f'{d.loc["emission_repo","dav_hi"]:.2f}'
    e = fd.set_index('config')
    N['DfoProcDavMin'] = f'{e.dav.abs().min():.2f}'; N['DfoProcDavMax'] = f'{e.dav.abs().max():.2f}'
    N['DfoProcHLSmall'] = f'{e.loc["allow50_005","HL"]:.3f}'; N['DfoProcDavSmall'] = f'{e.loc["allow50_005","dav"]:.2f}'
    N['DfoProcDzSmall'] = f'{e.loc["allow50_005","dz"]:.2f}'; N['DfoProcCorrSmall'] = f'{e.loc["allow50_005","corr"]:.2f}'
    N['DfoProcSigCfg'] = str(int((e.p_holm < 0.05).sum()))
    if 'DupInstDropped' not in N: N['DupInstDropped'] = '0'
    # LaTeX table: family comparisons
    with open(f'{TAB}/tab_family_tests.tex', 'w') as f:
        f.write('\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\n')
        f.write(' & & \\multicolumn{4}{c}{Procedures vs. generators} & \\multicolumn{4}{c}{DFO vs. procedures} \\\\\n\\cmidrule(lr){3-6}\\cmidrule(l){7-10}\n')
        f.write('Configuration & $n$ & Medians & HL & $d_{\\mathrm{av}}$ [95\\% CI] & $p_{\\mathrm{Holm}}$ & Medians & HL & $d_{\\mathrm{av}}$ [95\\% CI] & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n')
        for cfg in CFG_ORDER:
            a = T[(T.config == cfg) & (T.A == 'proc') & (T.B == 'base')].iloc[0]
            b = T[(T.config == cfg) & (T.A == 'dfo') & (T.B == 'proc')]
            def pf(p): return '$<$0.001' if p < 0.001 else f'{p:.3f}'
            def dzs(r): return f'${r.dav:+.2f}$ [${r.dav_lo:+.2f}$, ${r.dav_hi:+.2f}$]'
            sb = f'{b.iloc[0].med_a:.3f}/{b.iloc[0].med_b:.3f} & ${b.iloc[0].HL:+.3f}$ & {dzs(b.iloc[0])} & {pf(b.iloc[0].p_holm)}' if len(b) else '-- & -- & -- & --'
            f.write(f'{CFG_LABEL[cfg]} & {a.n} & {a.med_a:.3f}/{a.med_b:.3f} & ${a.HL:+.3f}$ & {dzs(a)} & {pf(a.p_holm)} & {sb} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    # ---- the proposed counterfactual (certified constrained solve, column G_ref) against every method
    # Paired difference G_ref - G_m on the same instances: negative values mean that the proposal attains a lower gap.
    # Holm correction within each configuration over the family comparisons, and over the 11 methods at default.
    PR = []
    for cfg in CFG_ORDER:
        X = A[A.config == cfg]
        P = X.pivot_table(index=['hour', 'w', 'cut'], columns='method', values='G_fallback').dropna(axis=1, how='all').dropna()
        ref = X.drop_duplicates(['hour', 'w', 'cut']).set_index(['hour', 'w', 'cut']).G_ref.reindex(P.index).values
        fams = [('gen', BASE_M), ('proc', PROC_M)] + ([('dfo', DFO_M)] if all(m in P for m in DFO_M) else [])
        rows_c = [dict(config=cfg, B=f, ref_med=float(np.median(ref)), **test(ref, P[ms].min(axis=1).values)) for f, ms in fams]
        adj = holm([r['p'] for r in rows_c])
        for r, q in zip(rows_c, adj): r['p_holm'] = q
        PR += rows_c
    PR = pd.DataFrame(PR); PR['sig'] = PR.p_holm < 0.05; PR['prac'] = PR.HL <= -0.05
    PR.to_csv(f'{PROC}/methods_proposal_vs_families.csv', index=False)
    g = PR[PR.B == 'gen']; q = PR[PR.B == 'proc']; o = PR[PR.B == 'dfo']
    N['PropGenSig'] = str(int(g.sig.sum())); N['PropGenPrac'] = str(int((g.sig & g.prac).sum())); N['PropCfgN'] = str(len(g))
    N['PropGenHLMin'] = f'{g.HL.abs().min():.3f}'; N['PropGenHLMax'] = f'{g.HL.abs().max():.3f}'
    N['PropGenDavMin'] = f'{g.dav.abs().min():.2f}'; N['PropGenDavMax'] = f'{g.dav.abs().max():.2f}'
    N['PropGenPMax'] = pf_tex(g.p_holm.max())
    N['PropProcSig'] = str(int(q.sig.sum())); N['PropProcPrac'] = str(int((q.sig & q.prac).sum()))
    N['PropProcDavMin'] = f'{q.dav.abs().min():.2f}'; N['PropProcDavMax'] = f'{q.dav.abs().max():.2f}'
    N['PropDfoN'] = str(len(o)); N['PropDfoSig'] = str(int(o.sig.sum())); N['PropDfoHLMax'] = f'{o.HL.abs().max():.3f}'
    N['PropDfoDavMax'] = f'{o.dav.abs().max():.2f}'
    # within-tolerance shares of each family against the reference (descriptive: the sign of G_m - G* is fixed)
    SH = []
    for cfg in CFG_ORDER:
        X = A[A.config == cfg]
        P = X.pivot_table(index=['hour', 'w', 'cut'], columns='method', values='G_fallback').dropna(axis=1, how='all').dropna()
        ref = X.drop_duplicates(['hour', 'w', 'cut']).set_index(['hour', 'w', 'cut']).G_ref.reindex(P.index).values
        row = dict(config=cfg)
        for f_, ms in [('gen', BASE_M), ('proc', PROC_M), ('dfo', DFO_M)]:
            if all(m in P for m in ms):
                sf = P[ms].min(axis=1).values - ref
                row[f'{f_}_w01'] = float((sf <= 0.01).mean()); row[f'{f_}_w05'] = float((sf <= 0.05).mean()); row[f'{f_}_q90'] = float(np.quantile(sf, 0.9))
        SH.append(row)
    SH = pd.DataFrame(SH).set_index('config'); SH.to_csv(f'{PROC}/methods_within_tolerance.csv')
    # superiority by a margin: one-sided Wilcoxon test of G_m - G_ref - EPS > 0 for the best member of each family.
    # The sign of G_m - G_ref is fixed by optimality, but the sign of G_m - G_ref - EPS is not, so this test is informative.
    # A rejection means that the family stops more than the tolerance short of the attainable. Holm over all tests.
    EPS = 0.05; MT = []
    for cfg in CFG_ORDER:
        X = A[A.config == cfg]
        P = X.pivot_table(index=['hour', 'w', 'cut'], columns='method', values='G_fallback').dropna(axis=1, how='all').dropna()
        ref = X.drop_duplicates(['hour', 'w', 'cut']).set_index(['hour', 'w', 'cut']).G_ref.reindex(P.index).values
        for f_, ms in [('gen', BASE_M), ('proc', PROC_M), ('dfo', DFO_M)]:
            if all(m in P for m in ms):
                sf = P[ms].min(axis=1).values - ref; dm = sf - EPS
                p = wilcoxon(dm, zero_method='zsplit', alternative='greater').pvalue if np.any(dm != 0) else 1.0
                MT.append(dict(config=cfg, fam=f_, n=len(sf), short_med=float(np.median(sf)), short_HL=hl(sf), rbc=rbc(dm), p=p))
    MT = pd.DataFrame(MT); MT['p_holm'] = holm(MT.p.values); MT['short'] = MT.p_holm < 0.05
    MT.to_csv(f'{PROC}/methods_margin_tests.csv', index=False)
    mg, mp, md = MT[MT.fam == 'gen'], MT[MT.fam == 'proc'], MT[MT.fam == 'dfo']
    N['MarginGenSig'] = str(int(mg.short.sum())); N['MarginProcSig'] = str(int(mp.short.sum())); N['MarginDfoSig'] = str(int(md.short.sum()))
    N['MarginCfgN'] = str(len(mg)); N['MarginDfoN'] = str(len(md)); N['MarginTests'] = str(len(MT))
    N['MarginGenHLMin'] = f'{mg.short_HL.min():.2f}'; N['MarginGenHLMax'] = f'{mg.short_HL.max():.2f}'
    N['MarginGenPMax'] = pf_tex(mg.p_holm.max())
    # hour blocks: requests of one hour share data, so repeat the margin test on hour means and bootstrap the
    # within-0.05 shares over hours. With 6 to 8 hours the smallest attainable one-sided p is 1/64 or 1/256.
    HB, rngh = [], np.random.default_rng(3)
    for cfg in CFG_ORDER:
        X = A[A.config == cfg]
        P = X.pivot_table(index=['hour', 'w', 'cut'], columns='method', values='G_fallback').dropna(axis=1, how='all').dropna()
        ref = X.drop_duplicates(['hour', 'w', 'cut']).set_index(['hour', 'w', 'cut']).G_ref.reindex(P.index).values
        hrs = np.array(P.index.get_level_values(0)); uh = np.unique(hrs)
        for f_, ms in [('gen', BASE_M), ('proc', PROC_M), ('dfo', DFO_M)]:
            if not all(m in P for m in ms): continue
            sf = pd.Series(P[ms].min(axis=1).values - ref, index=hrs); hm = sf.groupby(level=0).mean()
            ph = wilcoxon(hm.values - EPS, alternative='greater').pvalue if np.any(hm.values != EPS) else 1.0
            sh = []
            for _ in range(5000):
                hs = rngh.choice(uh, len(uh)); v = np.concatenate([sf.values[hrs == h] for h in hs]); sh.append((v <= 0.05).mean())
            HB.append(dict(config=cfg, fam=f_, n_hours=len(hm), blocks_above=int((hm > EPS).sum()), p_hour=ph,
                           w05=float((sf <= 0.05).mean()), w05_lo=float(np.quantile(sh, 0.025)), w05_hi=float(np.quantile(sh, 0.975)),
                           med=float(np.median(sf.values))))
    HB = pd.DataFrame(HB)
    for f_ in HB.fam.unique():
        k = HB.fam == f_; HB.loc[k, 'p_hour_holm'] = holm(HB[k].p_hour.values)
    HB.to_csv(f'{PROC}/methods_hour_blocks.csv', index=False)
    hg, hp, hd = HB[HB.fam == 'gen'], HB[HB.fam == 'proc'], HB[HB.fam == 'dfo']
    N['HourGenAbove'] = str(int(hg.blocks_above.sum())); N['HourGenBlocks'] = str(int(hg.n_hours.sum()))
    N['HourGenMinP'] = str(int((hg.p_hour <= 1 / 2 ** hg.n_hours + 1e-12).sum()))
    N['HourGenHolmSig'] = str(int((hg.p_hour_holm < 0.05).sum()))
    N['HourGenWHi'] = f'{100 * hg.w05_hi.max():.0f}'; N['HourDfoWLo'] = f'{100 * hd.w05_lo.min():.0f}'
    N['HourProcWLo'] = f'{100 * hp.w05_lo.min():.0f}'; N['HourProcWHi'] = f'{100 * hp.w05_hi.max():.0f}'
    N['WithinGenFiveMin'] = f'{100*SH.gen_w05.min():.0f}'; N['WithinGenFiveMax'] = f'{100*SH.gen_w05.max():.0f}'
    N['WithinProcFiveMin'] = f'{100*SH.proc_w05.min():.0f}'; N['WithinProcFiveMax'] = f'{100*SH.proc_w05.max():.0f}'
    N['WithinDfoOneMin'] = f'{100*SH.dfo_w01.dropna().min():.0f}'
    HBi = HB.set_index(['config', 'fam'])
    def ci(cfg, fam_):
        if (cfg, fam_) not in HBi.index: return '--'
        r_ = HBi.loc[(cfg, fam_)]
        return f'{100*r_.w05:.0f} [{100*r_.w05_lo:.0f}, {100*r_.w05_hi:.0f}]'
    with open(f'{TAB}/tab_proposal_tests.tex', 'w') as f:
        f.write('\\begin{tabular}{@{}lrrrlllrrrrlrrr@{}}\n\\toprule\n')
        f.write(' & & & & \\multicolumn{6}{c}{Quality, against the reference} & \\multicolumn{3}{c}{Efficiency, Rep. vs. CF} & \\multicolumn{2}{c}{DFO vs. Rep.} \\\\\n')
        f.write('\\cmidrule(lr){5-10}\\cmidrule(lr){11-13}\\cmidrule(l){14-15}\n')
        f.write(' & & & & \\multicolumn{3}{c}{Within 0.05 (\\%) [95\\% CI over hours]} & $p$, hours & \\multicolumn{2}{c}{$p_{\\mathrm{Holm}}$, requests} & & & & & \\\\\n')
        f.write('Configuration & $n$ & $H$ & $G^{*}$ & CF & Rep. & DFO & CF & CF & Rep. & HL & $d_{\\mathrm{av}}$ [95\\% CI] & $p_{\\mathrm{Holm}}$ & HL & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n')
        def pf(p): return '$<$0.001' if p < 0.001 else f'{p:.3f}'
        for cfg in CFG_ORDER:
            R_ = PR[PR.config == cfg].set_index('B'); mt = MT[MT.config == cfg].set_index('fam')
            a = T[(T.config == cfg) & (T.A == 'proc') & (T.B == 'base')].iloc[0]
            b = T[(T.config == cfg) & (T.A == 'dfo') & (T.B == 'proc')]
            hb = HBi.loc[(cfg, 'gen')]
            sb = f'${b.iloc[0].HL:+.3f}$ & {pf(b.iloc[0].p_holm)}' if len(b) else '-- & --'
            f.write(f'{CFG_LABEL[cfg]} & {int(R_.iloc[0].n)} & {int(hb.n_hours)} & {R_.iloc[0].ref_med:.3f} & {ci(cfg, "gen")} & {ci(cfg, "proc")} & {ci(cfg, "dfo")} & '
                    f'{pf(hb.p_hour_holm)} & {pf(mt.loc["gen"].p_holm)} & {pf(mt.loc["proc"].p_holm)} & ${a.HL:+.3f}$ & ${a.dav:+.2f}$ [${a.dav_lo:+.2f}$, ${a.dav_hi:+.2f}$] & {pf(a.p_holm)} & {sb} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    # descriptive version for the manuscript: shares, hours above the tolerance and median shortfalls, no tests.
    # The full table with the tests of the O5 criteria (RQ5 of docs/evaluation_criteria.md, tab_proposal_tests.tex) goes to the supplement.
    with open(f'{TAB}/tab_proposal_desc.tex', 'w') as f:
        f.write('\\begin{tabular}{@{}lrrrrrrrrrrrr@{}}\n\\toprule\n')
        f.write(' & & & & \\multicolumn{3}{c}{Within 0.05 (\\%)} & \\multicolumn{3}{c}{Hours above $\\varepsilon$} & \\multicolumn{3}{c}{Median $\\Delta$} \\\\\n')
        f.write('\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-13}\n')
        f.write('Configuration & $n$ & $H$ & $G^{*}$ & CF & Rep. & DFO & CF & Rep. & DFO & CF & Rep. & DFO \\\\\n\\midrule\n')
        for cfg in CFG_ORDER:
            R_ = PR[PR.config == cfg].set_index('B'); hb = HBi.loc[(cfg, 'gen')]
            def cell(fam_, k):
                if (cfg, fam_) not in HBi.index: return '--'
                r_ = HBi.loc[(cfg, fam_)]
                return f'{100*r_.w05:.0f}' if k == 'w' else (f'{int(r_.blocks_above)}' if k == 'h' else f'{max(r_.med, 0.0):.3f}')
            f.write(f'{CFG_LABEL[cfg]} & {int(R_.iloc[0].n)} & {int(hb.n_hours)} & {R_.iloc[0].ref_med:.3f} & '
                    + ' & '.join(cell(fm, k) for k in ('w', 'h', 'm') for fm in ('gen', 'proc', 'dfo')) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    # hour-cluster bootstrap of d_av for the equal-budget family comparisons (sensitivity to the resampling unit)
    CB = []
    rngc = np.random.default_rng(1)
    for cfg in CFG_ORDER:
        P = A[A.config == cfg].pivot_table(index=['hour', 'w', 'cut'], columns='method', values='G_fallback').dropna(axis=1, how='all').dropna()
        pairs = [('proc', 'base', P[PROC_M].min(axis=1), P[BASE_M].min(axis=1))]
        if all(m in P for m in DFO_M): pairs.append(('dfo', 'proc', P[DFO_M].min(axis=1), P[PROC_M].min(axis=1)))
        for a_, b_, xa, xb in pairs:
            hours = np.array(P.index.get_level_values(0)); uh = np.unique(hours); vals = []
            for _ in range(5000):
                hs = rngc.choice(uh, len(uh)); idx = np.concatenate([np.where(hours == h)[0] for h in hs])
                xa_, xb_ = xa.values[idx], xb.values[idx]; sd = (xa_.std(ddof=1) + xb_.std(ddof=1)) / 2
                vals.append(np.nan if sd == 0 else (xa_ - xb_).mean() / sd)
            lo, hi = np.nanquantile(vals, [0.025, 0.975]); CB.append(dict(config=cfg, A=a_, B=b_, dav_lo_hour=lo, dav_hi_hour=hi))
    CB = pd.DataFrame(CB); CB.to_csv(f'{PROC}/methods_cluster_bootstrap.csv', index=False)
    pb = CB[(CB.A == 'proc')]; N['ClusterProcBaseExcl'] = str(int((pb.dav_hi_hour < 0).sum()))
    db = CB[(CB.A == 'dfo')]; N['ClusterDfoProcExcl'] = str(int((db.dav_hi_hour < 0).sum()))
    # per-method comparison with the proposal at default (Table of the appendix)
    D0 = A[A.config == 'default']
    P0 = D0.pivot_table(index=['hour', 'w', 'cut'], columns='method', values='G_fallback').dropna()
    ref0 = D0.drop_duplicates(['hour', 'w', 'cut']).set_index(['hour', 'w', 'cut']).G_ref.reindex(P0.index).values
    PM = {m: test(ref0, P0[m].values) for m in PROC_M + BASE_M + DFO_M}
    adj = holm([PM[m]['p'] for m in PM])
    for m, q_ in zip(PM, adj): PM[m]['p_holm'] = q_
    gm = [m for m in BASE_M]
    N['PropMethGenDavMin'] = f'{min(abs(PM[m]["dav"]) for m in gm):.2f}'; N['PropMethGenDavMax'] = f'{max(abs(PM[m]["dav"]) for m in gm):.2f}'
    # per-method table at default
    D = A[A.config == 'default']
    order = PROC_M + BASE_M + DFO_M
    worse = rb[rb.config == 'default'].groupby('method').G.apply(lambda g: (g > 1 + 1e-9).mean())
    with open(f'{TAB}/tab_methods_default.tex', 'w') as f:
        f.write('\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\nFamily & Method & Median $G_m$ & Median shortfall & $d_{\\mathrm{av}}$ vs. reference [95\\% CI] & Worse than inc. & Calls \\\\\n\\midrule\n')
        for fam, ms in [('Repaired', PROC_M), ('CF-adapted', BASE_M), ('DFO', DFO_M)]:
            for i, m in enumerate(ms):
                x = D[D.method == m]
                w_ = f'{100*worse.get(m, 0):.0f}\\%' if m in worse else '0\\%'
                t_ = PM[m]; pp = '$<$0.001' if t_['p_holm'] < 0.001 else f"{t_['p_holm']:.3f}"
                f.write(f'{fam if i == 0 else ""} & {LABEL[m]} & {x.G_fallback.median():.3f} & {x.shortfall.median():.3f} & ${t_["dav"]:+.2f}$ [${t_["dav_lo"]:+.2f}$, ${t_["dav_hi"]:+.2f}$] & {w_} & {x.evals.mean():.0f} \\\\\n')
            if fam != 'DFO': f.write('\\addlinespace\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    N['DiceWorseMax'] = f'{100*rb[rb.method=="DiCE"].groupby("config").G.apply(lambda g:(g>1+1e-9).mean()).max():.0f}'
    N['DiceWorseDefault'] = f'{100*worse.get("DiCE",0):.0f}'
    N['NiceCalls'] = f'{D[D.method=="NICE_R"].evals.mean():.0f}'
    N['CmaSpreadMed'] = f'{cma_spread.median():.3f}'
    T50 = A[A.config == 'allow50_005']           # tight allowance: attainable gap and best repaired procedure
    N['TightGrefMed'] = f'{T50.drop_duplicates(["hour", "w", "cut"]).G_ref.median():.3f}'
    N['TightBestProcShort'] = f'{min(T50[T50.method == m].shortfall.median() for m in PROC_M):.3f}'
    for m in order:
        N[f'Short{m.replace("_","")}'] = f'{D[D.method==m].shortfall.median():.3f}'
    with open(f'{MAC}/numbers_methods.tex', 'w') as f:
        for k, v in sorted(N.items()): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(N, open(f'{PROC}/numbers_methods.json', 'w'), indent=1)
    print(json.dumps(N, indent=1))
    print(T[T.A.isin(['proc', 'dfo'])][['config', 'A', 'B', 'n', 'med_a', 'med_b', 'HL', 'rbc', 'p_holm', 'A_wins']].round(4).to_string())


if __name__ == '__main__':
    main()
