"""Further analyses: counterfactuals computed with each search method, ablations of the method comparison, the SCIP
checks of the 10-unit anchors, incumbents and targets, and the 40-unit campaign under both emission conventions. Writes results/macros/numbers_additional2.tex, results/processed/numbers_additional2.json and the processed
tables named below. Every function returns silently when its raw files are missing."""
import glob, sys, json, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
from common import *
from eeldattain import paths

REV = str(paths.RAW / 'additional')
N = {}
EPS = 0.05
GEN, PROC_, DFO = ['GS', 'FACE', 'NICE_R'], ['POLY', 'ELLP', 'CGM'], ['SLSQP_FD', 'CMAES']
B_ = 10000


def fmt(x, d=2):
    return f'{x:.{d}f}'


def boot(x, B=B_, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    v = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return np.quantile(v, [0.025, 0.975])


def boot_hours(x, hours, B=5000, seed=1):
    x = np.asarray(x, float); hours = np.asarray(hours); uh = np.unique(hours); rng = np.random.default_rng(seed)
    idx = {h: np.where(hours == h)[0] for h in uh}
    v = []
    for _ in range(B):
        hs = rng.choice(uh, len(uh)); v.append(x[np.concatenate([idx[h] for h in hs])].mean())
    return np.quantile(v, [0.025, 0.975])


def first_pass(g, grid, thr):
    """Smallest grid value whose gap meets the threshold, NaN when none does."""
    ok = [x for x, G in zip(grid, g) if G <= thr + 1e-9]
    return min(ok) if ok else np.nan


def cf_with_methods():
    fs = sorted(glob.glob(f'{REV}/cf_with_methods_p*.jsonl'))
    if not fs:
        return
    D = pd.concat([pd.read_json(f, lines=True) for f in fs], ignore_index=True)
    key = ['hour', 'w', 'cut']
    # the first requests were run by an earlier version of the script (with COBYQA, without REF_NOTAR) in sorted
    # order before the random order was introduced; they are excluded, which keeps the sample random
    old = D[D.method == 'COBYQA'][key].drop_duplicates()
    D = D.merge(old.assign(_old=1), on=key, how='left'); D = D[D._old.isna()].drop(columns='_old')
    N['CfmExcludedOld'] = str(len(old))
    W = D.pivot_table(index=key + ['kind', 'value'], columns='method', values='G').reset_index()
    for fam, ms in (('GEN', GEN), ('PROC', PROC_), ('DFO', DFO)):
        W[fam] = W[ms].min(axis=1)
    thr = D.drop_duplicates(key).set_index(key)[['thr', 'rho_star', 'R_star', 'G_phys']]
    rows = []
    methods = ['REF', 'REF_NOTAR'] + GEN + PROC_ + DFO + ['GEN', 'PROC', 'DFO']
    for (h, w, c, kind), X in W.groupby(key + ['kind']):
        X = X.sort_values('value'); t = thr.loc[(h, w, c)]
        grid = X.value.values
        r = dict(hour=h, w=w, cut=c, kind=kind, n_grid=len(grid), stored=t.rho_star if kind == 'rho' else t.R_star)
        for m in methods:
            r[m] = first_pass(X[m].values, grid, t.thr)
        rows.append(r)
    C = pd.DataFrame(rows)
    complete = C.groupby(key).kind.transform('count') == 2
    C = C[complete & (C.n_grid >= 9)]
    C.to_csv(f'{PROC}/cf_with_methods.csv', index=False)
    N['CfmReq'] = str(C.drop_duplicates(key).shape[0])
    # consistency of the grid reference with the stored bisection: the stored value lies at or below the grid value
    # up to one bisection step
    for kind, nm in (('rho', 'Rho'), ('R', 'R')):
        X = C[C.kind == kind]
        step = 0.95 / 256 if kind == 'rho' else 1000 / 256      # one bisection step of the campaign
        ok = (X.stored <= X.REF + step + 1e-9)
        N[f'Cfm{nm}RefConsistent'] = fmt(100 * ok.mean(), 0)
        N[f'Cfm{nm}RefNan'] = str(int(X.REF.isna().sum()))
    out = []
    for kind in ('rho', 'R'):
        X = C[C.kind == kind]
        for m in methods[1:]:
            same = (X[m] == X.REF); over = X[m] > X.REF; none = X[m].isna() & X.REF.notna(); under = X[m] < X.REF
            wrong = (over | none).astype(float).values
            # exact Clopper-Pearson interval over requests (a bootstrap degenerates to [1, 1] when every request fails)
            ci = binomtest(int(wrong.sum()), len(wrong)).proportion_ci(confidence_level=0.95, method='exact')
            lo, hi = ci.low, ci.high; hlo, hhi = boot_hours(wrong, X.hour.values)
            ratio = (X[m] / X.REF)[over]
            out.append(dict(kind=kind, method=m, n=len(X), k_same=int(same.sum()), k_over=int(over.sum()), k_none=int(none.sum()),
                            k_under=int(under.sum()), k_fail=int(wrong.sum()),
                            same=same.mean(), over=over.mean(), none=none.mean(), under=under.mean(),
                            wrong=wrong.mean(), wrong_lo=lo, wrong_hi=hi, wrong_hlo=hlo, wrong_hhi=hhi,
                            over_ratio_med=ratio.median() if len(ratio) else np.nan,
                            over_mw_med=(X[m] - X.REF)[over].median() if kind == 'R' and over.any() else np.nan))
    O = pd.DataFrame(out); O.to_csv(f'{PROC}/cf_with_methods_summary.csv', index=False)
    # wall-clock time of one grid counterfactual (all grid values) per method family, median over requests and kinds
    Dk = D.merge(C[key + ['kind']], on=key + ['kind'])
    fam = {**{m: 'Gen' for m in GEN}, **{m: 'Proc' for m in PROC_}, **{m: 'Dfo' for m in DFO}, 'REF': 'Ref', 'REF_NOTAR': 'Notar'}
    Dk['fam'] = Dk.method.map(fam)
    T_ = Dk.groupby(key + ['kind', 'fam']).runtime.sum().groupby('fam').median()
    for f_, v in T_.items():
        N[f'CfmTime{f_}'] = fmt(v, 0)
    for kind, nm in (('rho', 'Rho'), ('R', 'R')):
        o = O[O.kind == kind].set_index('method')
        for m, mm in (('GEN', 'Gen'), ('PROC', 'Proc'), ('DFO', 'Dfo'), ('REF_NOTAR', 'Notar'), ('GS', 'Gs')):
            N[f'Cfm{nm}{mm}Wrong'] = fmt(100 * o.loc[m, 'wrong'], 0)
            N[f'Cfm{nm}{mm}WrongLo'] = fmt(100 * o.loc[m, 'wrong_lo'], 0); N[f'Cfm{nm}{mm}WrongHi'] = fmt(100 * o.loc[m, 'wrong_hi'], 0)
            N[f'Cfm{nm}{mm}WrongHLo'] = fmt(100 * o.loc[m, 'wrong_hlo'], 0); N[f'Cfm{nm}{mm}WrongHHi'] = fmt(100 * o.loc[m, 'wrong_hhi'], 0)
            N[f'Cfm{nm}{mm}None'] = fmt(100 * o.loc[m, 'none'], 0); N[f'Cfm{nm}{mm}Over'] = fmt(100 * o.loc[m, 'over'], 0)
            N[f'Cfm{nm}{mm}Under'] = str(int(round(o.loc[m, 'under'] * o.loc[m, 'n'])))
            for cat in ('same', 'over', 'none', 'fail'):
                N[f'Cfm{nm}{mm}{cat.capitalize()}K'] = str(int(o.loc[m, f'k_{cat}']))
            if np.isfinite(o.loc[m, 'over_ratio_med']):
                N[f'Cfm{nm}{mm}Ratio'] = fmt(o.loc[m, 'over_ratio_med'], 1)
            if kind == 'R' and np.isfinite(o.loc[m, 'over_mw_med']):
                N[f'Cfm{nm}{mm}ExtraMw'] = fmt(o.loc[m, 'over_mw_med'], 0)
        N[f'Cfm{nm}N'] = str(int(o.n.iloc[0]))
        N[f'Cfm{nm}UnderAll'] = str(int(o.k_under.sum()))
    # LaTeX table: share of wrong instructions per method and kind
    lab = {'REF_NOTAR': 'Multistart, no target', 'GS': 'GS-R', 'FACE': 'FACE-R', 'NICE_R': 'NICE-R', 'POLY': 'POLY',
           'ELLP': 'ELLP', 'CGM': 'CGM', 'SLSQP_FD': 'SLSQP-FD', 'CMAES': 'CMA-ES', 'GEN': 'CF generators',
           'PROC': 'Repaired proc.', 'DFO': 'DFO methods'}
    with open(f'{TAB}/tab_cf_with_methods.tex', 'w') as f:
        nr, nb = int(O[O.kind == 'rho'].n.iloc[0]), int(O[O.kind == 'R'].n.iloc[0])
        f.write('\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n & \\multicolumn{4}{c}{Band $\\rho^{*}$, $n=%d$} & \\multicolumn{4}{c}{Budget $R^{*}$, $n=%d$} \\\\\n' % (nr, nb))
        f.write('\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\nInner solver & Equal & Over & None & Fail [CI] & Equal & Over & None & Fail [CI] \\\\\n\\midrule\n')
        for m in ['GEN', 'PROC', 'DFO', 'REF_NOTAR']:      # per-method rows: results/processed/cf_with_methods_summary.csv
            a = O[(O.kind == 'rho') & (O.method == m)].iloc[0]; b = O[(O.kind == 'R') & (O.method == m)].iloc[0]
            cell = lambda r: f'{r.k_same} & {r.k_over} & {r.k_none} & {100*r.wrong:.0f} [{100*r.wrong_lo:.0f}, {100*r.wrong_hi:.0f}]'
            f.write(f'{lab[m]} & {cell(a)} & {cell(b)} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')


def ablation():
    f = f'{REV}/ablation_generators.csv'
    if os.path.exists(f):
        A = pd.read_csv(f)
        rb = pd.concat([pd.read_csv(x) for x in glob.glob(str(paths.RAW_ROBUST) + '/raw_[ABC]*.csv')]).drop_duplicates(['config', 'hour', 'w', 'cut', 'method'])
        # The rerun regenerates the instances. In hour 1 its multistart anchors differ from the stored ones (BLAS-thread
        # dependence of the multistart solver), which moves the incumbents between weights, so the
        # reference of the same rerun (ablation_reference.csv, variant full) is used rather than the stored G_ref.
        RF = pd.read_csv(f'{REV}/ablation_reference.csv'); RF = RF[RF.variant == 'full']
        ref = RF.set_index(['config', 'hour', 'w', 'cut']).G
        st_ref = rb.drop_duplicates(['config', 'hour', 'w', 'cut']).set_index(['config', 'hour', 'w', 'cut']).G_ref
        ch = (ref - st_ref.reindex(ref.index)).abs() > 1e-4
        N['AblInstChanged'] = str(int(ch[ch.index.get_level_values(0) == 'default'].sum()))
        rows = []
        for cfg, X in A.groupby('config'):
            P = X.pivot_table(index=['hour', 'w', 'cut'], columns=['method', 'setting'], values='G_fallback')
            r_ = ref.loc[cfg].reindex(P.index).values
            stored = P.xs('stored', level='setting', axis=1).min(axis=1).values
            tuned = P.min(axis=1).values
            for nm, v in (('stored', stored), ('tuned', tuned)):
                sf = v - r_; dm = sf - EPS
                p = wilcoxon(dm, zero_method='zsplit', alternative='greater').pvalue if np.any(dm != 0) else 1.0
                rows.append(dict(config=cfg, variant=nm, n=len(sf), w05=(sf <= 0.05).mean(), short_med=np.median(sf), p=p))
            best = {m: P[m].min(axis=1) for m in P.columns.get_level_values(0).unique()}
            gain = {m: float((P[m]['stored'] - best[m]).median()) for m in best}
            rows[-1]['gain'] = json.dumps(gain)
        T = pd.DataFrame(rows); T.to_csv(f'{PROC}/ablation_generators_summary.csv', index=False)
        t = T[T.variant == 'tuned'].set_index('config'); s_ = T[T.variant == 'stored'].set_index('config')
        N['AblTunedWDef'] = fmt(100 * t.loc['default', 'w05'], 0); N['AblTunedWTight'] = fmt(100 * t.loc['allow50_005', 'w05'], 0)
        N['AblStoredWDef'] = fmt(100 * s_.loc['default', 'w05'], 0); N['AblStoredWTight'] = fmt(100 * s_.loc['allow50_005', 'w05'], 0)
        N['AblTunedShortDef'] = fmt(t.loc['default', 'short_med'], 3); N['AblTunedShortTight'] = fmt(t.loc['allow50_005', 'short_med'], 3)
        N['AblTunedPMax'] = f'{t.p.max():.1e}' if t.p.max() < 1e-3 else fmt(t.p.max(), 3)
        N['AblTunedSig'] = str(int((t.p * len(t) < 0.05).sum())); N['AblCfgN'] = str(len(t))
        N['AblSettings'] = str(A.drop_duplicates(['method', 'setting']).shape[0])
    f = f'{REV}/ablation_reference.csv'
    if os.path.exists(f):
        R = pd.read_csv(f)
        P = R.pivot_table(index=['config', 'hour', 'w', 'cut'], columns='variant', values='G')
        for v, nm in (('notar', 'Notar'), ('noanch', 'Noanch'), ('inc', 'Inc')):
            d = P[v] - P['full']
            N[f'AblRef{nm}Med'] = fmt(d.median(), 3); N[f'AblRef{nm}Within'] = fmt(100 * (d <= 0.01).mean(), 0)
            N[f'AblRef{nm}Max'] = fmt(d.max(), 3)
        N['AblRefN'] = str(len(P)); N['AblRefTimeMed'] = fmt(R[R.variant == 'full'].runtime.median(), 2)


def scip_anchors():
    f = f'{REV}/scip/anchors.csv'; g = sorted(glob.glob(f'{REV}/scip/incumbents_targets_p*.csv'))
    if not os.path.exists(f) or not g:
        return
    A = pd.read_csv(f); X = pd.concat([pd.read_csv(x) for x in g])
    cs = (A.Cmin_stored - A.Cmin_lb) / A.Cmin_lb; es = (A.Emin_stored - A.Emin_lb) / A.Emin_lb
    N['ScipAnchHours'] = str(len(A)); N['ScipAnchMax'] = f'{max(cs.max(), es.max()):.0e}'
    N['ScipAnchStoredLocal'] = str(int((cs > 1e-5).sum()))
    inc = X.drop_duplicates(['hour', 'w']); ie = (inc.C_inc - inc.inc_lb) / inc.inc_lb
    t = X.dropna(subset=['tar_lb']); te = (t.C_tar - t.tar_lb) / t.tar_lb
    N['ScipIncN'] = str(len(inc)); N['ScipIncMax'] = f'{max(ie.max(), 1e-12):.0e}'
    fi = paths.RAW / 'additional' / 'scip' / 'incumbents_inverse.csv'
    if fi.exists():                     # inverse check, min E s.t. C <= C(P_inc)
        I = pd.read_csv(fi)
        N['ScipIncInvN'] = str(len(I)); N['ScipIncInvMax'] = f'{max(I.rel_excess.max(), 1e-12):.0e}'
    N['ScipTarN'] = str(len(t)); N['ScipTarOk'] = str(int((te <= 1e-5).sum())); N['ScipTarBad'] = str(int((te > 1e-5).sum()))
    N['ScipTarMax'] = f'{max(te.max(), 1e-12):.0e}'
    N['ScipAnchTime'] = fmt(A.t.median() / 2, 1)
    # the SCIP decisions inside the certified campaign rerun (hours whose multistart points SCIP improved)
    L = []
    for c in sorted(glob.glob(str(paths.RAW_ATTAIN / 'certify_e10_h*.json'))):
        h = int(c.split('_h')[-1].split('.')[0])
        L += [dict(hour=h, **r) for r in json.load(open(c))]
    if L:
        L = pd.DataFrame(L); ad = L[L.adopted.astype(bool)]
        N['CertRerunHours'] = ', '.join(str(h) for h in sorted(L.hour.unique()))
        a = ad[ad.kind == 'C']
        N['CertRerunCminHours'] = str(a.hour.nunique())
        N['CertRerunCminMaxPct'] = fmt(100 * ((a.v_multistart - a.v_final) / a.v_final).max(), 2) if len(a) else '0'
    # the same check inside the primary 40-unit campaign (120 s per solve, often stopped by the time limit)
    L = []
    for c in sorted(glob.glob(str(paths.RAW_ATTAIN / 'certify_e40_h*.json'))):
        h = int(c.split('_h')[-1].split('.')[0])
        L += [dict(hour=h, i=i, **r) for i, r in enumerate(json.load(open(c)))]
    if L:
        L = pd.DataFrame(L); closed = L.scip_status.isin(['optimal', 'gaplimit'])
        an = L[L.Ecap.isna() & (L.kind != 'W')].drop_duplicates(['hour', 'kind'])      # first C and E entries are the anchors
        tar = L[L.Ecap.notna()]; ac = an[an.kind == 'C']; ae = an[an.kind == 'E']
        gap = lambda X: 100 * (X.v_final - X.scip_lb) / X.v_final.abs()
        N['FortyLevels'] = str(L.hour.nunique())
        N['FortyEminCert'] = str(int(closed[ae.index].sum())); N['FortyCminCert'] = str(int(closed[ac.index].sum()))
        N['FortyCminGapMax'] = fmt(gap(ac).max(), 1)
        N['FortyCminAdopt'] = str(int(ac.adopted.astype(bool).sum()))
        N['FortyCminAdoptMax'] = fmt(100 * ((ac.v_multistart - ac.v_final) / ac.v_final).max(), 1)
        N['FortyTarN'] = str(len(tar)); N['FortyTarAdopt'] = str(int(tar.adopted.astype(bool).sum()))
        N['FortyTarCert'] = str(int(closed[tar.index].sum())); N['FortyTarGapMax'] = fmt(gap(tar).max(), 1)
        w = L[L.kind == 'W']; N['FortyIncN'] = str(len(w)); N['FortyIncAdopt'] = str(int(w.adopted.astype(bool).sum()))


def read_ckpt(path):
    """Read a checkpointed requests file whose rows were appended with different key sets. Each row keeps the
    insertion order of run.py: the base keys and G_alt, then rho_star and R_star when a counterfactual was needed,
    then the five unit-set keys when the budget-free gap still exceeded the threshold, and solver_fallbacks last."""
    import csv
    rows = list(csv.reader(open(path)))
    if all(len(r) == len(rows[0]) for r in rows):
        return pd.read_csv(path, dtype={'S_units': str})
    i_alt = rows[0].index('G_alt'); base = rows[0][:i_alt + 1]
    cf = ['rho_star', 'R_star']; sets = ['n_pinned', 'S_size', 'S_units', 'S_closed', 'S_exact_check']
    head = base + cf + sets + ['solver_fallbacks']
    out = []
    for r in rows[1:]:
        if len(r) == len(head): keys = head
        elif len(r) == len(base) + 3: keys = base + cf + ['solver_fallbacks']
        elif len(r) == len(base) + 1: keys = base + ['solver_fallbacks']
        else: raise ValueError(f'{path}: unexpected row length {len(r)}')
        out.append(dict(zip(keys, r)))
    D = pd.DataFrame(out, columns=head).replace('', np.nan)
    for c in ('reachable', 'S_closed'):
        if c in D:
            D[c] = D[c].map({'True': True, 'False': False})
    for c in D.columns:
        if c not in ('S_units', 'reachable', 'S_closed'):
            try:
                D[c] = pd.to_numeric(D[c])
            except (ValueError, TypeError):
                pass
    return D


def forty_printed():
    """40-unit variant with the printed emission convention of the data source (factor 1e-2 on the emission
    polynomial), run with the protocol of the primary campaign (experiments/additional/forty_main.py --escale 0.01)."""
    d = paths.RAW / 'additional' / 'forty_printed'
    if not glob.glob(str(d / 'requests_e40*.csv')):
        return None
    R = pd.concat([read_ckpt(f) for f in sorted(glob.glob(str(d / 'requests_e40*.csv')))], ignore_index=True)
    G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(d / 'grid_e40*.csv')))], ignore_index=True)
    return R, G


def summ40(R, G):
    u = R[R.duplicate_of_w.isna()].copy()
    g = G[(G.kind == 'grid')].merge(u[['hour', 'w', 'cut']], on=['hour', 'w', 'cut'])
    reg = g[g.R.isin([100, 250, 500]) & g.rho.isin([0.05, 0.10, 0.20])].copy(); reg['A'] = (reg.G - reg.G_phys).clip(lower=0)
    per = reg.groupby(['hour', 'w', 'cut']).A.mean()
    need = u[u.G_ref > u.G_phys + EPS].copy()
    z = need.G_Rfree <= need.G_phys + EPS; need.loc[z, 'rho_star'] = need.loc[z, 'rho0']
    pg = g.pivot_table(index='R', columns='rho', values='G', aggfunc='mean')
    tot = need.G_ref - need.G_phys
    fR = (need.G_ref - need.G_Rfree) / tot; fr = (need.G_ref - need.G_rhofree) / tot
    hw = u.groupby('hour').agg(W=('front_width', 'first'))
    sets = need[need.G_Rfree > need.G_phys + EPS].dropna(subset=['S_size']) if 'S_size' in need else need.iloc[0:0]
    return dict(n=len(u), reach=100 * u.reachable.mean(), eta=u.eta_E.mean(), wmin=100 * hw.W.min(), wmax=100 * hw.W.max(),
                allow=100 * (per >= 0.10).mean(), need=len(need), rho=need.rho_star.median(), Rs=need.R_star.median(),
                sat=max(pg.loc[250, r] - pg.loc[500, r] for r in pg.columns if r <= 0.10),
                satB=max(pg.loc[100, r] - pg.loc[250, r] for r in pg.columns if r <= 0.10),
                joint=100 * ((fR < 0.7) & (fr < 0.7)).mean(), le2=100 * (sets.S_size <= 2).mean() if len(sets) else np.nan,
                nset=len(sets), nbudget=int(z.sum()), gphys=u.G_phys.mean(), gref=u.G_ref.mean())


def forty_conventions():
    """The primary 40-unit campaign (emission without the factor 1e-2, results/raw/attainability) against the variant
    with the printed factor. Macros Pub* hold the primary campaign and Pub*St the printed variant."""
    M = forty_printed()
    if M is None or not glob.glob(str(paths.RAW_ATTAIN / 'requests_e40*.csv')):
        return
    P = (pd.concat([read_ckpt(f) for f in sorted(glob.glob(str(paths.RAW_ATTAIN / 'requests_e40*.csv')))], ignore_index=True),
         pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(paths.RAW_ATTAIN / 'grid_e40*.csv')))], ignore_index=True))
    a = summ40(*M); b = summ40(*P)
    name = dict(n='N', reach='Reach', eta='Eta', wmin='Wmin', wmax='Wmax', allow='Allow', need='Need', rho='Rho', Rs='RStar',
                sat='Sat', satB='SatB', joint='Joint', le2='LeTwo', gphys='Gphys', gref='Gref', nset='SetN', nbudget='BudgetOnly')
    for k in a:
        d = 3 if k in ('rho', 'eta', 'sat', 'satB', 'gphys', 'gref') else 0 if k in ('n', 'need', 'Rs', 'allow', 'joint', 'le2', 'reach', 'nset', 'nbudget') else 1
        N[f'Pub{name[k]}St'] = fmt(a[k], d); N[f'Pub{name[k]}'] = fmt(b[k], d)
    json.dump(dict(printed=a, primary=b), open(f'{PROC}/forty_conventions_summary.json', 'w'), indent=1, default=float)


def main():
    cf_with_methods(); ablation(); scip_anchors(); forty_conventions(); closed_form_phys()
    methods_extended(); scip_direct(); ramp_grid(); decomposition(); consistency_checks(); onesided(); forty_region()
    with open(f'{MAC}/numbers_additional2.tex', 'w') as f:
        for k, v in sorted(N.items()):
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(N, open(f'{PROC}/numbers_additional2.json', 'w'), indent=1)
    print(json.dumps(N, indent=1))



def cat(pattern, **kw):
    fs = sorted(glob.glob(str(pattern)))
    return pd.concat([pd.read_csv(f, **kw) for f in fs], ignore_index=True) if fs else None


def consistency_checks():
    """Checks from stored files: the direct counterfactuals at the floor rho_0, an hour-resampled interval for the
    allowance share of RQ2(a), the window gap without locked schedules (RQ10), the floor of the extension systems
    and the SCIP times per system."""
    X = pd.read_csv(f'{REV}/direct_cf.csv'); st = 0.95 / 256
    first = X.rho_star <= 0.05 + st + 1e-9; dr = X.rho_direct - X.rho_star; off = dr.abs() > st + 1e-9
    N['DirFloorN'] = str(int(first.sum())); N['DirOffN'] = str(int(off.sum()))
    N['DirOffAtFloor'] = str(int((off & first).sum())); N['DirOffBelowRhoZero'] = str(int((off & (X.rho_direct < 0.05)).sum()))
    N['DirRestN'] = str(int((~first).sum())); N['DirRestWithin'] = str(int((~first & ~off).sum()))
    N['DirOffMin'] = fmt(X.rho_direct[off].min(), 3)
    u = cat(f'{paths.RAW_ATTAIN}/requests_e10*.csv', dtype={'S_units': str}); u = u[u.duplicate_of_w.isna()]
    g = cat(f'{paths.RAW_ATTAIN}/grid_e10*.csv')
    g = g[(g.kind == 'grid') & g.R.isin([25, 50, 100]) & g.rho.isin([0.05, 0.1, 0.2])].merge(u[['hour', 'w', 'cut']], on=['hour', 'w', 'cut'])
    m = (g.G - g.G_phys).clip(lower=0).groupby([g.hour, g.w, g.cut]).mean().reset_index(name='a')
    lo, hi = boot_hours((m.a >= 0.1).astype(float).values, m.hour.values, B=B_, seed=0)
    N['AllowShareHourLoTen'] = fmt(100 * lo, 1); N['AllowShareHourHiTen'] = fmt(100 * hi, 1)
    W = cat(f'{paths.RAW_OPS}/window_*.csv').dropna(subset=['G_window'])
    L = pd.read_csv(paths.RAW_SUPP / 'window_lock.csv'); e = L[L.construction == 'efficient_window']
    lk = set(zip(e[e.locked].hour, e[e.locked].w)); keep = np.array([(h, w) not in lk for h, w in zip(W.hour, W.w)])
    d = (W.G_fixed - W.G_single)[keep]
    N['WinNoLockN'] = str(int(keep.sum())); N['WinNoLockDiff'] = fmt(d.mean(), 3); N['WinNoLockMedAbs'] = fmt(d.abs().median(), 3)
    N['WinNoLockMax'] = fmt(d.max(), 2); N['WinNoLockSmall'] = fmt(100 * (d.abs() < 0.05).mean(), 0)
    nb = 0; nn = 0
    for case in ('dc', 'copper'):
        R = {(r['hour'], r['w'], r['cut']): r for f in glob.glob(f'{REV}/network/{case}/requests_*.jsonl') for r in map(json.loads, open(f))}
        for f in glob.glob(f'{REV}/network/{case}/vectors_*.jsonl'):
            for v in map(json.loads, open(f)):
                r = R[(v['hour'], v['w'], v['cut'])]; nn += 1
                nb += int(np.any(np.abs(np.array(v['Ft']) - np.array([r['C_inc'], r['E_inc']])) < np.array(v['scale']) * (1 - 1e-6)))
    N['NetFloorActive'] = str(nb); N['NetFloorN'] = str(nn)
    for nm, f in (('Ten', f'{REV}/scip/cells10_p*.csv'), ('Forty', f'{REV}/scip/forty_cells_p*.csv'), ('Ops', f'{REV}/scip/ops_cells.csv'),
                  ('Net', f'{REV}/network/scip_*.csv')):
        T = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f))]).time_s
        N[f'ScipT{nm}Med'] = fmt(T.median(), 1); N[f'ScipT{nm}Max'] = fmt(T.max(), 0) if T.max() >= 1 else fmt(T.max(), 1)


def onesided():
    """One-sided distance d+ of Proposition 2 on the 10-unit requests (experiments/additional/onesided.py)."""
    fc = f'{REV}/onesided/cells_p0.csv'
    if not glob.glob(fc):
        return
    C = cat(f'{REV}/onesided/cells_p*.csv'); F = cat(f'{REV}/onesided/cf_p*.csv')
    key = ['hour', 'w', 'cut']
    N['OsN'] = str(len(F))
    eq = (C.Gp - C.lb).abs() < 1e-6
    g = C[C.kind == 'grid']
    N['OsEqReach'] = fmt(100 * eq[(C.kind == 'grid') & C.reachable].mean(), 0)
    N['OsEqUnreach'] = fmt(100 * eq[(C.kind == 'grid') & ~C.reachable].mean(), 0)
    ph = C[C.kind == 'physical'].set_index(key).Gp
    r = g[g.R.isin([25, 50, 100])].copy()
    r['a'] = (r.Gp.values - ph.reindex(pd.MultiIndex.from_frame(r[key])).values).clip(min=0)
    m = r.groupby(key + ['reachable']).a.mean().reset_index()
    x = (m.a >= 0.10).astype(float).values
    N['OsAllow'] = fmt(100 * x.mean(), 1)
    lo, hi = boot(x); N['OsAllowLo'] = fmt(100 * lo, 1); N['OsAllowHi'] = fmt(100 * hi, 1)
    lo, hi = boot_hours(x, m.hour.values, B=B_, seed=0); N['OsAllowHourLo'] = fmt(100 * lo, 1); N['OsAllowHourHi'] = fmt(100 * hi, 1)
    N['OsAllowReach'] = fmt(100 * (m[m.reachable].a >= 0.10).mean(), 1)
    N['OsAllowUnreach'] = fmt(100 * (m[~m.reachable].a >= 0.10).mean(), 1)
    pg = g.pivot_table(index='R', columns='rho', values='Gp', aggfunc='mean')
    N['OsSat'] = fmt(max(pg.loc[100, q] - pg.loc[250, q] for q in pg.columns if q <= 0.10), 3)
    N['OsNeed'] = str(int(F['need_0.05'].sum()))
    N['OsRho'] = fmt(F['rho_0.05'].median(), 3); N['OsRStar'] = fmt(F['R_0.05'].median(), 0)
    N['OsNeedOne'] = str(int(F['need_0.01'].sum()))
    N['OsCondEval'] = fmt(100 * F['cond_0.01'].sum() / F['evals_0.01'].sum(), 0)
    K = cat(f'{REV}/compliance/compliance_p0.csv'); K = K[np.isclose(K.eps_E, 0.01)].drop_duplicates(key)
    M = F.merge(K[key + ['rho_star', 'need_E', 'rho_star_E', 'R_star_E']], on=key)
    b = M[M['need_0.05'] & M.rho_star.notna()]; d = b['rho_0.05'] - b.rho_star
    N['OsRhoDiffMed'] = fmt(d.median(), 3); N['OsRhoWithin'] = fmt(100 * (d.abs() <= 0.95 / 256 + 1e-9).mean(), 0)
    N['OsRhoAbove'] = fmt(100 * (d > 0.95 / 256 + 1e-9).mean(), 0); N['OsPairN'] = str(len(b))
    rr = M[M.reachable & M['need_0.01'] & M.need_E.astype(bool)]; dc = rr['rho_0.01'] - rr.rho_star_E
    N['OsCompN'] = str(len(rr)); N['OsCompRhoDiffMed'] = fmt(dc.median(), 3)
    N['OsCompWithin'] = fmt(100 * (dc.abs() <= 0.95 / 256 + 1e-9).mean(), 0)
    N['OsCompNeedReach'] = str(int(M[M.reachable]['need_0.01'].sum())); N['CompNeedReachE'] = str(int(M[M.reachable].need_E.astype(bool).sum()))
    print('onesided', {k: v for k, v in N.items() if k.startswith('Os')})


def forty_region():
    """SCIP on the study-region cells of the 40-unit campaign (experiments/additional/scip_forty_region.py)."""
    X = cat(f'{REV}/scip/forty_region_p*.csv')
    if X is None or not len(X):
        return
    key = ['hour', 'w', 'cut']
    d = X.G_stored - X.G_lb
    N['FrN'] = str(len(X)); N['FrReq'] = str(len(X.groupby(key)))
    N['FrCert'] = str(int((d <= 0.01 + 1e-9).sum())); N['FrCertShare'] = fmt(100 * (d <= 0.01 + 1e-9).mean(), 0)
    N['FrBetter'] = str(int((X.G_ub < X.G_stored - 0.01).sum()))
    N['FrBetterMax'] = fmt((X.G_stored - X.G_ub).max(), 3)
    N['FrGapMed'] = fmt(d.clip(lower=0).median(), 3); N['FrGapMax'] = fmt(d.max(), 3)
    N['FrGapNinety'] = fmt(d.clip(lower=0).quantile(0.9), 3)
    N['FrOpen'] = str(int((d > 0.01 + 1e-9).sum()))
    N['FrTimeMed'] = fmt(X.time_s.median(), 0)
    N['FrOpenGapMed'] = fmt(d[d > 0.01 + 1e-9].median(), 3)
    pgain = X.pivot_table(index='R', columns='rho', values='G_lb', aggfunc='mean'); sgain = X.pivot_table(index='R', columns='rho', values='G_stored', aggfunc='mean')
    N['FrGainStored'] = fmt(sgain.loc[100, 0.10] - sgain.loc[250, 0.10], 3)   # mean gain from 100 to 250 MW at rho = 0.10
    N['FrGainLb'] = fmt(pgain.loc[100, 0.10] - sgain.loc[250, 0.10], 3)       # smallest gain that the SCIP bounds allow
    R = cat(f'{paths.RAW_ATTAIN}/requests_e40*.csv'); R = R.drop_duplicates(key)[key + ['G_phys']]
    M = X.merge(R, on=key)
    lb = (M.G_lb - M.G_phys).clip(lower=0).groupby([M.hour, M.w, M.cut]).mean()
    st = (M.G_stored - M.G_phys).clip(lower=0).groupby([M.hour, M.w, M.cut]).mean()
    N['FrAllowStored'] = fmt(100 * (st >= 0.10).mean(), 1); N['FrAllowLb'] = fmt(100 * (lb >= 0.10).mean(), 1)
    N['FrFlip'] = str(int(((st >= 0.10) != (lb >= 0.10)).sum()))
    pg = M.pivot_table(index='R', columns='rho', values='G_lb', aggfunc='mean'); ps = M.pivot_table(index='R', columns='rho', values='G_stored', aggfunc='mean')
    N['FrSatLb'] = fmt(max(pg.loc[250, q] - pg.loc[500, q] for q in pg.columns if q <= 0.10), 3)
    N['FrSatUbMax'] = fmt(max(ps.loc[250, q] - pg.loc[500, q] for q in pg.columns if q <= 0.10), 3)
    print('forty region', {k: v for k, v in N.items() if k.startswith('Fr')})


def decomposition():
    """Split the family-best result of each family into physical, allowance and search parts (Eq. shortfall).

    Audit of the default configuration (methods_extended, ten seeds averaged), by stratum, and the tight configuration
    (50 MW, rho = 0.05) of the nine configurations, which holds reachable requests only (physical part zero).
    The shares are of the summed attainment, so they add to 100 within rounding.
    """
    fam = {'CF': ('GS', 'FACE', 'NICE_R', 'DAACE_R', 'DiCE'), 'Rep.': ('POLY', 'ELLP', 'CGM'), 'DFO': ('COBYQA', 'SLSQP_FD', 'CMAES')}
    F = sorted(glob.glob(f'{paths.RAW}/additional/methods_extended/raw_p*.csv'))
    if not F:
        return
    A = pd.concat([pd.read_csv(f) for f in F], ignore_index=True)
    keys = A[['hour', 'w', 'cut', 'inst_hash']].drop_duplicates().sort_values(['hour', 'w', 'cut'])
    keys = keys[~keys.inst_hash.duplicated()]
    A = A.merge(keys[['hour', 'w', 'cut']], on=['hour', 'w', 'cut'])
    rows = []
    def parts(b, lab, sub, nm):
        g = b.g.sum(); ph = b.ph.clip(lower=0).sum(); al = (b.ref - b.ph).clip(lower=0).sum(); se = (b.g - b.ref).clip(lower=0).sum()
        tot = ph + al + se
        rows.append((lab, sub, nm, len(b), b.g.mean(), 100 * ph / tot, 100 * al / tot, 100 * se / tot))
    for rch, lab in ((True, 'Audit, reachable'), (False, 'Audit, unreachable')):
        for nm, ms in fam.items():
            x = A[A.method.isin(ms)]
            seeds = sorted(set(x.mseed) - {0}) or [0]
            bb = []
            for sd in seeds:
                y = x[x.mseed.isin([0, sd])].groupby(['hour', 'w', 'cut']).agg(g=('G_fallback', 'min'), ref=('G_ref', 'first'),
                                                                              ph=('G_phys', 'first'), r=('reachable', 'first'))
                bb.append(y)
            b = pd.concat(bb).groupby(level=[0, 1, 2]).mean()
            b = b[b.r.astype(bool) == rch]
            parts(b, lab, 'audit', nm)
    sys.path.insert(0, os.path.dirname(__file__))
    import methods as MM
    _, _, allm, _ = MM.load()
    t = allm[allm.config == 'allow50_005']
    for nm, ms in fam.items():
        b = t[t.method.isin(ms)].groupby(['hour', 'w', 'cut']).agg(g=('G_fallback', 'min'), ref=('G_ref', 'first'))
        b['ph'] = 0.0
        parts(b, 'Tight, reachable', 'tight', nm)
    D = pd.DataFrame(rows, columns=['set', 'sub', 'family', 'n', 'Gm', 'phys', 'allow', 'search'])
    D.to_csv(f'{PROC}/decomposition.csv', index=False)
    tag = {'Audit, reachable': 'DecReach', 'Audit, unreachable': 'DecUnreach', 'Tight, reachable': 'DecTight'}
    fn = {'CF': 'Gen', 'Rep.': 'Proc', 'DFO': 'Dfo'}
    for r in D.itertuples():
        P = tag[r.set] + fn[r.family]
        N[P + 'Phys'] = fmt(r.phys, 0); N[P + 'Allow'] = fmt(r.allow, 0); N[P + 'Search'] = fmt(r.search, 0); N[P + 'G'] = fmt(r.Gm, 2)
        N[tag[r.set] + 'N'] = str(r.n)
    with open(f'{TAB}/tab_decomp.tex', 'w') as f:
        f.write('\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n')
        f.write('Instances & Family & $n$ & Mean $G_m$ & Physical & Allowance & Search \\\\\n\\midrule\n')
        last = None
        for r in D.itertuples():
            if last is not None and r.set != last:
                f.write('\\midrule\n')
            f.write(f"{r.set if r.set != last else ''} & {r.family} & {r.n} & {r.Gm:.2f} & {r.phys:.0f} & {r.allow:.0f} & {r.search:.0f} \\\\\n")
            last = r.set
        f.write('\\bottomrule\n\\end{tabular}\n')


def closed_form_phys():
    """G*_phys of unreachable requests against its closed form (E_min - E_tar) / (s_E d(P_inc))."""
    from eeldattain.model import System
    S = System(str(paths.COEFFS['10']))
    R = pd.concat([pd.read_csv(f) for f in glob.glob(f'{paths.RAW_ATTAIN}/requests_e10*.csv')])
    R = R[R.duplicate_of_w.isna() & ~R.reachable.astype(bool)]
    V = {(v['hour'], v['w'], v['cut']): v for f in glob.glob(f'{paths.RAW_ATTAIN}/vectors_e10*.jsonl') for v in map(json.loads, open(f))}
    d = []
    for r in R.itertuples():
        v = V[(r.hour, r.w, r.cut)]; Pinc, Ft, sc = (np.array(v[k]) for k in ('Pinc', 'Ft', 'scale'))
        e0 = np.linalg.norm((S.F(Pinc) - Ft) / sc)
        d.append(abs(r.G_phys - (r.E_min - Ft[1]) / (sc[1] * e0)))
    m, e = f'{max(d):.0e}'.split('e'); N['ClosedPhysN'] = str(len(d)); N['ClosedPhysMax'] = f'\\ensuremath{{{m}\\times10^{{{int(e)}}}}}'


FAM = {'proc': ['POLY', 'ELLP', 'CGM'], 'gen': ['GS', 'FACE', 'NICE_R', 'DAACE_R', 'DiCE'], 'dfo': ['COBYQA', 'SLSQP_FD', 'CMAES']}


def sci(x):
    m, e = f'{x:.1e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


def signflip_p(x):
    """Exact one-sided sign-flip test of mean(x) > 0 over all 2^n sign patterns (n hour means)."""
    x = np.asarray(x, float); n = len(x); obs = x.mean(); ge = 0; tot = 0
    bits = np.arange(n)
    for start in range(0, 2 ** n, 2 ** 16):
        idx = np.arange(start, min(start + 2 ** 16, 2 ** n))
        sg = 1 - 2 * ((idx[:, None] >> bits) & 1)
        ge += int((((sg * x).mean(axis=1)) >= obs - 1e-12).sum()); tot += len(idx)
    return ge / tot


def methods_extended():
    """Default configuration over all hours, reachable and unreachable requests, ten seeds (methods_extended.py)."""
    rows = ext_stats('methods_extended', 'Ext', table=True)
    cvx = ext_stats('methods_extended_convex', 'Cvx', table=True)  # the same audit without the valve-point term
    if not rows:
        return
    # Table of the primary comparison, by family and stratum, with the convex variant as a second panel
    L = ['\\begin{tabular}{@{}lrrrrrrr@{}}', '\\toprule',
         'Family & All & Seeds & Reach. & Unreach. & Hour & Med.\\ $\\Delta$ & Hours \\\\', '\\midrule',
         f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{Valve-point cost, {N['ExtN']} instances}}}} \\\\"] + rows
    if cvx:
        L += ['\\midrule', f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{Convex cost, {N['CvxN']} instances}}}} \\\\"] + cvx
    L += ['\\bottomrule', '\\end{tabular}']
    open(f'{TAB}/tab_audit.tex', 'w').write('\n'.join(L) + '\n')


def ext_stats(folder, P, table):
    F = sorted(glob.glob(f'{REV}/{folder}/raw_p*.csv'))
    if not F:
        return []
    D = pd.concat([pd.read_csv(f) for f in F], ignore_index=True)
    N[f'{P}Runs'] = f'{len(D):,}'.replace(',', '{,}'); N[f'{P}Beat'] = str(int(((D.G_fallback - D.G_ref) < -0.01).sum()))
    first = D.drop_duplicates(['hour', 'w', 'cut']).sort_values(['hour', 'w', 'cut']).drop_duplicates('inst_hash')
    D = D.merge(first[['hour', 'w', 'cut']], on=['hour', 'w', 'cut'])
    seeds = sorted(s for s in D.mseed.unique() if s != 0)
    inst = first.set_index(['hour', 'w', 'cut'])
    N[f'{P}N'] = str(len(first)); N[f'{P}H'] = str(first.hour.nunique()); N[f'{P}Seeds'] = str(len(seeds))
    N[f'{P}NReach'] = str(int(first.reachable.astype(bool).sum())); N[f'{P}NUnreach'] = str(int((~first.reachable.astype(bool)).sum()))
    best = {}
    for fam, ms in FAM.items():
        cols = []
        for s in seeds:
            X = D[D.method.isin(ms) & D.mseed.isin([s, 0])]
            cols.append(X.groupby(['hour', 'w', 'cut']).G_fallback.min().rename(s))
        best[fam] = pd.concat(cols, axis=1).sub(inst.G_ref, axis=0)          # shortfall per instance and seed
    reach = inst.reachable.astype(bool)
    for fam, nm in (('gen', 'Gen'), ('proc', 'Proc'), ('dfo', 'Dfo')):
        S = best[fam]; w5 = (S <= EPS).mean(axis=0) * 100              # share within 0.05, one value per seed
        N[f'{P}{nm}Within'] = fmt(w5.mean(), 0); N[f'{P}{nm}WithinMin'] = fmt(w5.min(), 0); N[f'{P}{nm}WithinMax'] = fmt(w5.max(), 0)
        for lab, sel in (('Reach', reach), ('Unreach', ~reach)):
            N[f'{P}{nm}Within{lab}'] = fmt(100 * (S[sel.values] <= EPS).mean(axis=0).mean(), 0)
        hm = S.mean(axis=1).groupby(level='hour').mean()                     # hour means over instances and seeds
        v = S.mean(axis=1).median(); N[f'{P}{nm}ShortMed'] = fmt(0.0 if abs(v) < 0.0005 else v, 3)
        N[f'{P}{nm}HoursAbove'] = str(int((hm > EPS).sum()))
        hs = (S <= EPS).mean(axis=1).groupby(level='hour').mean() * 100      # share within 0.05 per hour, seeds averaged
        N[f'{P}{nm}HourShareMed'] = fmt(hs.median(), 0); N[f'{P}{nm}HourShareMax'] = fmt(hs.max(), 0)
        N[f'{P}{nm}HourShareMin'] = fmt(hs.min(), 0)
        pv = signflip_p(hm.values - EPS)
        N[f'{P}{nm}P'] = sci(pv) if pv < 0.001 else fmt(pv, 3)
        N[f'{P}{nm}SeedSpread'] = fmt((S.max(axis=1) - S.min(axis=1)).median(), 3)
    N[f'{P}Pmin'] = sci(0.5 ** first.hour.nunique())
    if not table:
        return []
    lab = {'gen': 'Generators', 'proc': 'Procedures', 'dfo': 'DFO'}
    rng = {nm: ('--' if nm == 'Proc' else f"{N[f'{P}{nm}WithinMin']}--{N[f'{P}{nm}WithinMax']}") for nm in ('Gen', 'Proc', 'Dfo')}  # procedures are deterministic
    rows = [f"{lab[fam]} & {N[f'{P}{nm}Within']} & {rng[nm]} & "
            f"{N[f'{P}{nm}WithinReach']} & {N[f'{P}{nm}WithinUnreach']} & {N[f'{P}{nm}HourShareMed']} & {N[f'{P}{nm}ShortMed']} & "
            f"{N[f'{P}{nm}HoursAbove']} of {N[f'{P}H']} \\\\"
            for fam, nm in (('gen', 'Gen'), ('proc', 'Proc'), ('dfo', 'Dfo'))]
    d = (best['dfo'] - best['proc']).mean(axis=1).groupby(level='hour').mean()
    N[f'{P}DfoProcDiff'] = fmt(d.mean(), 3); N[f'{P}DfoProcHours'] = str(int((d < 0).sum()))
    d = (best['proc'] - best['gen']).mean(axis=1).groupby(level='hour').mean()
    N[f'{P}ProcGenDiff'] = fmt(d.mean(), 3); N[f'{P}ProcGenHours'] = str(int((d < 0).sum()))
    return rows


def scip_direct():
    """SCIP on the direct formulation (22) for the counterfactuals whose minimality the bisection bounds leave open."""
    f = f'{REV}/scip/direct.csv'
    if not os.path.exists(f):
        return
    D = pd.read_csv(f)
    ok = D.status.isin(['optimal', 'gaplimit'])
    N['DirScipN'] = str(len(D)); N['DirScipSolved'] = str(int(ok.sum()))
    N['DirScipMinimal'] = str(int(D.minimal.astype(bool).sum()))
    sm = ok & (D.U < D.below - 1e-9)
    N['DirScipSmaller'] = str(int(sm.sum()))
    N['DirScipSmallerRhoMax'] = fmt((D.value - D.U)[sm & (D.cf == 'rho')].max() if (sm & (D.cf == 'rho')).any() else 0, 3)
    N['DirScipSmallerRMax'] = fmt((D.value - D.U)[sm & (D.cf == 'R')].max() if (sm & (D.cf == 'R')).any() else 0, 1)
    N['DirScipOpen'] = str(int((~D.minimal.astype(bool) & ~sm).sum()))
    N['DirScipTimeMed'] = fmt(D.time_s.median(), 0)
    N['DirScipAdopted'] = str(int(sm.sum()))
    j = json.load(open(f'{PROC}/numbers_additional.json')) if os.path.exists(f'{PROC}/numbers_additional.json') else {}
    if 'CfbMinimalAll' in j:
        N['CfbMinimalFinal'] = str(int(j['CfbMinimalAll']) + int(D.minimal.astype(bool).sum()) + int(sm.sum()))


def ramp_grid():
    """SCIP on every cell of the ramp-band grid of the loss model."""
    f = f'{REV}/scip/ops_ramp_grid.csv'
    if not os.path.exists(f):
        return
    D = pd.read_csv(f)
    N['RampGridN'] = f'{len(D):,}'.replace(',', '{,}')
    N['RampGridCert'] = f'{int(((D.G_stored - D.G_lb) <= 0.01).sum()):,}'.replace(',', '{,}')
    N['RampGridBetter'] = str(int(((D.G_stored - D.G_ub) > 0.01).sum()))
    N['RampGridBetterMax'] = fmt(max((D.G_stored - D.G_ub).max(), 0), 3)
    N['RampGridOpen'] = str(int(((D.G_stored - D.G_lb) > 0.01).sum()))


if __name__ == '__main__':
    main()
