"""Number macros and table of the 30-bus network case.

Reads results/raw/additional/network/ and writes results/macros/numbers_network.tex,
results/processed/numbers_network.json and results/tables/tab_network.tex.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROC, MAC, TAB, boot_ci, paths

NET = paths.RAW / 'additional' / 'network'
N = {}
EPS = 0.05
REAL = [(R, rho) for R in (25, 50, 100) for rho in (0.05, 0.10, 0.20)]
MULT = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2}


def cnt(n):
    return f'{int(n):,}'.replace(',', '{,}')


def fmt(x, d=2):
    return 'nan' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def load(net):
    d = NET / net; tag = f'n30{net}'
    R = pd.DataFrame([json.loads(l) for l in open(d / f'requests_{tag}.jsonl')])
    G = pd.read_csv(d / f'grid_{tag}.csv')
    R['unique'] = R.duplicate_of_w.isna()
    G = G.merge(R[['hour', 'w', 'cut', 'unique']], on=['hour', 'w', 'cut'])
    A = {h: json.load(open(d / f'anchors_e30_h{h}.json')) for h in MULT if (d / f'anchors_e30_h{h}.json').exists()}
    return R, G, A


def summary(net, T):
    R, G, _ = load(net)
    Ru = R[R.unique].copy(); Gu = G[G.unique & (G.kind == 'grid')].copy()
    N[f'NetN{T}'] = str(len(Ru))
    hw = Ru.groupby('hour').front_width.first()
    N[f'NetWMin{T}'] = fmt(100 * hw.min(), 1); N[f'NetWMax{T}'] = fmt(100 * hw.max(), 1)
    N[f'NetReach{T}'] = fmt(100 * Ru.reachable.astype(bool).mean(), 0)
    N[f'NetEta{T}'] = fmt(Ru.eta_E.mean())
    N[f'NetGphys{T}'] = fmt(Ru.G_phys.mean())
    Gu['ga'] = (Gu.G - Gu.G_phys).clip(lower=0)
    reg = Gu[[(R_, r_) in REAL for R_, r_ in zip(Gu.R, Gu.rho)]]
    per = reg.groupby(['hour', 'w', 'cut']).ga.mean(); ind = (per >= 0.10).astype(float)
    lo, hi = boot_ci(ind.values)
    N[f'NetAllowShare{T}'] = fmt(100 * ind.mean(), 0); N[f'NetAllowShareLo{T}'] = fmt(100 * lo, 0); N[f'NetAllowShareHi{T}'] = fmt(100 * hi, 0)
    N[f'NetAllowMean{T}'] = fmt(per.mean())
    pg = Gu.pivot_table(index='R', columns='rho', values='G', aggfunc='mean')
    N[f'NetSat{T}'] = fmt(max(pg.loc[100, r] - pg.loc[250, r] for r in pg.columns if r <= 0.10 + 1e-9), 3)
    need = Ru[Ru.G_ref > Ru.G_phys + EPS]; tot = need.G_ref - need.G_phys
    fR = (need.G_ref - need.G_Rfree) / tot; fr = (need.G_ref - need.G_rhofree) / tot
    N[f'NetNeed{T}'] = str(len(need))
    N[f'NetReliefR{T}'] = fmt(100 * fR.mean(), 0); N[f'NetReliefRho{T}'] = fmt(100 * fr.mean(), 0)
    N[f'NetJoint{T}'] = fmt(100 * ((fR < 0.7) & (fr < 0.7)).mean(), 0)
    N[f'NetRhoMed{T}'] = fmt(need.rho_star.median(), 3); N[f'NetRStarMed{T}'] = fmt(need.R_star.median(), 0)
    sets = need.dropna(subset=['S_size']); sz = sets.S_size.astype(int)
    N[f'NetNeedSet{T}'] = str(len(sets)); N[f'NetSLeTwo{T}'] = fmt(100 * (sz <= 2).mean(), 0) if len(sets) else 'n/a'
    units = pd.Series([u for s in sets.S_units.fillna('').astype(str) for u in s.split('-') if u]).value_counts()
    if len(units):
        N[f'NetTopUnit{T}'] = units.index[0]; N[f'NetTopUnitShare{T}'] = fmt(100 * units.iloc[0] / len(sets), 0)
    # RQ2 rule, all three parts
    okA = ind.mean() >= 0.5 and lo > 0.4; okB = float(N[f'NetSat{T}']) <= 0.02; okC = float(N[f'NetJoint{T}']) > 50
    N[f'NetRqTwo{T}'] = 'supported' if okA and okB and okC else 'not supported'
    return Ru


def anchors_and_lines():
    from eeldattain.network import NetSystem
    from eeldattain.runio import load_cache
    rows = []
    for h, m in MULT.items():
        c = {}
        for net in ('copper', 'dc'):
            f = NET / net / f'anchors_e30_h{h}.json'
            if f.exists():
                A, _, INC = load_cache(f); c[net] = (A, INC)
        if len(c) < 2:
            continue
        s = NetSystem(m, limits=True)
        Ac, Ad = c['copper'][0], c['dc'][0]
        rows.append(dict(h=h, mult=m, PD=s.net.PD, Cmin_cu=Ac['ideal'][0], Cmin_dc=Ad['ideal'][0],
                         Emin_cu=Ac['ideal'][1], Emin_dc=Ad['ideal'][1],
                         bind_Pc_dc='|'.join(s.net.label(i) for i in s.net.binding(Ad['Pc'])),
                         over_Pc_cu='|'.join(s.net.label(i) for i in np.where(np.abs(s.net.flows(Ac['Pc'])) > s.net.rate + 1e-6)[0]),
                         bind_Pe_dc='|'.join(s.net.label(i) for i in s.net.binding(Ad['Pe'])),
                         inc_bind=sum(bool(s.net.binding(P)) for P in c['dc'][1].values()),
                         W_cu=(Ac['nadir'][1] - Ac['ideal'][1]) / Ac['nadir'][1], W_dc=(Ad['nadir'][1] - Ad['ideal'][1]) / Ad['nadir'][1]))
    X = pd.DataFrame(rows)
    X.to_csv(f'{PROC}/network_anchors.csv', index=False)
    cong = 100 * (X.Cmin_dc / X.Cmin_cu - 1)
    N['NetCongCostMin'] = fmt(cong.min(), 1); N['NetCongCostMax'] = fmt(cong.max(), 1)
    N['NetEminSame'] = 'yes' if np.allclose(X.Emin_cu, X.Emin_dc, rtol=1e-6) else 'no'
    lines = sorted({l for s in X.bind_Pc_dc for l in s.split('|') if l})
    N['NetBindLinesPc'] = ', '.join(lines)
    N['NetBindPeAny'] = str(int((X.bind_Pe_dc != '').sum()))
    N['NetLevels'] = str(len(X))
    ab = json.load(open(NET / 'abido_check.json'))
    N['NetAbidoCmin'] = fmt(ab['Cmin'], 2); N['NetAbidoMaxDiff'] = fmt(max(ab['maxdiff_Pc_pu'], ab['maxdiff_Pe_pu']), 4)
    N['NetAbidoEoffset'] = fmt(ab['E_Pe'] - 0.1942, 4)


def split():
    f = NET / 'split_dc.csv'
    if not f.exists():
        return
    S = pd.read_csv(f).fillna('')
    dphys = S.G_phys - S.G_phys_nolim; dref = S.G_ref - S.G_ref_nolim
    N['NetSplitN'] = str(len(S))
    N['NetNetPhysMean'] = fmt(dphys.mean()); N['NetNetPhysMax'] = fmt(dphys.max())
    N['NetNetPhysShare'] = fmt(100 * (dphys > 0.01).mean(), 0)
    N['NetNetRefMean'] = fmt(dref.mean()); N['NetNetRefShare'] = fmt(100 * (dref > 0.01).mean(), 0)
    N['NetBindIncShare'] = fmt(100 * (S.bind_inc != '').mean(), 0)
    N['NetBindRefShare'] = fmt(100 * (S.bind_ref != '').mean(), 0)
    N['NetBindPhysShare'] = fmt(100 * (S.bind_phys != '').mean(), 0)
    lines = pd.Series([l for c in ('bind_inc', 'bind_ref', 'bind_phys') for s in S[c] for l in str(s).split('|') if l]).value_counts()
    N['NetBindLinesAll'] = ', '.join(lines.index)
    N['NetSplitRawWorse'] = str(int(((S.G_phys_nolim_raw > S.G_phys + 1e-4) | (S.G_ref_nolim_raw > S.G_ref + 1e-4)).sum()))


def scip():
    rows = []
    for net, T in (('dc', 'Dc'), ('copper', 'Cu')):
        f = NET / f'scip_{net}.csv'
        if not f.exists():
            continue
        D = pd.read_csv(f)
        cert = D.G_stored - D.G_lb <= 0.01
        N[f'NetScipCells{T}'] = cnt(len(D)); N[f'NetScipCert{T}'] = cnt(cert.sum())
        N[f'NetScipBetter{T}'] = str(int((D.G_stored - D.G_ub > 0.01).sum()))
        N[f'NetScipMaxImp{T}'] = fmt(max(0.0, (D.G_stored - D.G_ub).max()), 4)
        N[f'NetScipMaxTime{T}'] = fmt(D.time_s.max(), 1)
        rows.append(D)
    if rows:
        D = pd.concat(rows)
        N['NetScipCells'] = cnt(len(D)); N['NetScipCert'] = cnt((D.G_stored - D.G_lb <= 0.01).sum())
        N['NetScipMaxImp'] = fmt(max(0.0, (D.G_stored - D.G_ub).max()), 4)


def table():
    def r(lbl, k, suf=''):
        return f"{lbl} & {N.get(k + 'Cu', 'n/a')}{suf} & {N.get(k + 'Dc', 'n/a')}{suf} \\\\\n"
    t = '\\begin{tabular}{@{}p{4.6cm}cc@{}}\n\\toprule\nQuantity & Copper plate & DC line limits \\\\\n\\midrule\n'
    t += f"Front width $W_h$ (\\%) & {N['NetWMinCu']}--{N['NetWMaxCu']} & {N['NetWMinDc']}--{N['NetWMaxDc']} \\\\\n"
    t += r('Reachable requests (\\%)', 'NetReach')
    t += r('Mean $\\eta_E$', 'NetEta')
    t += r('Mean $G^{*}_{\\mathrm{phys}}$', 'NetGphys')
    t += f"Requests with mean $G^{{*}}_{{\\mathrm{{allow}}}}\\ge0.10$ (\\%) & {N['NetAllowShareCu']} [{N['NetAllowShareLoCu']}, {N['NetAllowShareHiCu']}] & {N['NetAllowShareDc']} [{N['NetAllowShareLoDc']}, {N['NetAllowShareHiDc']}] \\\\\n"
    t += r('Decrease of mean $G^{*}$, 100 to \\SI{250}{MW}', 'NetSat')
    t += r('Neither relaxation removes 70\\% (\\%)', 'NetJoint')
    t += r('Median $\\rho^{*}$', 'NetRhoMed')
    t += r('Median $R^{*}$ (MW)', 'NetRStarMed')
    t += r('$|S^{*}|\\le2$ (\\%)', 'NetSLeTwo')
    t += '\\bottomrule\n\\end{tabular}\n'
    open(f'{TAB}/tab_network.tex', 'w').write(t)


def main():
    for net, T in (('copper', 'Cu'), ('dc', 'Dc')):
        if (NET / net / f'requests_n30{net}.jsonl').exists():
            summary(net, T)
    anchors_and_lines(); split(); scip()
    if 'NetNCu' in N and 'NetNDc' in N:
        table()
    with open(f'{MAC}/numbers_network.tex', 'w') as f:
        for k, v in sorted(N.items()):
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(N, open(f'{PROC}/numbers_network.json', 'w'), indent=1)
    print(json.dumps(N, indent=0))


if __name__ == '__main__':
    main()
