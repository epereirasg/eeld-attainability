"""Ablation of the allowance grid: marginal effect of the budget R and of the band rho (Table 7).

For every distinct request, the marginal effect of one instrument is the change in G* when that instrument
moves one grid step up while the other is held at a fixed level. The script reports, per step and system,
the mean change with a 95% bootstrap interval over requests, the two-sided Wilcoxon signed-rank p-value,
the rank-biserial effect size and the share of requests that gain more than 0.01. Tables only.
Outputs: results/processed/ablation_e10.csv, ablation_e40.csv, results/tables/tab_ablation.tex,
results/processed/numbers_ablation.json and results/macros/numbers_ablation.tex
"""
import glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from eeldattain import paths

N = {}


def boot_ci(x, B=10000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    s = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return np.quantile(s, [0.025, 0.975])


def rbc(d):
    d = d[d != 0]
    if not len(d): return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def load(case):
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{paths.RAW_ATTAIN}/requests_e{case}*.csv'))])
    G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{paths.RAW_ATTAIN}/grid_e{case}*.csv'))])
    u = R[R.duplicate_of_w.isna()][['hour', 'w', 'cut']]
    return G[G.kind == 'grid'].merge(u, on=['hour', 'w', 'cut'])


def steps(G, case):
    P = G.pivot_table(index=['hour', 'w', 'cut'], columns=['R', 'rho'], values='G')
    Rs = sorted(G.R.unique()); rhos = sorted(G.rho.unique())
    rows = []
    for rho in rhos:                                     # budget steps at a fixed band
        for r0, r1 in zip(Rs[:-1], Rs[1:]):
            d = (P[(r0, rho)] - P[(r1, rho)]).values     # gain = reduction of G*
            rows.append(dict(case=case, instrument='R', step=f'{r0:g}->{r1:g} MW', fixed=f'rho={rho:.2f}', lo=r0, hi=r1, held=rho, d=d))
    for R in Rs:                                         # band steps at a fixed budget
        for p0, p1 in zip(rhos[:-1], rhos[1:]):
            d = (P[(R, p0)] - P[(R, p1)]).values
            rows.append(dict(case=case, instrument='rho', step=f'{p0:.2f}->{p1:.2f}', fixed=f'R={R:g} MW', lo=p0, hi=p1, held=R, d=d))
    out = []
    for r in rows:
        d = r.pop('d'); lo, hi = boot_ci(d)
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        out.append(dict(r, n=len(d), mean_gain=d.mean(), ci_lo=lo, ci_hi=hi, median_gain=np.median(d),
                        p=p, rbc=rbc(d), share_gain_gt_001=float((d > 0.01).mean())))
    return pd.DataFrame(out)


def main():
    tabs = {}
    for case, T in (('10', 'Ten'), ('40', 'Forty')):
        A = steps(load(case), case); A.to_csv(f'{paths.PROCESSED}/ablation_e{case}.csv', index=False); tabs[case] = A
        Rs = A[A.instrument == 'R']; Ps = A[A.instrument == 'rho']
        last_R = Rs[Rs.hi == Rs.hi.max()]; last_rho = Ps[Ps.hi == Ps.hi.max()]
        N[f'AblLastRGain{T}'] = f'{last_R.mean_gain.mean():.3f}'           # last budget step, averaged over bands
        N[f'AblLastRhoGain{T}'] = f'{last_rho.mean_gain.mean():.3f}'       # last band step, averaged over budgets
        N[f'AblLastRStep{T}'] = last_R.step.iloc[0]; N[f'AblLastRhoStep{T}'] = last_rho.step.iloc[0]
        top_rho = Ps[(Ps.hi == Ps.hi.max()) & (Ps.held == Ps.held.max())].iloc[0]
        N[f'AblLastRhoGainTopR{T}'] = f'{top_rho.mean_gain:.3f}'; N[f'AblLastRhoShareTopR{T}'] = f'{100 * top_rho.share_gain_gt_001:.0f}'
        top_R = Rs[(Rs.hi == Rs.hi.max()) & (Rs.held == Rs.held.max())].iloc[0]
        N[f'AblLastRGainTopRho{T}'] = f'{top_R.mean_gain:.3f}'; N[f'AblLastRShareTopRho{T}'] = f'{100 * top_R.share_gain_gt_001:.0f}'
    def pf(p):
        return '$<$0.001' if p < 0.001 else f'{p:.3f}'
    with open(f'{paths.TABLES}/tab_ablation.tex', 'w') as f:
        # Proposition 1(ii) makes every paired step difference nonnegative for exact minima, so a signed-rank
        # test of these steps is uninformative. The table reports the mean gain, its interval and the share
        # above 0.01. The p-values and rank-biserial correlations stay in processed/ablation_e*.csv.
        z = lambda x: f'{(0.0 if abs(x) < 0.0005 else x):.3f}'      # no negative zero in print
        f.write('\\begin{tabular}{@{}llllrr@{}}\n\\toprule\n')
        f.write('System & Instrument & Step & Held at & Mean gain [95\\% CI] & Share $>0.01$ \\\\\n\\midrule\n')
        for case, label in (('10', '10-unit'), ('40', '40-unit')):
            A = tabs[case]; first = True
            sel = pd.concat([A[(A.instrument == 'R') & (A.held.isin([A[A.instrument == 'R'].held.min(), A[A.instrument == 'R'].held.max()]))],
                             A[(A.instrument == 'rho') & (A.held.isin([A[A.instrument == 'rho'].held.min(), A[A.instrument == 'rho'].held.max()]))]])
            for r in sel.itertuples():
                inst = '$R$' if r.instrument == 'R' else '$\\rho$'
                step = r.step.replace('->', '$\\to$')
                held = r.fixed.replace('rho=', '$\\rho=$').replace('R=', '$R=$')
                f.write(f"{label if first else ''} & {inst} & {step} & {held} & {z(r.mean_gain)} [{z(r.ci_lo)}, {z(r.ci_hi)}] & {100 * r.share_gain_gt_001:.0f}\\% \\\\\n")
                first = False
            if case == '10':
                f.write('\\midrule\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    with open(f'{paths.MACROS}/numbers_ablation.tex', 'w') as f:
        for k, v in sorted(N.items()):
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(N, open(f'{paths.PROCESSED}/numbers_ablation.json', 'w'), indent=1)
    for k, v in sorted(N.items()):
        print(f'{k:28s} {v}')


if __name__ == '__main__':
    main()
