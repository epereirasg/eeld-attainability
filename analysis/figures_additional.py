"""Figs. 5 and 6, drawn from the stored campaign and the additional experiments.

Fig. 5 (fig_attainability_map.pdf): attainability maps over the dense allowance grid of the 10-unit system,
  (a) mean allowance gap, (b) median bounded emission reach, (c) joint counterfactual (share attained).
Fig. 6 (fig_frontier_counterfactuals.pdf): (a) budget-only frontier, (b) band against budget
  counterfactual per request, (c) exact unit counterfactual by tolerance (exhaustive search over all units).

Inputs: results/processed/additional_*.csv (written by analysis/additional.py) and results/raw/attainability.
Style follows figures_seaborn.py (one sequential "Blues" scale, panel titles top left, 9 pt type).
Run:  PYTHONPATH=src python3 analysis/figures_additional.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import figures_seaborn as FS                      # style, palette and helpers
from figures_seaborn import plt, sns, title, logx, save, B2, B3, B4, B5, B6, B7, B8, B9, INK, PT, DCOL, CUTS, CUT_LAB, CUT_COL, CUT_MK
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.ticker as mt
from common import PROC, load_case

RLAB = ['10', '25', '50', '100', '250', r'$\infty$']
RVAL = [10, 25, 50, 100, 250, np.inf]
PLAB = ['0.02', '0.05', '0.10', '0.20', '0.50', '1']
PVAL = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
CMAP = LinearSegmentedColormap.from_list('bl', sns.color_palette('Blues', 256))
EPS = 0.05


def matrix(df, col, agg):
    M = np.full((len(PVAL), len(RVAL)), np.nan)
    for i, p in enumerate(PVAL):
        for j, R in enumerate(RVAL):
            v = df[np.isclose(df.rho, p) & ((df.R == R) if np.isfinite(R) else np.isinf(df.R))][col]
            if len(v):
                M[i, j] = agg(v)
    return M


XPOS = {10: 10, 25: 25, 50: 50, 100: 100, 250: 250, np.inf: 700}     # unrestricted budget drawn at the right edge
RHO_SHOW = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
RHO_C = dict(zip(RHO_SHOW, [B3, B4, B5, B7, B8, B9]))
RHO_M = dict(zip(RHO_SHOW, ['^', 's', 'D', 'o', 'v', 'P']))


def budget_axis(a, label_y=0.985, va='top'):
    a.set_xscale('log'); a.minorticks_off()
    a.set_xticks(list(XPOS.values())); a.set_xticklabels(['10', '25', '50', '100', '250', r'$\infty$'])
    a.axvspan(25, 100, color=B2, alpha=0.55, lw=0, zorder=0)
    a.text(50, label_y, 'study region', transform=a.get_xaxis_transform(), ha='center', va=va, fontsize=PT - 2, color=B8)
    a.set_xlabel(r'Budget $R$ (MW)')


def end_labels(a, ys, xpos=820):
    """Direct labels at the right end of the lines, nudged apart so that they do not overlap."""
    order = sorted(ys, key=lambda k: ys[k]); placed = []
    for k in order:
        y = ys[k]
        if placed and y < placed[-1] + 0.055:
            y = placed[-1] + 0.055
        placed.append(y)
        a.text(xpos, y, rf'$\rho={k:g}$', fontsize=PT - 2, color=RHO_C[k], va='center', ha='left', clip_on=False)


def fig5():
    G = pd.read_csv(f'{PROC}/additional_dense_grid.csv')
    G['R'] = G.R.astype(float); G['x'] = G.R.map(lambda r: XPOS[np.inf] if not np.isfinite(r) else XPOS[int(r)])
    R, _ = load_case('10'); Ru = R[R.unique]
    need = Ru[Ru.G_ref > Ru.G_phys + EPS]
    key = ['hour', 'w', 'cut']
    Gn = G.merge(need[key], on=key); Gn['ok'] = (Gn.G_best <= Gn.G_phys_campaign + EPS).astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(DCOL, 2.7), gridspec_kw={'width_ratios': [1.0, 1.12, 1.0]})
    # (a) mean allowance gap against the budget, one line per band, 95% bootstrap ribbons
    a = ax[0]; ends = {}
    for r in RHO_SHOW:
        sub = G[np.isclose(G.rho, r)]
        sns.lineplot(data=sub, x='x', y='G_allow', ax=a, color=RHO_C[r], marker=RHO_M[r], markersize=4.2,
                     markeredgecolor='white', markeredgewidth=0.5, errorbar=('ci', 95), n_boot=1000, seed=0,
                     err_kws={'alpha': 0.15, 'lw': 0}, lw=1.4, label=f'{r:g}')
        ends[r] = sub[sub.x == XPOS[np.inf]].G_allow.mean()
    a.axhline(0.10, color=INK, ls=':', lw=0.9)
    budget_axis(a, 0.015, 'bottom'); a.set_ylim(-0.01, 0.66); a.set_ylabel(r'Mean $G^{*}_{\mathrm{allow}}$')
    a.get_legend().remove()
    h = [Line2D([0], [0], color=RHO_C[r], marker=RHO_M[r], ms=4, mec='white', lw=1.4) for r in RHO_SHOW]
    a.legend(h, [f'{r:g}' for r in RHO_SHOW], title=r'Band $\rho$', loc='upper right', fontsize=PT - 2.5,
             title_fontsize=PT - 2.5, ncol=3, handlelength=1.3, columnspacing=0.6, borderpad=0.3, handletextpad=0.3)
    title(a, 'a', 'Allowance gap')
    # (b) bounded emission reach: distribution over requests for a ladder of allowances
    a = ax[1]
    ladder = [(25, 0.05, r'25, 0.05'), (50, 0.05, r'50, 0.05 ($R_0,\rho_0$)'), (100, 0.10, r'100, 0.10'),
              (100, 0.20, r'100, 0.20'), (np.inf, 0.20, r'$\infty$, 0.20'), (np.inf, 1.0, r'$\infty$, 1 (physical)')]
    rows = []
    for k, (Rv, r, lab) in enumerate(ladder):
        sub = G[np.isclose(G.rho, r) & ((G.R == Rv) if np.isfinite(Rv) else ~np.isfinite(G.R))]
        rows.append(pd.DataFrame({'lab': lab, 'eta': 100 * sub.eta_b.values, 'k': k}))
    D = pd.concat(rows)
    order = [l for _, _, l in ladder]
    pal = [B4, B5, B6, B7, B8, B9]
    sns.violinplot(data=D, y='lab', x='eta', order=order, hue='lab', hue_order=order, palette=pal, cut=0, inner=None,
                   density_norm='width', width=0.82, linewidth=0, ax=a, legend=False, split=False, alpha=0.55)
    np.random.seed(0)
    sns.stripplot(data=D, y='lab', x='eta', order=order, color=INK, size=1.4, alpha=0.35, jitter=0.18, ax=a)
    med = D.groupby('lab').eta.median()
    for i, l in enumerate(order):
        a.plot([med[l]] * 2, [i - 0.38, i + 0.38], color=INK, lw=2.2, solid_capstyle='round', zorder=5)
        a.text(103, i, f'{med[l]:.0f}%', va='center', ha='left', fontsize=PT - 2, color=INK, clip_on=False)
    a.set_xlim(0, 100); a.set_xlabel(r'$\eta_E(R,\rho)$ (%), median as bar'); a.set_ylabel(r'Allowance $(R,\rho)$', fontsize=PT - 1.5)
    a.tick_params(axis='y', labelsize=PT - 2); a.grid(axis='y', visible=False)
    title(a, 'b', 'Bounded emission reach')
    # (c) joint counterfactual: share of the requests that need one which an allowance closes
    a = ax[2]; ends = {}
    for r in RHO_SHOW:
        sub = Gn[np.isclose(Gn.rho, r)]
        m = sub.groupby('x').ok.mean().sort_index()
        a.step(m.index, 100 * m.values, where='post', color=RHO_C[r], lw=1.5, alpha=0.95)
        a.plot(m.index, 100 * m.values, ls='none', marker=RHO_M[r], ms=4.2, color=RHO_C[r], mec='white', mew=0.5)
        ends[r] = m.iloc[-1] / 1.0
    budget_axis(a); a.set_ylim(-3, 103); a.set_ylabel('Requests attained (%)')
    h = [Line2D([0], [0], color=RHO_C[r], marker=RHO_M[r], ms=4, mec='white', lw=1.4) for r in RHO_SHOW]
    a.legend(h, [f'{r:g}' for r in RHO_SHOW], title=r'Band $\rho$', loc='center left', fontsize=PT - 2.5,
             title_fontsize=PT - 2.5, ncol=1, handlelength=1.3, borderpad=0.3, handletextpad=0.3, bbox_to_anchor=(0.0, 0.62))
    title(a, 'c', 'Joint counterfactual')
    fig.tight_layout(w_pad=0.5); FS.even_gaps(fig)
    save(fig, 'fig_attainability_map')


def fig6():
    R, G = load_case('10'); Ru = R[R.unique]
    need = Ru[Ru.G_ref > Ru.G_phys + EPS].copy()
    F = G[G.unique & (G.kind == 'budget_frontier') & G.reachable].copy()
    S = pd.read_csv(f'{PROC}/additional_unit_sets.csv')
    fig, ax = plt.subplots(1, 3, figsize=(DCOL, 2.45), gridspec_kw={'width_ratios': [1.1, 1.15, 1.0]})
    # (a) budget-only frontier: median and quartile fan
    a = ax[0]
    for c in CUTS:
        sub = F[np.isclose(F.cut, c)]
        q = sub.groupby('R').G.quantile([0.25, 0.5, 0.75]).unstack()
        a.fill_between(q.index, q[0.25], q[0.75], color=CUT_COL[c], alpha=0.18, lw=0)
        a.plot(q.index, q[0.5], color=CUT_COL[c], marker=CUT_MK[c], ms=4.5, mec='white', mew=0.6, lw=1.6, label=CUT_LAB[c])
        mf = Ru[np.isclose(Ru.cut, c) & Ru.reachable].mw_full.median()
        a.axvline(mf, color=CUT_COL[c], ls=':', lw=1.0)
    logx(a, [5, 10, 20, 40, 80, 160, 320]); a.set_ylim(-0.02, 1.02)
    a.set_xlabel(r'Budget $R$ (MW), bands free'); a.set_ylabel(r'$G^{*}(R,1)$, median and IQR')
    a.legend(title='Cut', loc='upper right', fontsize=PT - 2, title_fontsize=PT - 2)
    title(a, 'a', 'Budget-only frontier')
    # (b) band against budget counterfactual, one point per request, with marginal quartiles
    a = ax[1]
    d = need.dropna(subset=['rho_star', 'R_star'])
    for c in CUTS:
        sub = d[np.isclose(d.cut, c)]
        a.scatter(sub.R_star, sub.rho_star, s=16, color=CUT_COL[c], marker=CUT_MK[c], edgecolor='white', linewidth=0.4,
                  alpha=0.9, label=CUT_LAB[c], zorder=3)
    a.axvline(50, color=INK, ls='--', lw=0.8); a.axhline(0.05, color=INK, ls='--', lw=0.8)
    a.text(50, 0.985, r' $R_0$', transform=a.get_xaxis_transform(), va='top', fontsize=PT - 2)
    a.text(0.99, 0.05, r'$\rho_0$ ', transform=a.get_yaxis_transform(), ha='right', va='bottom', fontsize=PT - 2)
    a.set_xscale('log'); a.minorticks_off(); a.set_xticks([25, 50, 100, 200]); a.set_xticklabels(['25', '50', '100', '200'])
    a.set_ylim(0.03, max(0.5, d.rho_star.max() * 1.08)); a.set_xlim(left=min(15, d.R_star.min() * 0.8))
    a.xaxis.set_minor_formatter(mt.NullFormatter())
    # medians of each counterfactual, drawn as a crosshair
    a.axvline(d.R_star.median(), color=B7, lw=0.8, alpha=0.7); a.axhline(d.rho_star.median(), color=B7, lw=0.8, alpha=0.7)
    a.text(d.R_star.median(), 0.985, ' median', transform=a.get_xaxis_transform(), va='top', fontsize=PT - 3, color=B8)
    a.set_xlabel(r'Minimal budget $R^{*}$ (MW)'); a.set_ylabel(r'Minimal band $\rho^{*}$')
    a.legend(title='Cut', loc='upper left', fontsize=PT - 2, title_fontsize=PT - 2, handletextpad=0.2, borderpad=0.3)
    title(a, 'b', 'Band and budget counterfactuals')
    # (c) exact unit counterfactual by tolerance
    a = ax[2]
    rows, shares = [], []
    for eps in (0.02, 0.05, 0.10):
        s = S[(S.eps == eps) & (S['size'] > 0)]['size']
        rows.append(rf'$\varepsilon={eps:g}$' + f'\n($n={len(s)}$)')
        shares.append([(s == 1).mean(), (s == 2).mean(), (s == 3).mean(), (s > 3).mean()])
    FS.stacked_shares(a, rows, ['1', '2', '3', '>3'], shares, [B3, B5, B7, B9], r'$|S^{*}|$')
    a.set_xlabel('Share of requests'); a.set_title('c) Unit counterfactual, all units', loc='left', pad=17)
    fig.tight_layout(w_pad=0.9); FS.even_gaps(fig)
    save(fig, 'fig_frontier_counterfactuals')


if __name__ == '__main__':
    fig5(); fig6()
