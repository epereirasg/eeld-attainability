"""Data figures of the manuscript (Figs. 3, 4 and 7 to 9), rebuilt from results/raw.

Every figure reuses the loading and selection code of attainability.py and operational.py,
so each plotted value passes through the same code path as the numbers in the text.
The script writes PDF files to results/figures/.

Style: one sequential seaborn scale ("Blues"), mean lines with 95% bootstrap bands over requests, box
plots with the individual requests overlaid, panel titles at the top left and 9 pt type (the body text
is 10 pt and figures are placed at 100% width). Categories carry a second channel (marker or line
style) besides colour.

Run:  PYTHONPATH=src python3 analysis/figures_seaborn.py [--no-concept]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

import attainability as AT                             # noqa: E402
import operational as OP                               # noqa: E402
from common import PROC, load_case, paths              # noqa: E402

FIG = str(paths.FIGURES)

import matplotlib                                      # noqa: E402
import matplotlib.ticker                               # noqa: E402
import matplotlib.pyplot as plt                        # noqa: E402
import seaborn as sns                                  # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.patches import Patch                   # noqa: E402

# ------------------------------------------------------------------------------------------- style
# One sequential scale (seaborn "Blues"). Categories are ordered from light to dark and always carry a
# second channel (marker or line style). Bands are 95% bootstrap confidence intervals over requests
# unless a caption states otherwise (seed fixed for reproducibility).
COL, DCOL = 3.5, 7.16
PT = 9
SEED = 0
BL = sns.color_palette('Blues', 9)          # light -> dark
B9, B8, B7, B6, B5, B4, B3, B2 = BL[8], BL[7], BL[6], BL[5], BL[4], BL[3], BL[2], BL[1]
INK, GREY = '#2b2b2b', '#8a8a8a'

sns.set_theme(context='paper', style='whitegrid', palette='Blues_d', font='serif', rc={
    'font.serif': ['STIXGeneral', 'TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': PT, 'axes.titlesize': PT, 'axes.labelsize': PT,
    'xtick.labelsize': PT - 1, 'ytick.labelsize': PT - 1, 'legend.fontsize': PT - 1, 'legend.title_fontsize': PT - 1,
    'axes.edgecolor': '#b0b0b0', 'axes.linewidth': 0.8, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK, 'grid.color': '#e6e9ee', 'grid.linewidth': 0.7,
    'axes.titlelocation': 'left', 'axes.titlepad': 6, 'axes.titleweight': 'bold',
    'legend.frameon': True, 'legend.framealpha': 0.92, 'legend.edgecolor': '#d0d0d0', 'legend.fancybox': True,
    'lines.linewidth': 1.6, 'lines.markersize': 5, 'patch.linewidth': 0.5,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'pdf.fonttype': 42,
    'legend.handlelength': 1.8, 'hatch.linewidth': 0.5, 'hatch.color': '#1f4e79',
})

CUTS = [0.025, 0.05, 0.10]
CUT_LAB = {0.025: '2.5%', 0.05: '5%', 0.10: '10%'}
CUT_COL = {0.025: B4, 0.05: B6, 0.10: B9}
CUT_MK = {0.025: '^', 0.05: 's', 0.10: 'o'}
CUT_DASH = {'2.5%': (1, 1.2), '5%': (4, 1.5), '10%': ''}
RHO_COL = {0.02: B3, 0.05: B5, 0.10: B7, 0.20: B9}
RHO_MK = {0.02: '^', 0.05: 's', 0.10: 'D', 0.20: 'o'}
TAU_COL = {10: B3, 15: B5, 30: B7, 60: B9}
TAU_MK = {10: '^', 15: 's', 30: 'D', 60: 'o'}


def title(ax, letter, text):
    ax.set_title(f'{letter}) {text}', loc='left')


def logx(ax, ticks):
    ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(ticks); ax.set_xticklabels([f'{t:g}' for t in ticks])


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f'{FIG}/{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    print('wrote', name)


PANEL_GAP = 0.20     # blank space between neighboring panels in inches, the same in every figure


def panel_groups(fig):
    """Group the axes of a figure into panels: twin axes share the position of their host axes."""
    groups = []
    for a in sorted(fig.axes, key=lambda x: x.get_position().x0):
        p = a.get_position()
        for g in groups:
            q = g[0].get_position()
            if abs(p.x0 - q.x0) < 1e-6 and abs(p.width - q.width) < 1e-6:
                g.append(a); break
        else:
            groups.append([a])
    return groups


def group_extents(fig, groups):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.dpi_scale_trans.inverted()
    ext = []
    for g in groups:
        bb = [a.get_tightbbox(r).transformed(inv) for a in g]
        ext.append((min(b.x0 for b in bb), max(b.x1 for b in bb)))
    return ext


def even_gaps(fig, groups=None, gap=None):
    """Give every pair of neighboring panels the same blank gap, widening the panels to keep the figure width.

    Each group lists the axes of one panel, its host axes first. Twin axes keep the position of the host, and an
    attached colorbar keeps its offset from the right edge of the host.
    """
    groups = groups or panel_groups(fig)
    gap = PANEL_GAP if gap is None else gap
    W = fig.get_figwidth()
    ext = group_extents(fig, groups); total = ext[-1][1] - ext[0][0]     # keep the overall width of the figure
    for _ in range(12):
        ext = group_extents(fig, groups)
        gaps = [ext[i + 1][0] - ext[i][1] for i in range(len(ext) - 1)]
        drift = total - (ext[-1][1] - ext[0][0])
        if max(abs(g - gap) for g in gaps) < 2e-3 and abs(drift) < 2e-3:
            break
        extra = sum(gaps) - gap * len(gaps) + drift
        ws = [g[0].get_position().width * W for g in groups]
        x = ext[0][0]
        for i, g in enumerate(groups):
            m = g[0]; p = m.get_position()
            lead = p.x0 * W - ext[i][0]; tail = ext[i][1] - (p.x0 + p.width) * W
            new_w = ws[i] + extra * ws[i] / sum(ws); new_x0 = x + lead
            for a in g:
                q = a.get_position()
                if abs(q.x0 - p.x0) < 1e-6 and abs(q.width - p.width) < 1e-6:
                    a.set_position([new_x0 / W, q.y0, new_w / W, q.height])
                else:
                    off = q.x0 * W - (p.x0 + p.width) * W
                    a.set_position([(new_x0 + new_w + off) / W, q.y0, q.width, q.height])
            x = new_x0 + new_w + tail + gap


def band_lines(ax, data, x, y, hue, colors, markers, labels=None, ci=True):
    """Mean line per hue level with a 95% bootstrap band over requests (seaborn lineplot)."""
    for lev, col in colors.items():
        sub = data[np.isclose(data[hue], lev)]
        if not len(sub):
            continue
        sns.lineplot(data=sub, x=x, y=y, ax=ax, color=col, marker=markers[lev], markersize=5, markeredgecolor='white',
                     markeredgewidth=0.6, errorbar=('ci', 95) if ci else None, n_boot=2000, seed=SEED,
                     err_kws={'alpha': 0.16, 'lw': 0}, label=(labels or {}).get(lev, f'{lev:g}'))


def stacked_shares(ax, rows, classes, shares, colors, legend_title):
    """100% stacked horizontal bars: one row per setting, one segment per size class, labels inside segments."""
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for k, (cl, col) in enumerate(zip(classes, colors)):
        v = np.array([sh[k] for sh in shares])
        ax.barh(y, v, left=left, height=0.62, color=col, edgecolor='white', linewidth=1.2, label=cl)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        for yi, li, vi in zip(y, left, v):
            if vi >= 0.06:
                ax.text(li + vi / 2, yi, f'{100*vi:.0f}%', ha='center', va='center', fontsize=PT - 1.5 if vi >= 0.1 else PT - 3,
                        color='white' if lum < 0.55 else INK)
        left += v
    ax.set_yticks(y); ax.set_yticklabels(rows); ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
    h, l = ax.get_legend_handles_labels()
    l = [f'{legend_title} $=$ {l[0]}'] + l[1:]
    ax.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(classes), frameon=False,
              handlelength=1.1, columnspacing=0.9, borderaxespad=0.15, fontsize=PT - 1.5)


# ------------------------------------------------------------------------------- Fig. 3
def fig_concept():
    """One real request in objective space. Same request, anchors and seed as attainability.concept_request."""
    from eeldattain.model import DATA, Request, System, anchors, balance, eps_point
    V = [json.loads(l) for f in sorted(glob.glob(f'{AT.ATT}/vectors_e10*.jsonl')) for l in open(f)]
    R10, _ = load_case('10')
    pick = R10[(R10.unique) & (R10.cut == 0.05) & (R10.reachable) & (R10.G_ref > 0.25) & (R10.G_ref < 0.6)].sort_values('hour').iloc[0]
    v = [x for x in V if x['hour'] == pick.hour and x['w'] == pick.w and x['cut'] == pick.cut][0]
    s = System(f'{DATA}/eeld10_coeffs.csv'); rng = np.random.default_rng(3)
    Pinc = np.array(v['Pinc']); Ft = np.array(v['Ft']); PD = v['PD']
    A = anchors(s, PD, rng, K=40, sweep=25); FR = np.array([s.F(P) for P in A['front']])
    FR = FR[np.argsort(FR[:, 1])]; keep = [0]
    for i in range(1, len(FR)):
        if FR[i, 0] < FR[keep[-1], 0] - 1e-9: keep.append(i)
    FR = FR[keep]
    req = Request(s, PD, Pinc, Ft, np.array(v['scale']), np.array(v['Pfr']), A['Pc'], A['Pe'])

    def cloud(R, rho, n):
        lo, hi = req.band(rho); out = []
        while len(out) < n:
            P = balance(rng.uniform(lo, hi), PD, lo, hi)
            if P is None: continue
            dd = np.abs(P - Pinc).sum(); P = P if dd <= R else Pinc + R / dd * (P - Pinc); out.append(s.F(P))
        return np.array(out)
    Om = cloud(1e9, 1.0, 1500)   # drawn only to keep the random stream of the original figure (outside the window)
    Ab = cloud(pick.R0, pick.rho0, 1500)
    Gb, Pb = req.gap(pick.R0, pick.rho0, rng); Fb = s.F(Pb); Finc = s.F(Pinc)
    # Dense local front for drawing only: epsilon-constraint solves over the plotted emission range, warm-started
    # from the incumbent, the target dispatch and the anchors. Run after the gap, so the random stream is unchanged.
    Elo, Ehi = min(Ft[1], Fb[1]) - 0.02 * Finc[1], Finc[1] + 0.03 * Finc[1]
    dense, prev = [], np.array(v['Pfr'])
    for Ee in np.linspace(Elo, Ehi, 40):
        P = eps_point(s, PD, Ee, [prev, Pinc, np.array(v['Pfr']), A['Pc'], A['Pe']] + A['x0'][:4])
        if P is not None:
            dense.append(s.F(P)); prev = P
    pts = np.vstack([FR] + ([np.array(dense)] if dense else []) + [Finc[None], Ft[None]])
    pts = pts[np.argsort(pts[:, 1])]; keep = [0]
    for i in range(1, len(pts)):
        if pts[i, 0] < pts[keep[-1], 0] - 1e-9: keep.append(i)
    FR = pts[keep]
    on_front = bool(np.any(np.all(np.isclose(FR, Finc[None]), axis=1)))
    print('incumbent kept as nondominated:', on_front, '| front points:', len(FR))
    fig, a = plt.subplots(figsize=(COL, 2.75))
    sns.kdeplot(x=Ab[:, 0] / 1e3, y=Ab[:, 1] / 1e3, ax=a, fill=True, levels=6, thresh=0.06, cmap='Blues', alpha=0.75)
    a.scatter(Ab[:, 0] / 1e3, Ab[:, 1] / 1e3, s=1.6, color=B6, alpha=0.35, lw=0, rasterized=True)
    a.plot(FR[:, 0] / 1e3, FR[:, 1] / 1e3, color=INK, lw=1.4)
    a.annotate('', xy=Fb / 1e3, xytext=Finc / 1e3, arrowprops=dict(arrowstyle='-|>', color=B9, lw=1.2, mutation_scale=9))
    a.plot([Fb[0] / 1e3, Ft[0] / 1e3], [Fb[1] / 1e3, Ft[1] / 1e3], color=B7, lw=1.0, ls=(0, (3, 2)))
    a.plot(*Finc / 1e3, 'o', color=INK, ms=6.5, mec='white', mew=0.7)
    a.plot(*Ft / 1e3, '*', color=B9, mec='white', mew=0.6, ms=13)
    a.plot(*Fb / 1e3, 'D', color=B6, mec=INK, mew=0.6, ms=6)
    a.annotate('incumbent', xy=Finc / 1e3, xytext=(6, 6), textcoords='offset points', fontsize=PT - 1)
    a.annotate('target', xy=Ft / 1e3, xytext=(8, -2), textcoords='offset points', fontsize=PT - 1)
    a.annotate(f'bounded optimum\n$G^*={Gb:.2f}$', xy=Fb / 1e3, xytext=(9, -2), textcoords='offset points', fontsize=PT - 1,
               va='top', ha='left', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    pts = np.vstack([Ab, Finc[None], Ft[None], Fb[None]]) / 1e3
    lo_, hi_ = pts.min(axis=0), pts.max(axis=0); span = hi_ - lo_
    a.set_xlim(lo_[0] - 0.08 * span[0], hi_[0] + 0.08 * span[0]); a.set_ylim(lo_[1] - 0.08 * span[1], hi_[1] + 0.08 * span[1])
    a.set_xlabel(r'Fuel cost $C$ (k\$/h)'); a.set_ylabel(r'Emission $E$ (klb/h)')
    a.set_title(f'Hour {int(pick.hour)}, $w={pick.w:g}$, 5% cut', loc='left')
    hd = [Patch(fc=B4, ec='none', alpha=0.8), Line2D([], [], color=INK, lw=1.4)]
    a.legend(hd, [r'Random points of $\mathcal{A}_h(R_0,\rho_0)$', 'Efficient front'], loc='upper center',
             bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=PT - 1, frameon=False, columnspacing=1.2)
    save(fig, 'fig_concept')
    return dict(hour=int(pick.hour), w=float(pick.w), G=float(Gb))


# ------------------------------------------------------------------------------- Fig. 4
def figs10():
    R, G, Ru, Gu, hw, need = AT.analyse('10')
    # ---- physical reach
    fig, ax = plt.subplots(1, 3, figsize=(DCOL, 2.2), gridspec_kw={"width_ratios": [1.25, 1.25, 0.95]})
    a = ax[0]
    a.fill_between(hw.index, 0, hw.PD, color=B2, step=None, lw=0)
    a.plot(hw.index, hw.PD, color=B4, lw=1.0, label='Demand (left)')
    a.set_ylabel('Demand (MW)'); a.set_xlabel('Hour'); a.set_ylim(0, 4700); a.set_yticks([0, 1000, 2000, 3000]); a.set_xticks([1, 6, 12, 18, 24]); a.set_xlim(1, 24)
    b = a.twinx(); b.grid(False)
    b.plot(hw.index, 100 * hw.W, color=B9, marker='o', ms=4, mec='white', mew=0.5, lw=1.5, label='Front width (right)')
    b.set_ylabel(r'Front width $W_h$ (%)'); b.set_ylim(0, 54); b.set_yticks([0, 10, 20, 30, 40])
    for sp in b.spines.values(): sp.set_color('#b0b0b0')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc='upper left', ncol=1, handlelength=1.2, handletextpad=0.3, borderaxespad=0.3)
    title(a, 'a', 'Demand and front width')
    a = ax[1]
    D = Ru.copy(); D['Cut'] = D.cut.map(lambda c: CUT_LAB[round(c, 3)])
    sns.lineplot(data=D, x='hour', y='eta_E', hue='Cut', hue_order=['2.5%', '5%', '10%'], style='Cut', style_order=['2.5%', '5%', '10%'],
                 palette=[CUT_COL[c] for c in CUTS], markers=['^', 's', 'o'], dashes=False, markersize=4.5, markeredgecolor='white',
                 errorbar=None, ax=a)
    a.set_ylim(0, 1.38); a.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); a.set_xlabel('Hour'); a.set_ylabel(r'Mean emission reach $\eta_E$'); a.set_xticks([1, 6, 12, 18, 24]); a.set_xlim(1, 24)
    a.legend(loc='upper center', ncol=3, handlelength=1.2, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.3)
    title(a, 'b', 'Physical reach by hour')
    a = ax[2]
    rs = AT.spearmanr(hw.W, hw.eta)[0]
    sns.scatterplot(x=100 * hw.W, y=hw.eta, hue=hw.PD, size=hw.PD, sizes=(18, 70), palette='Blues', edgecolor=INK, linewidth=0.4,
                    ax=a, legend=False, zorder=3)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(hw.PD.min(), hw.PD.max()))
    cb = fig.colorbar(sm, ax=a, pad=0.03, fraction=0.06); cb.set_label('Demand (MW)'); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=PT - 2)
    a.text(0.96, 0.05, f'Spearman {rs:.2f}', transform=a.transAxes, ha='right', va='bottom', fontsize=PT - 1,
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d0d0d0', lw=0.5))
    a.set_xlabel(r'Front width $W_h$ (%)'); a.set_ylabel(r'Mean $\eta_E$ of the hour'); a.set_ylim(0, 1.05)
    title(a, 'c', 'Reach against width')
    fig.tight_layout(w_pad=2.2); even_gaps(fig, [[ax[0], b], [ax[1]], [ax[2], cb.ax]]); save(fig, 'fig_hourly_reach')

    return R, G, Ru, Gu, hw, need


# ------------------------------------------------------------------------------- Fig. 8
def fig_forty(Ru10, need10):
    R4, G4, Ru4, Gu4, hw4, need4 = AT.analyse('40')
    fig, ax = plt.subplots(1, 3, figsize=(DCOL, 2.4), gridspec_kw={'width_ratios': [1.1, 1, 1.15]})
    a = ax[0]
    D = Ru4.copy(); D['GW'] = D.PD / 1000
    band_lines(a, D, 'GW', 'eta_E', 'cut', {0.05: CUT_COL[0.05], 0.10: CUT_COL[0.10]}, CUT_MK,
               labels={0.05: r'$\eta_E$, 5% cut', 0.10: r'$\eta_E$, 10% cut'}, ci=False)
    b = a.twinx(); b.grid(False)
    b.plot(hw4.PD / 1000, 100 * hw4.W, color=GREY, ls=':', marker='x', ms=4, lw=1.1, label='Front width (right)')
    b.set_ylabel('Front width (%)'); b.set_ylim(0, max(100, 1.1 * 100 * hw4.W.max()))
    for sp in b.spines.values(): sp.set_color('#b0b0b0')
    a.set_ylim(0, 1.15); a.set_xlabel('Demand (GW)'); a.set_ylabel(r'Mean emission reach $\eta_E$')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc='lower left'); title(a, 'a', 'Physical reach')
    a = ax[1]
    band_lines(a, Gu4, 'R', 'G_allow', 'rho', {0.05: RHO_COL[0.05], 0.10: RHO_COL[0.10], 0.20: RHO_COL[0.20]}, RHO_MK)
    a.set_ylim(0, 0.75); logx(a, [50, 100, 250, 500]); a.set_xlabel(r'Budget $R$ (MW)'); a.set_ylabel(r'Mean $G^{*}_{\mathrm{allow}}$')
    title(a, 'b', 'Allowance gap'); a.legend(title=r'$\rho$', loc='upper right')
    a = ax[2]
    s10 = need10.dropna(subset=['S_size']).S_size.astype(int)
    X = pd.concat([pd.read_csv(f, dtype={'exact_size': str}) for f in sorted(glob.glob(f'{paths.RAW_OPS}/sets40_*.csv'))]); X = X[X.needs_set]
    C = pd.read_csv(f'{PROC}/supplementary_eeld40_sets_corrected.csv', dtype={'corrected': str, 'exact_size': str})
    rows = [f'10 units\n($n={len(s10)}$)', f'40 units, band-\nactive pairs ($n={len(X)}$)', f'40 units,\nall pairs ($n={len(C)}$)']
    shares = [[(s10 == 1).mean(), (s10 == 2).mean(), (s10 >= 3).mean()],
              [(X.exact_size == '1').mean(), (X.exact_size == '2').mean(), (X.exact_size == '>=3').mean()],
              [(C.corrected == '1').mean(), (C.corrected == '2').mean(), C.corrected.str.startswith('>=3').mean()]]
    stacked_shares(a, rows, ['1', '2', 'no pair found'], shares, [B4, B6, B9], r'$|S^{*}|$')
    a.set_xlabel('Share of requests')
    a.set_title('c) Units to relax', loc='left', pad=17)
    fig.tight_layout(w_pad=0.8); even_gaps(fig); save(fig, 'fig_eeld40_and_sets')


# ------------------------------------------------------------------------------- Figs. 7 and 9
def figs_ops():
    R, Ru, G, M, need = OP.x1_x2(); W = OP.x3()
    fig, ax = plt.subplots(1, 3, figsize=(DCOL, 2.4))
    a = ax[0]
    Mx = M.copy(); Mx['Cut'] = Mx.cut.map(lambda c: CUT_LAB[round(c, 3)])
    a.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls='--', zorder=1)
    sns.scatterplot(data=Mx, x='G_ref', y='G_ref_uniform', hue='Cut', style='Cut', hue_order=['5%', '10%'], style_order=['5%', '10%'],
                    palette=[CUT_COL[0.05], CUT_COL[0.10]], markers=['s', 'o'], s=20, edgecolor='white', linewidth=0.4, ax=a, zorder=3)
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect('equal', adjustable='box')
    a.set_xlabel(r'$G^{*}(R_0,\rho_0)$, no losses'); a.set_ylabel(r'$G^{*}(R_0,\rho_0)$, losses')
    title(a, 'a', f'Effect of losses ($n={len(M)}$)'); a.legend(title='Cut', loc='upper left')
    a = ax[1]
    Gr = G[G.band == 'ramp']
    band_lines(a, Gr, 'R', 'G_allow', 'level', TAU_COL, TAU_MK, labels={t: f'{t}' for t in TAU_COL})
    logx(a, [25, 50, 100, 250]); a.set_xlabel(r'Budget $R$ (MW)'); a.set_ylabel(r'Mean $G^{*}_{\mathrm{allow}}$'); a.set_ylim(0, 0.5)
    title(a, 'b', 'Ramp-derived bands'); a.legend(title=r'$\tau$ (min)', ncol=2, loc='upper right', columnspacing=0.8)
    a = ax[2]
    x = np.sort(need.tau_star.dropna().values)
    sns.histplot(x=x, ax=a, bins=np.arange(10, 70, 5), stat='proportion', color=B3, edgecolor='white', linewidth=0.6, label='Share per 5 min')
    sns.ecdfplot(x=x, ax=a, color=B9, lw=1.8, label='Cumulative share')
    sns.rugplot(x=x, ax=a, color=B7, height=0.035, lw=0.6, alpha=0.7)
    a.axvline(15, color=B7, ls='--', lw=0.9); a.axvline(60, color=B7, ls=':', lw=0.9)
    a.text(16, 0.93, r'$\tau_0$', ha='left', fontsize=PT - 1, color=B8); a.text(59, 0.45, '1 h', ha='right', fontsize=PT - 1, color=B8)
    med = np.median(x); a.plot([med], [0.5], 'o', color=B9, ms=5.5, mec='white', mew=0.6)
    a.annotate(f'median {med:.0f} min', xy=(med, 0.5), xytext=(med + 11, 0.36), fontsize=PT - 2, color=B9,
               arrowprops=dict(arrowstyle='-', color=B9, lw=0.6))
    a.set_ylim(0, 1.02); a.set_xlabel(r'Minimal response time $\tau^{*}$ (min)'); a.set_ylabel('Share of requests'); a.set_xlim(10, 66)
    title(a, 'c', f'Response time ($n={len(need)}$)')
    fig.tight_layout(w_pad=1.0); even_gaps(fig); save(fig, 'fig_ops_losses_ramp')

    # ---- window and unit counterfactual
    L = pd.read_csv(f'{paths.RAW_SUPP}/window_lock.csv'); e = L[L.construction == 'efficient_window']
    lk = set(zip(e[e.locked].hour, e[e.locked].w)); W = W.copy(); W['locked'] = [(h, w) in lk for h, w in zip(W.hour, W.w)]
    fig, ax = plt.subplots(1, 2, figsize=(DCOL, 2.45), gridspec_kw={"width_ratios": [1, 1.15]})
    a = ax[0]
    u = W[~W.locked]; k = W[W.locked]
    a.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls='--', zorder=1)
    a.scatter(u.G_single, u.G_fixed, s=18, marker='o', facecolor='white', edgecolor=B7, lw=0.9, label='Fixed', zorder=3)
    a.scatter(k.G_single, k.G_fixed, s=22, marker='o', facecolor=B4, edgecolor=B8, lw=0.8, label=f'Fixed, locked ({len(k)})', zorder=3)
    a.scatter(W.G_single, W.G_window, s=16, marker='^', color=B9, lw=0, label='Adjustable', zorder=4)
    a.set_xlim(0, 1); a.set_ylim(0, 1.06)
    a.legend(title='Neighbors', loc='lower right', fontsize=PT - 2, title_fontsize=PT - 2)
    title(a, 'a', f'Inter-hour ramp coupling ($n={len(W)}$)')
    a.set_xlabel(r'$G^{*}$, single hour'); a.set_ylabel(r'$G^{*}$, three-hour window')
    a = ax[1]
    s10 = load_case('10')[0]; s10 = s10[s10.unique & (s10.G_ref > s10.G_phys + OP.EPS)].dropna(subset=['S_size']).S_size.astype(int)
    sr = need.dropna(subset=['S_size']).S_size.astype(int)
    rows = [f'Uniform $\\rho_0$, no losses\n($n={len(s10)}$)', f'Ramp $b(\\tau_0)$, losses\n($n={len(sr)}$)']
    shares = [[(s10 == 1).mean(), (s10 == 2).mean(), (s10 == 3).mean(), (s10 > 3).mean()],
              [(sr == 1).mean(), (sr == 2).mean(), (sr == 3).mean(), (sr > 3).mean()]]
    stacked_shares(a, rows, ['1', '2', '3', '>3'], shares, [B3, B5, B7, B9], r'$|S^{*}|$')
    a.set_xlabel('Share of requests'); a.set_title('b) Unit counterfactual, 10 units', loc='left', pad=17)
    fig.tight_layout(w_pad=1.5); even_gaps(fig); save(fig, 'fig_ops_window_sets')


if __name__ == '__main__':
    R, G, Ru, Gu, hw, need = figs10()
    fig_forty(Ru, need)
    figs_ops()
    if '--no-concept' not in sys.argv:
        print(fig_concept())
