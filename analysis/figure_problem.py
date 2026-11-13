"""Fig. 1 of the manuscript: schematic of the problem setting (no data), in the style of figures_seaborn.py.

(a) Decision space of a three-unit example in the moves of units 1 and 2 (unit 3 balances the demand). The
physical set, the unit bands, the budget and their intersection, the bounded admissible set, are exact
polygons of that example. (b) Objective space with the image of the physical set, the efficient front, the
image of the admissible set, the incumbent, a reachable target and an unreachable target.

Run:  python3 analysis/figure_problem.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import seaborn as sns                               # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from eeldattain import paths                        # noqa: E402

PT = 9
BL = sns.color_palette('Blues', 9)
B2, B3, B4, B5, B6, B7, B8, B9 = BL[1], BL[2], BL[3], BL[4], BL[5], BL[6], BL[7], BL[8]
INK, GREY = '#2b2b2b', '#8a8a8a'
sns.set_theme(context='paper', style='white', font='serif', rc={
    'font.serif': ['STIXGeneral', 'TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': PT, 'axes.titlesize': PT, 'axes.labelsize': PT, 'axes.titlelocation': 'left', 'axes.titleweight': 'bold',
    'axes.titlepad': 5, 'axes.edgecolor': '#b0b0b0', 'axes.linewidth': 0.8, 'text.color': INK, 'axes.labelcolor': INK,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'pdf.fonttype': 42})


def clip(subject, clipper):
    """Sutherland-Hodgman clipping of a polygon by a convex polygon (both counter-clockwise)."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    def cut(p, q, a, b):
        x1, y1, x2, y2, x3, y3, x4, y4 = *p, *q, *a, *b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    out = list(subject)
    for a, b in zip(clipper, clipper[1:] + clipper[:1]):
        inp, out = out, []
        for p, q in zip(inp, inp[1:] + inp[:1]):
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(cut(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(cut(p, q, a, b))
    return out


def strip_box(xl, xh, yl, yh, sl, sh):
    """Box [xl,xh] x [yl,yh] intersected with sl <= x + y <= sh, as a counter-clockwise polygon."""
    box = [(xl, yl), (xh, yl), (xh, yh), (xl, yh)]
    big = 50
    lower = [(sl - (-big), -big), (big, -big), (big, big), (-big, big), (-big, sl + big)]   # x + y >= sl
    lower = [(-big, sl + big), (sl + big, -big), (big, -big), (big, big), (-big, big)]
    upper = [(-big, -big), (big, -big), (big, sh - big), (sh - big, big), (-big, big)]     # x + y <= sh
    return clip(clip(box, lower), upper)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.95), gridspec_kw={'width_ratios': [1, 1.08], 'wspace': 0.12})
    # ---------------- (a) decision space, three units, moves of units 1 and 2 (unit 3 balances)
    omega = strip_box(-2.2, 1.6, -1.4, 2.0, -2.5, 2.4)            # unit limits of 1, 2 and 3
    band = strip_box(-1.0, 1.0, -0.7, 0.7, -0.9, 0.9)              # |dP1|<=b1, |dP2|<=b2, |dP1+dP2|<=b3
    R = 1.6                                                        # |dP1|+|dP2|+|dP1+dP2| <= R
    budget = [(R / 2, 0), (0, R / 2), (-R / 2, R / 2), (-R / 2, 0), (0, -R / 2), (R / 2, -R / 2)]
    adm = clip(clip(omega, band), budget)
    a.add_patch(Polygon(omega, closed=True, fc=B2, ec=B4, lw=0.8))
    a.add_patch(Polygon(adm, closed=True, fc=B6, ec='none', alpha=0.95))
    a.add_patch(Polygon(band, closed=True, fill=False, ec=B8, lw=1.0, ls=(0, (3, 1.6))))
    a.add_patch(Polygon(budget, closed=True, fill=False, ec=INK, lw=0.9, ls=(0, (1, 1.2))))
    a.axhline(0, color=GREY, lw=0.5, zorder=0); a.axvline(0, color=GREY, lw=0.5, zorder=0)
    a.plot(0, 0, 'o', color=INK, ms=4.5, mec='white', mew=0.6, zorder=5)
    a.annotate(r'$P^{\mathrm{inc}}$', (0, 0), xytext=(-1.55, -1.12), fontsize=PT - 1,
               arrowprops=dict(arrowstyle='-', color=INK, lw=0.5, shrinkB=3))
    a.text(-1.95, 1.62, r'$\Omega_h$', fontsize=PT, color=B9)
    a.annotate(r'$\mathcal{A}_h(R,\rho)$', (0.35, -0.38), xytext=(0.62, -1.25), fontsize=PT - 1, color=B9,
               arrowprops=dict(arrowstyle='-', color=B9, lw=0.5))
    a.annotate('bands', (0.6, 0.3), xytext=(0.75, 1.0), fontsize=PT - 2, color=B8,
               arrowprops=dict(arrowstyle='-', color=B8, lw=0.5))
    a.annotate('budget', (-0.8, 0.55), xytext=(-1.95, 1.0), fontsize=PT - 2, color=INK,
               arrowprops=dict(arrowstyle='-', color=INK, lw=0.5))
    a.set_xlim(-2.35, 1.75); a.set_ylim(-1.55, 2.15)
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel(r'$P_1-P_1^{\mathrm{inc}}$', labelpad=2); a.set_ylabel(r'$P_2-P_2^{\mathrm{inc}}$', labelpad=2)
    a.set_title('a) Decision space')
    # ---------------- (b) objective space
    t = np.linspace(0, 1, 200)
    C = 0.35 + 3.0 * t
    E = 0.55 + 2.25 * (1 - t) ** 2.3                                # convex decreasing front from P^C to P^E
    reg = np.vstack([np.c_[C, E], [[3.6, E[-1]], [3.6, 2.95], [C[0], 2.95]]])
    b.add_patch(Polygon(reg, closed=True, fc=B2, ec='none'))
    i0 = 40; Ci, Ei = C[i0], E[i0]                                  # incumbent on the front
    # image of the admissible set: a region above the front that contains the incumbent on its boundary
    blob_t = np.linspace(0, 2 * np.pi, 160)
    ex, ey, rot = 0.46, 0.27, np.deg2rad(18)
    ux, uy = ex * np.cos(blob_t), ey * np.sin(blob_t)
    bx0, by0 = ux * np.cos(rot) - uy * np.sin(rot), ux * np.sin(rot) + uy * np.cos(rot)
    k = int(np.argmin(bx0 + 0.6 * by0))                             # lower-left point of the ellipse
    bx, by = Ci + bx0 - bx0[k], Ei + by0 - by0[k]
    by = np.maximum(by, np.interp(bx, C, E))
    b.fill(bx, by, color=B6, alpha=0.9, lw=0)
    b.plot(C, E, color=INK, lw=1.3)
    b.axhline(E[-1], color=GREY, lw=0.7, ls=(0, (4, 2)), xmax=0.97)
    b.text(0.12, E[-1] + 0.05, r'$E_{\min}$', fontsize=PT - 1, color=GREY)
    cut = 0.27
    Et = Ei * (1 - cut); it = int(np.argmin(np.abs(E - Et))); Ct = C[it]
    b.plot(Ci, Ei, 'o', color=INK, ms=4.5, mec='white', mew=0.6, zorder=5)
    b.plot(Ct, E[it], '*', color=B9, ms=10, mec='white', mew=0.5, zorder=5)
    b.add_patch(FancyArrowPatch((Ci - 0.12, Ei), (Ci - 0.12, E[it]), arrowstyle='-|>', mutation_scale=7, color=B8, lw=0.9))
    b.plot([Ci - 0.12, Ct], [E[it], E[it]], color=B8, lw=0.6, ls=(0, (2, 1.5)))
    b.text(Ci - 0.2, (Ei + E[it]) / 2, r'$c$', ha='right', va='center', fontsize=PT, color=B8)
    b.text(Ci - 0.1, Ei + 0.08, 'inc', fontsize=PT - 2, ha='right')
    b.text(Ct - 0.1, E[it] - 0.1, 'reachable\ntarget', fontsize=PT - 2, color=B9, va='top', ha='right')
    Cu, Eu = C[-1], E[-1] - 0.32
    b.plot(Cu, Eu, '*', color='white', ms=10, mec=B9, mew=0.9, zorder=5)
    b.text(Cu - 0.12, Eu, 'unreachable\ntarget', fontsize=PT - 2, color=B9, ha='right', va='center')
    b.text(2.55, 2.45, r'$F(\Omega_h)$', fontsize=PT - 1, color=B8)
    b.text(bx.mean(), by.mean(), r'$F(\mathcal{A}_h)$', fontsize=PT - 2, color='white', ha='center', va='center')
    b.set_xlim(0, 3.6); b.set_ylim(0, 2.95); b.set_xticks([]); b.set_yticks([])
    b.set_xlabel('Fuel cost $C$', labelpad=2); b.set_ylabel('Emission $E$', labelpad=2)
    b.set_title('b) Objective space')
    os.makedirs(paths.FIGURES, exist_ok=True)
    fig.savefig(f'{paths.FIGURES}/fig_problem.pdf', metadata={'CreationDate': None, 'ModDate': None})
    print('wrote fig_problem')


if __name__ == '__main__':
    main()
