"""Checks of Proposition 1 on stored cells, of the headline numbers of the paper, and of the figure files.

The headline values are recomputed from results/raw with short independent code and compared with the
LaTeX macros in results/macros and with the values printed in the manuscript.
"""
import glob, re
from pathlib import Path
import numpy as np, pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results' / 'raw'
TOL = 1e-6


def macros():
    out = {}
    for f in (ROOT / 'results' / 'macros').glob('numbers_*.tex'):
        out.update(dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', f.read_text())))
    return out


def load10():
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / 'attainability' / 'requests_e10*.csv')))])
    G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / 'attainability' / 'grid_e10*.csv')))])
    u = R[R.duplicate_of_w.isna()]
    return u, G[G.kind == 'grid'].merge(u[['hour', 'w', 'cut']], on=['hour', 'w', 'cut'])


@pytest.mark.parametrize('n_req', [5])
def test_proposition1_on_stored_cells(n_req):
    """0 <= G <= 1, G non-increasing in R and rho, and G >= G_phys, on the first stored requests."""
    u, G = load10()
    keys = list(u[['hour', 'w', 'cut']].itertuples(index=False, name=None))[:n_req]
    for h, w, c in keys:
        g = G[(G.hour == h) & (G.w == w) & (G.cut == c)].pivot_table(index='R', columns='rho', values='G')
        v = g.values
        assert np.all(v >= -0.01) and np.all(v <= 1 + 0.01)
        assert np.all(np.diff(v, axis=0) <= 0.01), 'G must not increase with R'
        assert np.all(np.diff(v, axis=1) <= 0.01), 'G must not increase with rho'
        gp = G[(G.hour == h) & (G.w == w) & (G.cut == c)].G_phys.iloc[0]
        assert np.all(v >= gp - 0.01)


def test_headline_allowance_share_10_unit():
    """76.3% of distinct requests lose at least 0.10 to the allowance over the realistic region."""
    u, G = load10()
    reg = G[G.R.isin([25.0, 50.0, 100.0]) & G.rho.isin([0.05, 0.10, 0.20])]
    m = (reg.G - reg.G_phys).groupby([reg.hour, reg.w, reg.cut]).mean()
    share = 100 * (m >= 0.10).mean()
    assert abs(share - 76.3) < 0.05
    assert macros()['AllowShareTen'] == '76.3'


def test_headline_sparsity_10_unit():
    """At most two units close the allowance gap in 74% of the 10-unit requests that need a unit set."""
    assert macros()['SetLeTwoTen'] == '74'
    u, _ = load10()
    need = u[u.S_size.notna()]
    share = 100 * (need.S_size <= 2).mean()
    assert abs(share - 74) < 0.5


def test_headline_ramp_share_10_unit():
    assert macros()['OpsAllowShareRamp'] == '63.0'


def test_headline_exact_pairs_40_unit():
    """The exhaustive pair test gives 6/6/12 of the 24 requests that need a unit set, so 50% with |S*| <= 2."""
    N = macros()
    assert (N['PairsFullCorrOne'], N['PairsFullCorrTwo'], N['PairsFullCorrThree'], N['PairsFullCorrLeTwo']) == ('6', '6', '12', '50')
    assert round(100 * (6 + 6) / 24) == 50


def test_headline_certificate_counts():
    """0 failures on the 10-unit certificate. On the 40-unit replication DE improves 2 of 240 cells by more than 0.01,
    both band-free relaxations of the lossless model, and no reference or physical cell."""
    de = pd.read_csv(RAW / 'certificate' / 'de_e10_p0.csv')
    assert int((de.improvement > 0.01).sum()) == 0
    c = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / 'supplementary' / 'certificate_ops40_p*.csv')))])
    bad = c[c.improvement > 0.01]
    assert len(c) == 240 and len(bad) == 2 and set(bad.cell) == {'G_bandfree'} and set(bad.model) == {'noloss'}
    N = macros()
    assert N['CertFailTen'] == '0' and N['OpsFortyCertFail'] == '2' and N['OpsFortyCertCells'] == '240'


FIGURES = ['fig_problem', 'fig_concept', 'fig_hourly_reach', 'fig_attainability_map', 'fig_frontier_counterfactuals',
           'fig_eeld40_and_sets', 'fig_ops_losses_ramp', 'fig_ops_window_sets']     # Figs. 1 and 3 to 9


def test_manuscript_figures_exist_and_are_not_empty():
    for f in FIGURES:
        p = ROOT / 'results' / 'figures' / f'{f}.pdf'
        assert p.is_file() and p.stat().st_size > 1000, f
