"""Scientific checks for the lossless attainability model (run with: python -m pytest)."""
import numpy as np
import pytest
from eeldattain.model import System, Request, balance, anchors, efficient_incumbent, eps_point, DATA

S10 = System(f'{DATA}/eeld10_coeffs.csv')


def test_standard_emission_magnitude():
    # standard DEED form: emission at maximum output is about 4.2e4 lb/h; the 1e-2 variant is about 1.8e4
    assert abs(S10.E(S10.hi) - 41626.5) < 1.0
    assert abs(System(f'{DATA}/eeld10_coeffs.csv', escale=0.01).E(S10.hi) - 17941.8) < 1.0


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    P = S10.rand_feasible(1500.0, rng)
    h = 1e-6
    for fun, grad in [(S10.C, S10.gC), (S10.E, S10.gE)]:
        g = grad(P)
        for i in range(S10.n):
            e = np.zeros(S10.n); e[i] = h
            fd = (fun(P + e) - fun(P - e)) / (2 * h)
            assert abs(fd - g[i]) <= 1e-3 * max(1.0, abs(g[i]))


def test_balance_meets_demand_inside_bounds():
    rng = np.random.default_rng(1)
    for PD in (1036.0, 1776.0, 2220.0):
        P = balance(rng.uniform(S10.lo, S10.hi), PD, S10.lo, S10.hi)
        assert P is not None and abs(P.sum() - PD) < 1e-6
        assert np.all(P >= S10.lo - 1e-9) and np.all(P <= S10.hi + 1e-9)


@pytest.fixture(scope='module')
def request16():
    rng = np.random.default_rng(2)
    PD = 1554.0
    A = anchors(S10, PD, rng, K=30, sweep=5)
    Pinc = efficient_incumbent(S10, PD, 0.5, A, K=30)
    Et = 0.95 * S10.E(Pinc)
    Pfr = eps_point(S10, PD, Et, [Pinc, A['Pe']] + A['front'])
    Ft = S10.F(Pfr)
    sc = np.maximum(np.abs(Ft - S10.F(Pinc)), 1e-6 * np.abs(S10.F(Pinc)))
    return Request(S10, PD, Pinc, Ft, sc, Pfr, A['Pc'], A['Pe']), rng


def test_proposition_properties(request16):
    req, rng = request16
    G_phys, _ = req.gap(1e9, 1.0, rng)
    assert G_phys < 1e-6                                  # (iv) reachable request
    grid = np.array([[req.gap(R, r, rng)[0] for r in (0.02, 0.05, 0.1, 0.2)] for R in (10, 25, 50, 100)])
    assert np.all((grid >= -1e-9) & (grid <= 1 + 1e-9))   # (i)
    assert np.all(np.diff(grid, axis=0) <= 1e-2)          # (ii) in R
    assert np.all(np.diff(grid, axis=1) <= 1e-2)          # (ii) in rho
    assert np.all(grid >= G_phys - 1e-2)                  # (iii)


def test_bounded_optimum_is_feasible(request16):
    req, rng = request16
    G, P = req.gap(50.0, 0.05, rng)
    lo, hi = req.band(0.05)
    assert abs(P.sum() - req.PD) < 1e-4
    assert np.all(P >= lo - 1e-6) and np.all(P <= hi + 1e-6)
    assert np.abs(P - req.Pinc).sum() <= 50.0 + 1e-5
