"""Hand-checked objectives, constraints, tolerances and degenerate cases on small synthetic systems."""
import numpy as np
import pandas as pd
import pytest

from eeldattain.model import System, Request, SolveStats, balance, solve
from eeldops.model import OpsSystem, OpsRequest, ramp_cons_vec


def two_unit_csv(tmp_path):
    d = pd.DataFrame(dict(unit=[1, 2], Pmin=[10.0, 20.0], Pmax=[100.0, 120.0], a=[0.01, 0.02], b=[2.0, 1.5], c=[10.0, 5.0],
                          e=[50.0, 0.0], f=[0.06, 0.0], alpha=[4.0, 3.0], beta=[-0.05, -0.04], gamma=[6e-4, 5e-4],
                          zeta=[2e-4, 1e-4], **{'lambda': [0.02, 0.01]}))
    p = tmp_path / 'two.csv'; d.to_csv(p, index=False)
    return p, d


def test_cost_and_emission_match_hand_computation(tmp_path):
    p, d = two_unit_csv(tmp_path)
    s = System(str(p)); P = np.array([40.0, 60.0])
    C = sum(d.a[i] * P[i] ** 2 + d.b[i] * P[i] + d.c[i] + abs(d.e[i] * np.sin(d.f[i] * (d.Pmin[i] - P[i]))) for i in range(2))
    E = sum(d.alpha[i] + d.beta[i] * P[i] + d.gamma[i] * P[i] ** 2 + d.zeta[i] * np.exp(d['lambda'][i] * P[i]) for i in range(2))
    assert s.C(P) == pytest.approx(C, rel=1e-12) and s.E(P) == pytest.approx(E, rel=1e-12)
    # the earlier convention scales only the quadratic part of the emission
    E2 = sum(0.01 * (d.alpha[i] + d.beta[i] * P[i] + d.gamma[i] * P[i] ** 2) + d.zeta[i] * np.exp(d['lambda'][i] * P[i]) for i in range(2))
    assert System(str(p), escale=0.01).E(P) == pytest.approx(E2, rel=1e-12)
    # the operational model uses the same objectives
    assert OpsSystem(d).F(P) == pytest.approx(s.F(P), rel=1e-12)


def test_valve_point_gradient_away_from_kinks(tmp_path):
    p, _ = two_unit_csv(tmp_path)
    s = System(str(p)); P = np.array([33.3, 57.1]); h = 1e-6
    for fun, grad in ((s.C, s.gC), (s.E, s.gE)):
        fd = np.array([(fun(P + h * e) - fun(P - h * e)) / (2 * h) for e in np.eye(2)])
        assert np.allclose(fd, grad(P), rtol=1e-5, atol=1e-6)


def test_coefficient_file_validation(tmp_path):
    p, d = two_unit_csv(tmp_path)
    d.drop(columns='zeta').to_csv(tmp_path / 'bad.csv', index=False)
    with pytest.raises(ValueError, match='missing coefficient columns'):
        System(str(tmp_path / 'bad.csv'))
    d2 = d.copy(); d2.loc[0, 'Pmax'] = 5.0; d2.to_csv(tmp_path / 'bad2.csv', index=False)
    with pytest.raises(ValueError, match='Pmax <= Pmin'):
        System(str(tmp_path / 'bad2.csv'))


def test_kron_loss_and_balance_residual():
    d = pd.DataFrame(dict(Pmin=[0.0, 0.0], Pmax=[100.0, 100.0], a=[0, 0], b=[1, 1], c=[0, 0], e=[0, 0], f=[0, 0],
                          alpha=[0, 0], beta=[0, 0], gamma=[0, 0], zeta=[0, 0], **{'lambda': [0, 0]}))
    B = np.array([[1e-4, 2e-5], [2e-5, 2e-4]])
    s = OpsSystem(d, B)
    P = np.array([50.0, 30.0])
    PL = 1e-4 * 2500 + 2 * 2e-5 * 1500 + 2e-4 * 900
    assert s.loss(P) == pytest.approx(PL) and s.residual(P, 70.0) == pytest.approx(80.0 - 70.0 - PL)
    Pb = s.balance(np.array([10.0, 10.0]), 70.0, s.lo, s.hi)
    assert Pb is not None and abs(Pb.sum() - 70.0 - s.loss(Pb)) < 1e-6


def test_balance_respects_limits_and_reports_infeasible_demand(tmp_path):
    p, _ = two_unit_csv(tmp_path); s = System(str(p))
    P = balance(np.array([90.0, 110.0]), 150.0, s.lo, s.hi)
    assert abs(P.sum() - 150.0) < 1e-6 and np.all(P >= s.lo) and np.all(P <= s.hi)
    assert balance(np.array([50.0, 50.0]), 500.0, s.lo, s.hi) is None        # above total capacity
    assert balance(np.array([50.0, 50.0]), 10.0, s.lo, s.hi) is None         # below total minimum


def test_solve_counts_runs_without_feasible_end_point(tmp_path):
    p, _ = two_unit_csv(tmp_path); s = System(str(p))
    impossible = {'type': 'ineq', 'fun': lambda P: -1.0 - 0 * P.sum(), 'jac': lambda P: np.zeros_like(P)}
    SolveStats.reset()
    assert solve(s, 100.0, s.C, s.gC, [np.array([40.0, 60.0])], cons=(impossible,)) is None
    assert SolveStats.calls == 1 and SolveStats.no_feasible_endpoint == 1
    v, P = solve(s, 100.0, s.C, s.gC, [np.array([40.0, 60.0])])
    assert abs(P.sum() - 100.0) < 1e-5 and SolveStats.no_feasible_endpoint == 1


def _request(tmp_path, cut=0.05):
    p, _ = two_unit_csv(tmp_path); s = System(str(p)); PD = 100.0
    Pinc = np.array([40.0, 60.0]); Finc = s.F(Pinc)
    Ft = np.array([Finc[0] * 1.01, Finc[1] * (1 - cut)])
    sc = np.maximum(np.abs(Ft - Finc), 1e-6 * np.abs(Finc))
    return s, Request(s, PD, Pinc, Ft, sc, None, None, None), Finc, Ft, sc


def test_target_normalization_and_scale_floor(tmp_path):
    s, req, Finc, Ft, sc = _request(tmp_path)
    assert req.e0 == pytest.approx(np.sqrt(2.0))           # each component is normalized by its own requested change
    assert req.d(req.Pinc) == pytest.approx(req.e0)
    Ft2 = np.array([Finc[0], Ft[1]])                        # no cost change requested: the floor applies
    sc2 = np.maximum(np.abs(Ft2 - Finc), 1e-6 * np.abs(Finc))
    assert sc2[0] == pytest.approx(1e-6 * Finc[0])
    assert Request(s, 100.0, req.Pinc, Ft2, sc2, None, None, None).e0 == pytest.approx(1.0)


def test_zero_distance_request_is_rejected(tmp_path):
    s, req, Finc, _, _ = _request(tmp_path)
    sc = 1e-6 * np.abs(Finc)
    with pytest.raises(ValueError, match='zero distance'):
        Request(s, 100.0, req.Pinc, Finc.copy(), sc, None, None, None)


def test_gap_range_monotonicity_and_feasibility(tmp_path):
    s, req, *_ = _request(tmp_path); rng = np.random.default_rng(0)
    Gp, _ = req.gap(1e9, 1.0, rng)
    vals = {}
    for R in (1.0, 5.0, 20.0, 1e9):
        for rho in (0.01, 0.05, 0.2, 1.0):
            G, P = req.gap(R, rho, rng); vals[(R, rho)] = G
            lo, hi = req.band(rho)
            assert 0.0 <= G <= 1.0 and G >= Gp - 1e-6
            assert abs(P.sum() - 100.0) < 1e-5 and np.all(P >= lo - 1e-6) and np.all(P <= hi + 1e-6)
            if R < 1e8:
                assert np.abs(P - req.Pinc).sum() <= R + 1e-5
    Rs, rhos = (1.0, 5.0, 20.0, 1e9), (0.01, 0.05, 0.2, 1.0)
    for i in range(3):
        for rho in rhos:
            assert vals[(Rs[i + 1], rho)] <= vals[(Rs[i], rho)] + 1e-6
        for R in Rs:
            assert vals[(R, rhos[i + 1])] <= vals[(R, rhos[i])] + 1e-6


def test_empty_authorization_returns_the_incumbent(tmp_path):
    s, req, *_ = _request(tmp_path); rng = np.random.default_rng(1)
    G, P = req.gap(0.0, 1.0, rng)                            # zero budget: only the incumbent is admissible
    assert G == pytest.approx(1.0, abs=1e-6) and np.allclose(P, req.Pinc, atol=1e-5)
    G, P = req.gap(1e9, 0.0, rng)                            # zero band
    assert G == pytest.approx(1.0, abs=1e-6)


def test_contradictory_constraints_fall_back_to_incumbent_and_are_counted():
    d = pd.DataFrame(dict(Pmin=[0.0, 0.0], Pmax=[100.0, 100.0], a=[0.01, 0.02], b=[1, 1], c=[0, 0], e=[0, 0], f=[0, 0],
                          alpha=[1, 1], beta=[0.0, 0.0], gamma=[1e-3, 2e-3], zeta=[0, 0], **{'lambda': [0, 0]}))
    ramp = pd.DataFrame(dict(UR_MW_per_h=[10.0, 10.0], DR_MW_per_h=[10.0, 10.0]))
    s = OpsSystem(d, None, ramp); Pinc = np.array([50.0, 50.0]); Finc = s.F(Pinc)
    Ft = Finc * np.array([1.01, 0.95]); sc = np.abs(Ft - Finc)
    req = OpsRequest(s, 100.0, Pinc, Ft, sc, [])
    # neighbour hour fixed far away: |P - P_other| <= 10 MW cannot hold, so no admissible point exists
    req.extra_cons = ramp_cons_vec(s, np.array([90.0, 90.0]))   # needs P >= 80 MW per unit, sum >= 160 > 100
    G, P = req.gap(1e9, s.span, np.random.default_rng(0), k=2)
    assert G == 1.0 and np.array_equal(P, Pinc) and req.fallbacks == 1


def test_ramp_band_and_uniform_band():
    coef, B, ramp, _ = load()
    s = OpsSystem(coef, B, ramp)
    assert np.allclose(s.band_ramp(30), np.minimum(ramp.UR_MW_per_h, ramp.DR_MW_per_h) / 2)
    assert np.allclose(s.band_uniform(0.1), 0.1 * (coef.Pmax - coef.Pmin))
    req = OpsRequest(s, 1500.0, s.rand_feasible(1500.0, np.random.default_rng(0)), np.array([1.0, 1.0]), np.array([1.0, 1.0]), [])
    lo, hi = req.box(s.band_ramp(60))
    assert np.all(lo >= s.lo) and np.all(hi <= s.hi)         # bands are clipped at the generator limits


def load():
    from eeldops.data_loader import load_eeld10
    return load_eeld10()
