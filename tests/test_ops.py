"""Checks of the operational model: Kron losses and ramp-derived bands."""
import numpy as np
from eeldops.data_loader import load_eeld10
from eeldops.model import OpsSystem, OpsRequest


def setup():
    coef, B, ramp, dem = load_eeld10(); return OpsSystem(coef, B, ramp), dem


def test_loss_jacobian():
    s, _ = setup(); P = np.linspace(150, 100, 10); h = 1e-4
    num = np.array([(s.residual(P + h * e, 1500) - s.residual(P - h * e, 1500)) / (2 * h) for e in np.eye(10)])
    assert np.allclose(num, s.residual_jac(P), atol=1e-6)


def test_balance_with_losses():
    s, dem = setup(); rng = np.random.default_rng(0)
    P = s.rand_feasible(1500.0, rng); assert abs(P.sum() - 1500 - s.loss(P)) < 1e-6


def test_ramp_band_scaling():
    s, _ = setup(); assert np.allclose(s.band_ramp(60), [80, 80, 80, 50, 50, 50, 30, 30, 30, 30]); assert np.allclose(s.band_ramp(15), s.band_ramp(60) / 4)


def test_gap_monotone_in_tau():
    s, _ = setup(); rng = np.random.default_rng(1); PD = 1500.0
    A = s.anchors(PD, rng, K=10, sweep=3); Pinc = s.incumbent(PD, 0.5, A, K=10)
    Ft = np.array([s.C(A['Pe']), 0.95 * s.E(Pinc)]); sc = np.abs(Ft - s.F(Pinc))
    req = OpsRequest(s, PD, Pinc, Ft, sc, [A['Pe']])
    g = [req.gap(1e9, s.band_ramp(t), rng, k=2)[0] for t in (10, 30, 60)]
    assert g[0] >= g[1] - 1e-6 >= g[2] - 2e-6 and 0 <= g[-1] <= 1

