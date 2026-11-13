"""Documented 30-bus network variant with DC line limits.

Sources and assumptions (all stated in the manuscript, Appendix F):
  - Topology, branch reactances and bus loads: MATPOWER ``case_ieee30`` (IEEE 30-bus test case, 283.4 MW).
  - Line ratings: ``rateA`` of MATPOWER ``case30`` (same 41 branches in the same order), which takes the
    line limits from Alsac and Stott (1974). Used as MW limits of the DC flows.
  - Cost and emission coefficients: Table I of Abido (2003), converted from p.u. (100 MVA base) to MW
    (data/eeld30_coeffs.csv). No valve-point terms.
  - Generator buses 1, 2, 5, 8, 11, 13 for G1 to G6, in this order (standard IEEE 30-bus generator buses).
    Abido's Fig. 3 uses a different bus numbering, so this mapping is an assumption.
  - Generator limits 5 to 150 MW for every unit. Abido gives the limits only through his references, so the
    values are an assumption. The lossless anchors reproduce his Table II (checked in network30.py).
  - DC power flow, lossless, slack at bus 1. Flows are f = H (G P - load), with H the PTDF matrix.
"""
from __future__ import annotations
import re
import numpy as np
from .model import System
from .paths import DATA, EXTERNAL

MP = EXTERNAL / 'matpower'
GEN_BUS = (1, 2, 5, 8, 11, 13)


def parse_matpower(path):
    t = open(path).read(); out = {}
    for name in ('bus', 'gen', 'branch'):
        m = re.search(r'mpc\.%s\s*=\s*\[(.*?)\];' % name, t, re.S)
        rows = []
        for line in m.group(1).split('\n'):
            line = line.split('%')[0].strip().rstrip(';')
            if line:
                rows.append([float(x) for x in line.split()])
        out[name] = np.array(rows)
    return out


def ptdf(branch, nb, slack=0):
    f = branch[:, 0].astype(int) - 1; t = branch[:, 1].astype(int) - 1; x = branch[:, 3]
    nl = len(branch)
    Cft = np.zeros((nl, nb)); Cft[np.arange(nl), f] = 1.0; Cft[np.arange(nl), t] = -1.0
    Bf = np.diag(1.0 / x) @ Cft; Bbus = Cft.T @ Bf
    keep = [i for i in range(nb) if i != slack]
    H = np.zeros((nl, nb)); H[:, keep] = Bf[:, keep] @ np.linalg.inv(Bbus[np.ix_(keep, keep)])
    return H


class Network:
    """DC network of the 30-bus variant at a load multiplier."""

    def __init__(self, mult=1.0):
        net = parse_matpower(MP / 'case_ieee30.m'); rat = parse_matpower(MP / 'case30.m')
        if not np.array_equal(net['branch'][:, :2], rat['branch'][:, :2]):
            raise ValueError('case30 and case_ieee30 branch lists differ')
        self.branch = net['branch'][:, :2].astype(int)
        nb = len(net['bus'])
        self.load = mult * net['bus'][:, 2]
        self.PD = float(self.load.sum())
        self.rate = rat['branch'][:, 5].astype(float)
        H = ptdf(net['branch'], nb)
        G = np.zeros((nb, len(GEN_BUS))); G[np.array(GEN_BUS) - 1, np.arange(len(GEN_BUS))] = 1.0
        self.Hg = H @ G                      # flow sensitivity to the unit outputs (MW per MW)
        self.f0 = -H @ self.load             # flow from the loads alone

    def flows(self, P):
        return self.Hg @ np.asarray(P, float) + self.f0

    def binding(self, P, tol=1e-3):
        f = self.flows(P)
        return [i for i in range(len(f)) if abs(f[i]) >= self.rate[i] - tol]

    def label(self, i):
        return f'{self.branch[i, 0]}-{self.branch[i, 1]}'


class NetSystem(System):
    """System of data/eeld30_coeffs.csv with the DC line limits of ``Network`` as extra constraints.

    ``limits=False`` gives the same units on a copper plate (no line limits), for the paired comparison.
    """

    def __init__(self, mult=1.0, limits=True):
        super().__init__(str(DATA / 'eeld30_coeffs.csv'))
        self.net = Network(mult)
        self.limits = limits
        Hg, f0, r = self.net.Hg, self.net.f0, self.net.rate
        # rate - f >= 0 and rate + f >= 0 for every branch, as two vector-valued SLSQP constraints
        self.extra_cons = () if not limits else (
            {'type': 'ineq', 'fun': lambda P: r - (Hg @ P + f0), 'jac': lambda P: -Hg},
            {'type': 'ineq', 'fun': lambda P: r + (Hg @ P + f0), 'jac': lambda P: Hg},
        )

    def line_ok(self, P, tol=1e-6):
        return (not self.limits) or bool(np.all(np.abs(self.net.flows(P)) <= self.net.rate + tol))
