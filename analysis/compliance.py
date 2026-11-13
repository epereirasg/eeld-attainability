"""Compliance counterfactuals and the reachable/unreachable stratification of the 10-unit system.

Reads results/raw/additional/compliance/compliance_p*.csv (experiments/additional/compliance_cf.py), the stored
campaign and the dense allowance grid, and writes results/macros/numbers_compliance.tex and
results/processed/numbers_compliance.json.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROC, MAC, load_case, boot_ci, paths

ADD = paths.RAW_ADD
N = {}
REAL = [(R, rho) for R in (25, 50, 100) for rho in (0.05, 0.10, 0.20)]
EPS_MAIN = 0.01


def f(x, d=2):
    return 'nan' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


def q(x, d=3, p=(0.25, 0.5, 0.75)):
    x = pd.Series(x).dropna()
    return [f(v, d) for v in x.quantile(list(p))] if len(x) else ['nan'] * len(p)


def slack():
    """Reachable requests whose cheapest compliant dispatch leaves the cap slack (nonconvex cost)."""
    for case, T in (('10', 'Ten'), ('40', 'Forty')):
        R, _ = load_case(case)
        V = {(v['hour'], v['w'], v['cut']): v for f in sorted(glob.glob(f'{paths.RAW_ATTAIN}/vectors_e{case}*.jsonl')) for v in map(json.loads, open(f))}
        u = R[R.unique & R.reachable.astype(bool)]
        sl = np.array([(r.E_tar - V[(r.hour, r.w, r.cut)]['Ft'][1]) / r.E_tar for r in u.itertuples()])
        N[f'Slack{T}'] = str(int((sl > 1e-5).sum())); N[f'Slack{T}N'] = str(len(u))
        N[f'Slack{T}Max'] = f(100 * sl.max(), 1) if len(sl) else 'nan'


def forty():
    fs = sorted(glob.glob(f'{ADD}/compliance/compliance_e40_p*.csv'))
    if not fs:
        return
    R, _ = load_case('40'); Ru = R[R.unique]
    key = ['hour', 'w', 'cut']
    C = pd.concat([pd.read_csv(x, dtype={'S_sets_E': str}) for x in fs])
    M = C[C.eps_E == EPS_MAIN].merge(Ru[key], on=key)
    need_e = M.need_E.astype(bool); need_d = M.G_ref > M.G_phys + 0.05
    E = M[need_e]
    N['CompNeedForty'] = str(int(need_e.sum())); N['CompNeedBothForty'] = str(int((need_e & need_d).sum()))
    N['CompRhoMedForty'] = f(E.rho_star_E.median(), 3); N['CompRMedForty'] = f(E.R_star_E.median(), 0)
    N['CompSetLeTwoForty'] = f(100 * (E.S_size_E <= 2).mean(), 0) if len(E) else 'nan'
    N['CompNForty'] = str(len(M))


def units_text(sets):
    """'4-9-10;8-9-10' -> 'units 4, 9 and 10, or units 8, 9 and 10'."""
    out = []
    for s in [x for x in str(sets).replace(';', ',').split(',') if x and x != 'nan']:
        u = s.split('-')
        out.append(('unit ' + u[0]) if len(u) == 1 else 'units ' + ', '.join(u[:-1]) + ' and ' + u[-1])
    return ', or '.join(out[:3]) + (f' (or {len(out) - 3} more sets)' if len(out) > 3 else '')


def box():
    fbox = ADD / 'explanation_box.json'
    if not fbox.exists():
        return
    B = json.load(open(fbox))
    N['BoxHour'], N['BoxW'], N['BoxCut'] = str(B['request'][0]), str(B['request'][1]), f"{100 * B['request'][2]:.0f}"
    N['BoxCostPct'] = f(B['cost_increase_pct'], 1); N['BoxShiftMW'] = f(B['mw_shifted_full'], 0); N['BoxMoveMW'] = f(2 * B['mw_shifted_full'], 0)
    N['BoxEtaRef'] = f(100 * B['eta_b_ref'], 0); N['BoxRemoved'] = f(100 * (1 - B['G_ref']), 0); N['BoxGref'] = f(B['G_ref'], 2)
    N['BoxGphys'] = f(B['G_phys'], 2); N['BoxBudgetFreeGain'] = f(B['G_ref'] - B['G_Rfree'], 3)
    N['BoxRho'] = f(B['rho_star'], 2); N['BoxR'] = f(B['R_star'], 0); N['BoxRShift'] = f(B['R_star'] / 2, 0); N['BoxTau'] = f(B['tau_star'], 0)
    N['BoxSets'] = units_text(';'.join(B['minimal_sets']))
    d = B.get('deliver', {})
    if d:
        N['BoxDelRho'] = f(100 * d['rho_star']['share_of_cut'], 0); N['BoxDelR'] = f(100 * d['R_star']['share_of_cut'], 0)
        N['BoxAboveRho'] = f(d['rho_star']['above_cap_pct'], 2); N['BoxAboveR'] = f(d['R_star']['above_cap_pct'], 2)
        N['BoxResRho'] = f(100 * (1 - d['rho_star']['share_of_cut']), 0); N['BoxResR'] = f(100 * (1 - d['R_star']['share_of_cut']), 0)   # residual in percent of the cut
    c = B.get('compliance', {}).get(str(EPS_MAIN), {})
    if c and c.get('need'):
        N['BoxRhoE'] = f(c['rho_star_E'], 2); N['BoxRE'] = f(c['R_star_E'], 0); N['BoxREShift'] = f(c['R_star_E'] / 2, 0)
        N['BoxTauE'] = f(c['tau_star_E'], 0); N['BoxSetsE'] = units_text(c.get('S_sets_E', ''))
        N['BoxSetSizeE'] = str(int(c['S_size_E'])) if c.get('S_size_E') == c.get('S_size_E') else 'nan'


def main():
    slack(); box()
    R, G = load_case('10')
    Ru = R[R.unique].copy()
    C = pd.concat([pd.read_csv(x, dtype={'S_sets_E': str}) for x in sorted(glob.glob(f'{ADD}/compliance/compliance_p*.csv'))])
    key = ['hour', 'w', 'cut']
    M = C[C.eps_E == EPS_MAIN].merge(Ru[key + ['reachable']], on=key, how='inner', suffixes=('', '_r'))
    assert len(M) == len(Ru), (len(M), len(Ru))
    N['CompEps'] = f(EPS_MAIN, 2)
    N['CompN'] = str(len(M))
    N['CompEtaMaxDiff'] = f'{(M.eta_phys - M.eta_phys_stored).abs().max():.1e}'
    need_d = M.G_ref > M.G_phys + 0.05
    need_e = M.need_E.astype(bool)
    N['CompNeed'] = str(int(need_e.sum())); N['DistNeed'] = str(int(need_d.sum()))
    N['CompNeedBoth'] = str(int((need_e & need_d).sum())); N['CompNeedEOnly'] = str(int((need_e & ~need_d).sum()))
    N['CompNeedDistOnly'] = str(int((need_d & ~need_e).sum())); N['CompNeedCompOnly'] = str(int((need_e & ~need_d).sum()))
    E = M[need_e]
    for nm, col, d in (('Rho', 'rho_star_E', 3), ('R', 'R_star_E', 0), ('Tau', 'tau_star_E', 0)):
        a, b, c = q(E[col], d)
        N[f'Comp{nm}Qa'], N[f'Comp{nm}Med'], N[f'Comp{nm}Qc'] = a, b, c
        N[f'Comp{nm}NaN'] = str(int(E[col].isna().sum()))
    N['CompTauLeSixty'] = f(100 * (E.tau_star_E <= 60).mean(), 0)
    s = E.S_size_E
    for k in (0, 1, 2, 3, 4):
        N[f'CompSet{"ZOTHF"[k]}'] = str(int((s == k).sum()))
    N['CompSetLeTwo'] = f(100 * (s <= 2).mean(), 0)
    lo, hi = boot_ci((s <= 2).astype(float).values)
    N['CompSetLeTwoLo'], N['CompSetLeTwoHi'] = f(100 * lo, 0), f(100 * hi, 0)
    N['CompSetMulti'] = f(100 * (E.S_n_minimal_E >= 2).mean(), 0)
    # certificate of the compliance counterfactuals (feasible point at the value, SCIP bound one step below)
    for nm in ('rho', 'R'):
        ok = E[f'R_star_E' if nm == 'R' else 'rho_star_E'].notna()
        N[f'CompCertSuff{nm.capitalize()}'] = str(int(E.loc[ok, f'cert_suff_{nm}'].astype(bool).sum()))
        N[f'CompCertMin{nm.capitalize()}'] = str(int(E.loc[ok, f'cert_min_{nm}'].astype(bool).sum()))
        N[f'CompCertN{nm.capitalize()}'] = str(int(ok.sum()))
    # paired comparison with the distance counterfactuals on the requests that need both
    B = M[need_e & need_d]
    N['CompBothRhoLower'] = str(int((B.rho_star_E < B.rho_star - 1e-9).sum()))
    N['CompBothRhoHigher'] = str(int((B.rho_star_E > B.rho_star + 1e-9).sum()))
    N['CompBothRLower'] = str(int((B.R_star_E < B.R_star - 1e-9).sum()))
    N['CompBothRHigher'] = str(int((B.R_star_E > B.R_star + 1e-9).sum()))
    N['CompBothRhoRatio'] = f(np.nanmedian(B.rho_star_E / B.rho_star), 2)
    N['CompBothRRatio'] = f(np.nanmedian(B.R_star_E / B.R_star), 2)
    # what the distance counterfactual delivers in the units of the request
    D = M[need_d]
    thr = (1 - EPS_MAIN) * D.eta_phys
    N['DistDeliverRho'] = f(100 * (D.eta_at_rho_star >= thr - 1e-6).mean(), 0)
    N['DistDeliverR'] = f(100 * (D.eta_at_R_star >= thr - 1e-6).mean(), 0)
    N['DistEtaShareRhoMed'] = f(100 * np.nanmedian(D.eta_at_rho_star / D.eta_phys), 0)
    N['DistEtaShareRMed'] = f(100 * np.nanmedian(D.eta_at_R_star / D.eta_phys), 0)
    # sensitivity to eps_E
    for e, tag in ((0.0, 'Zero'), (0.05, 'Five')):
        X = C[C.eps_E == e].merge(Ru[key], on=key)
        Y = X[X.need_E.astype(bool)]
        N[f'CompNeed{tag}'] = str(len(Y))
        N[f'CompRhoMed{tag}'] = f(Y.rho_star_E.median(), 3); N[f'CompRMed{tag}'] = f(Y.R_star_E.median(), 0)
        N[f'CompTauMed{tag}'] = f(Y.tau_star_E.median(), 0)
    # matched tolerances. For a reachable request d(P_inc) = sqrt(2), so the gap tolerance eps admits an emission residual
    # of up to eps*sqrt(2) of the requested cut. The compliance run with eps_E = sqrt(2) eps (compliance_match_p*.csv)
    # matches it: eps = 0.02 with eps_E = 0.0283 and eps = 0.05 with eps_E = 0.0707. Each pair compares the band of
    # both families on the requests that need both.
    fm = sorted(glob.glob(f'{ADD}/compliance/compliance_match_p*.csv'))
    if fm:
        Cm = pd.concat([pd.read_csv(x) for x in fm])
        Ag = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f'{ADD}/allowance_grid/requests_p*.csv'))])
        G2 = Ag[key + ['need_0.02', 'rho_star_0.02']].rename(columns={'need_0.02': 'need_d2', 'rho_star_0.02': 'rho_d2'})
        for eg, ee, tag in ((0.05, 0.0707, 'Five'), (0.02, 0.0283, 'Two')):
            X = Cm[(Cm.eps_E - ee).abs() < 1e-6].merge(
                Ru[key + ['G_ref', 'G_phys', 'rho_star']].rename(columns={'G_ref': 'Gr', 'G_phys': 'Gp', 'rho_star': 'rs'}), on=key).merge(G2, on=key, how='left')
            nd = (X.Gr > X.Gp + 0.05) if eg == 0.05 else X.need_d2.astype(bool)
            rd = X.rs if eg == 0.05 else X.rho_d2
            both = nd & X.need_E.astype(bool); Y = X[both]
            N[f'Match{tag}N'] = str(int(both.sum()))
            N[f'Match{tag}Wider'] = str(int((Y.rho_star_E > rd[both] + 1e-9).sum()))
            N[f'Match{tag}Narrower'] = str(int((Y.rho_star_E < rd[both] - 1e-9).sum()))
            N[f'Match{tag}MedE'] = f(Y.rho_star_E.median(), 3); N[f'Match{tag}MedD'] = f(rd[both].median(), 3)
    # calibrated tolerances (Proposition 1 (inclusion)). Each request takes eps_E = eps d(P_inc) / eta_E, written by
    # compliance_cf.py --calibrated. The proposition gives rho*_E <= rho* and R*_E <= R* up to one bisection step, and a
    # request that needs a compliance counterfactual then also needs a gap counterfactual. The check counts inversions by
    # stratum (reachable with an active cap, reachable with a slack cap, unreachable).
    fc = sorted(glob.glob(f'{ADD}/compliance/compliance_calibrated_p*.csv'))
    if fc:
        Cc = pd.concat([pd.read_csv(x) for x in fc])
        Ag = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f'{ADD}/allowance_grid/requests_p*.csv'))])
        G2 = Ag[key + ['need_0.02', 'rho_star_0.02', 'R_star_0.02']]
        V = {(v['hour'], v['w'], v['cut']): v for x in sorted(glob.glob(f'{paths.RAW_ATTAIN}/vectors_e10*.jsonl')) for v in map(json.loads, open(x))}
        slack_cap = {k: (r.E_tar - V[k]['Ft'][1]) / r.E_tar > 1e-5 for r in Ru.itertuples() for k in [(r.hour, r.w, r.cut)]}
        stp = {'rho': 0.95 / 256, 'R': 1000 / 256}
        for eg, tag in ((0.05, 'Five'), (0.02, 'Two')):
            X = Cc[(Cc.eps_gap - eg).abs() < 1e-9].merge(Ru[key + ['G_ref', 'G_phys', 'rho_star', 'R_star', 'reachable']].rename(
                columns={'G_ref': 'Gr', 'G_phys': 'Gp', 'rho_star': 'rs', 'R_star': 'Rs', 'reachable': 'rch'}), on=key).merge(G2, on=key, how='left')
            nd = (X.Gr > X.Gp + 0.05) if eg == 0.05 else X['need_0.02'].astype(bool)
            rd = X.rs if eg == 0.05 else X['rho_star_0.02']; Rd = X.Rs if eg == 0.05 else X['R_star_0.02']
            ne = X.need_E.astype(bool); both = nd & ne
            inv = both & ((X.rho_star_E > rd + stp['rho'] + 1e-9) | (X.R_star_E > Rd + stp['R'] + 1e-9))
            N[f'Cal{tag}Both'] = str(int(both.sum())); N[f'Cal{tag}Inv'] = str(int(inv.sum()))
            N[f'Cal{tag}OnlyE'] = str(int((ne & ~nd).sum())); N[f'Cal{tag}OnlyD'] = str(int((nd & ~ne).sum()))
            N[f'Cal{tag}RhoNarrower'] = str(int((both & (X.rho_star_E < rd - 1e-9)).sum()))
            N[f'Cal{tag}RhoWider'] = str(int((both & (X.rho_star_E > rd + 1e-9)).sum()))
            N[f'Cal{tag}RWider'] = str(int((both & (X.R_star_E > Rd + 1e-9)).sum()))
            N[f'Cal{tag}MedE'] = f(X.rho_star_E[both].median(), 3); N[f'Cal{tag}MedD'] = f(rd[both].median(), 3)
            sl = np.array([slack_cap[(r.hour, r.w, r.cut)] for r in X.itertuples()]); rc = X.rch.astype(bool).values
            for lab, m in (('Active', rc & ~sl), ('Slack', rc & sl), ('Unreach', ~rc)):
                N[f'Cal{tag}{lab}Both'] = str(int((both & m).sum()))
            u = X[~X.rch.astype(bool)]
            N[f'Cal{tag}EpsUnreachMed'] = f(u.eps_E.median(), 3)
            N[f'Cal{tag}EpsUnreachMin'] = f(u.eps_E.min(), 3)
            N[f'Cal{tag}Vacuous'] = str(int((u.eps_E >= 1).sum()))      # eta_E <= sqrt(2) eps, the condition holds for every allowance
        # with the reachable calibration sqrt(2) eps applied to every request, every inversion lies in the unreachable stratum
        Xm = Cm[(Cm.eps_E - 0.0707).abs() < 1e-6].merge(Ru[key + ['G_ref', 'G_phys', 'rho_star', 'reachable']], on=key, suffixes=('', '_c'))
        bm = (Xm.G_ref_c > Xm.G_phys_c + 0.05) & Xm.need_E.astype(bool)
        wid = bm & (Xm.rho_star_E > Xm.rho_star_c + 1e-9)
        N['MatchFiveWiderUnreach'] = str(int((wid & ~Xm.reachable_c.astype(bool)).sum()))
    # stratification by reachability
    A = pd.read_csv(sorted(glob.glob(f'{ADD}/allowance_grid/grid_p*.csv'))[0]) if glob.glob(f'{ADD}/allowance_grid/grid_p*.csv') else None
    Ag = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f'{ADD}/allowance_grid/grid_p*.csv'))]) if A is not None else None
    rows = []
    for lab, mask in (('Reach', M.reachable_r.astype(bool) if 'reachable_r' in M else M.reachable.astype(bool)),):
        pass
    reach_flag = M.reachable.astype(bool).values
    for lab, sel in (('Reach', reach_flag), ('Unreach', ~reach_flag)):
        X = M[sel]; Rx = Ru.merge(X[key], on=key)
        N[f'Str{lab}N'] = str(len(X))
        Gx = G[G.unique & (G.kind == 'grid')].merge(X[key], on=key)
        real = Gx[Gx.apply(lambda r: (r.R, r.rho) in REAL, axis=1)]
        ga = (real.G - real.G_phys).groupby([real.hour, real.w, real.cut]).mean()
        N[f'Str{lab}Allow'] = f(100 * (ga >= 0.10).mean(), 1)
        lo, hi = boot_ci((ga >= 0.10).astype(float).values)
        N[f'Str{lab}AllowLo'], N[f'Str{lab}AllowHi'] = f(100 * lo, 1), f(100 * hi, 1)
        N[f'Str{lab}AllowMean'] = f(ga.mean(), 2)
        dd = X[X.G_ref > X.G_phys + 0.05]
        N[f'Str{lab}NeedD'] = str(len(dd))
        N[f'Str{lab}RhoMed'] = f(dd.rho_star.median(), 3); N[f'Str{lab}RMed'] = f(dd.R_star.median(), 0)
        ee = X[X.need_E.astype(bool)]
        N[f'Str{lab}NeedE'] = str(len(ee))
        N[f'Str{lab}RhoEMed'] = f(ee.rho_star_E.median(), 3); N[f'Str{lab}REMed'] = f(ee.R_star_E.median(), 0)
        N[f'Str{lab}EtaPhysMed'] = f(100 * X.eta_phys.median(), 0)
        N[f'Str{lab}GphysMed'] = f(X.G_phys.median(), 2)
        if Ag is not None:
            Ax = Ag.merge(X[key], on=key)
            Ax = Ax[Ax.apply(lambda r: (r.R, r.rho) in REAL, axis=1)]
            eb = Ax.groupby(['hour', 'w', 'cut']).eta_b.median() if 'eta_b' in Ax else pd.Series(dtype=float)
            N[f'Str{lab}EtaBMed'] = f(100 * eb.median(), 0) if len(eb) else 'nan'
        N[f'Str{lab}EtaRefMed'] = f(100 * X.eta_ref.median(), 0)
    forty()
    os.makedirs(MAC, exist_ok=True)
    with open(f'{MAC}/numbers_compliance.tex', 'w') as fh:
        for k, v in sorted(N.items()):
            fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(N, open(f'{PROC}/numbers_compliance.json', 'w'), indent=1)
    M.to_csv(f'{PROC}/compliance_requests.csv', index=False)
    print(json.dumps(N, indent=1))


if __name__ == '__main__':
    main()
