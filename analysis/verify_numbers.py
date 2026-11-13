"""Independent recomputation of manuscript numbers from the raw CSV files.

Written separately from the analysis scripts (no shared helper code, own formulas) so that a bug in the
analysis pipeline shows up as a mismatch here. Every check compares a value recomputed from
results/raw with the macro value written to results/processed/numbers_*.json, within a tolerance
matched to the printed precision. The script exits with status 1 if any check fails or a macro is
missing, so it can gate the paper build.

This is a numerical cross-check of the reporting chain (raw outputs -> macros). It does not test
whether the optimization results themselves are correct; see tests/ and the solver certificate.
"""
import glob, json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from eeldattain.paths import RAW, RAW_ATTAIN, RAW_CERT, RAW_ROBUST, RAW_DFO, RAW_OPS, RAW_SUPP, PROCESSED

N = {}
for f in sorted(glob.glob(f'{PROCESSED}/numbers_*.json')):
    N.update(json.load(open(f)))
checks = []


def check(name, value, tol):
    if name not in N:
        checks.append((name, value, 'missing', False)); return
    rep = float(str(N[name]).replace('{,}', '').replace(',', ''))
    checks.append((name, value, rep, bool(abs(rep - float(value)) <= tol)))


def cat(pattern, **kw):
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(f, **kw) for f in fs], ignore_index=True)


def boot_share(x, B=10000, seed=0):        # same resampling rule as the analysis (README.md, seeds)
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    s = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return np.quantile(s, [0.025, 0.975])


# ------------------------------------------------------------------ EELD10 attainability
req = cat(f'{RAW_ATTAIN}/requests_e10*.csv', dtype={'S_units': str})
grid = cat(f'{RAW_ATTAIN}/grid_e10*.csv')
u = req[req.duplicate_of_w.isna()].copy()
check('NumReqTen', len(req), 0); check('NumReqTenUnique', len(u), 0)
hrs = u.groupby('hour').agg(W=('front_width', 'first'), eta=('eta_E', 'mean'), PD=('PD', 'first'))
check('SpearmanTen', spearmanr(hrs.W, hrs.eta)[0], 0.005)
check('WidthMinTen', 100 * hrs.W.min(), 0.05); check('WidthMaxTen', 100 * hrs.W.max(), 0.05)
check('WidthMinPDTen', hrs.PD[hrs.W.idxmin()], 0.5); check('WidthMaxPDTen', hrs.PD[hrs.W.idxmax()], 0.5)
for c, k in [(0.025, 'A'), (0.05, 'B'), (0.10, 'C')]:
    x = u[u.cut == c]
    check(f'Reach{k}Ten', 100 * x.reachable.mean(), 0.5); check(f'Eta{k}Ten', x.eta_E.mean(), 0.005)
    check(f'CostTar{k}Ten', x[x.reachable].dC_target_pct.median(), 0.005)
    check(f'MwFull{k}Ten', x[x.reachable].mw_full.median(), 0.5)
g = grid[(grid.kind == 'grid') & grid.R.isin([25, 50, 100]) & grid.rho.isin([0.05, 0.1, 0.2])]
g = g.merge(u[['hour', 'w', 'cut']], on=['hour', 'w', 'cut'])
m = (g.G - g.G_phys).clip(lower=0).groupby([g.hour, g.w, g.cut]).mean()
check('AllowShareTen', 100 * (m >= 0.1).mean(), 0.05); check('AllowMeanTen', m.mean(), 0.005)
check('AllowMedianTen', m.median(), 0.005)
lo, hi = boot_share((m >= 0.1).astype(float).values)
check('AllowShareLoTen', 100 * lo, 0.05); check('AllowShareHiTen', 100 * hi, 0.05)
gg = grid[(grid.kind == 'grid')].merge(u[['hour', 'w', 'cut']], on=['hour', 'w', 'cut'])
pm = gg.groupby(['R', 'rho']).G.mean()
check('SaturationTen', max(pm[(100, r)] - pm[(250, r)] for r in (0.02, 0.05, 0.1)), 0.0005)
need = u[u.G_ref > u.G_phys + 0.05]
tot = need.G_ref - need.G_phys
check('NeedCFTen', len(need), 0)
check('ReliefRTen', 100 * ((need.G_ref - need.G_Rfree) / tot).mean(), 0.5)
check('ReliefRhoTen', 100 * ((need.G_ref - need.G_rhofree) / tot).mean(), 0.5)
check('JointTen', 100 * (((need.G_ref - need.G_Rfree) / tot < 0.7) & ((need.G_ref - need.G_rhofree) / tot < 0.7)).mean(), 0.5)
need = need.copy()
z0 = need.G_Rfree <= need.G_phys + 0.05            # rho0 already suffices: no wider band needed
need.loc[z0, 'rho_star'] = need.loc[z0, 'rho0']; check('RhoStarAtZeroTen', int(z0.sum()), 0)
for col, nm, tol in (('rho_star', 'RhoStar', 0.0005), ('R_star', 'RStar', 0.5)):
    q = need[col].quantile([0.25, 0.5, 0.75])
    check(f'{nm}MedTen', q[0.5], tol); check(f'{nm}QaTen', q[0.25], tol); check(f'{nm}QcTen', q[0.75], tol)
s = need.dropna(subset=['S_size'])
check('NeedSetTen', len(s), 0); check('SetLeTwoTen', 100 * (s.S_size <= 2).mean(), 0.5)
for k, nm in zip((1, 2, 3, 4), 'ABCD'):
    check(f'SetSize{nm}Ten', int((s.S_size == k).sum()), 0)
val = s[[(w, c) in {(0.25, 0.025), (0.5, 0.05), (0.75, 0.10)} for w, c in zip(s.w, s.cut)]]
check('SetValidated', len(val), 0)
check('SetMissed', int((val.S_exact_check < val.S_size).sum()), 0)
check('EffLossMedTen', u.ref_eff_loss_pct.median(), 0.005); check('EffLossQcTen', u.ref_eff_loss_pct.quantile(0.9), 0.005)
check('RefMwTen', u.ref_mw.median(), 0.05); check('RefUnitsTen', u.ref_units_moved.median(), 0.5)
check('CloudRefMadTen', (u.G_ref - u.G_ref_cloud).abs().median(), 0.0005)
check('CloudAltMadTen', (u.G_alt - u.G_alt_cloud).abs().median(), 0.0005)
check('CloudRefSpTen', spearmanr(u.G_ref, u.G_ref_cloud)[0], 0.005)

# ------------------------------------------------------------------ EELD40 attainability
q4 = cat(f'{RAW_ATTAIN}/requests_e40*.csv', dtype={'S_units': str}); q4u = q4[q4.duplicate_of_w.isna()]
g4 = cat(f'{RAW_ATTAIN}/grid_e40*.csv')
check('NumReqForty', len(q4), 0); check('NumReqFortyUnique', len(q4u), 0)
h4 = q4u.groupby('hour').agg(W=('front_width', 'first'), eta=('eta_E', 'mean'))
check('SpearmanForty', spearmanr(h4.W, h4.eta)[0], 0.005)
g4r = g4[(g4.kind == 'grid') & g4.R.isin([100, 250, 500]) & g4.rho.isin([0.05, 0.1, 0.2])]
m4 = (g4r.G - g4r.G_phys).clip(lower=0).groupby([g4r.hour, g4r.w, g4r.cut]).mean()
check('AllowShareForty', 100 * (m4 >= 0.1).mean(), 0.05)
n4 = q4u[q4u.G_ref > q4u.G_phys + 0.05].copy()
z4 = n4.G_Rfree <= n4.G_phys + 0.05; n4.loc[z4, 'rho_star'] = n4.loc[z4, 'rho0']
check('NeedCFForty', len(n4), 0)
check('RhoStarMedForty', n4.rho_star.median(), 0.0005); check('RStarMedForty', n4.R_star.median(), 0.5)
s4 = n4.dropna(subset=['S_size']); check('SetLeTwoForty', 100 * (s4.S_size <= 2).mean(), 0.5)
check('EffLossMedForty', q4u.ref_eff_loss_pct.median(), 0.005)

# ------------------------------------------------------------------ method comparison
# stored blocks A, B and C (hour 7), with the five configurations of the joint rerun (raw_C5, raw_all5) replacing their
# stored rows; an instance repeated under two weights (same hash) enters once
rb = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{RAW_ROBUST}/raw_*.csv')) if 'C5' not in f], ignore_index=True)
df = pd.read_csv(f'{RAW_DFO}/raw_all.csv')
if glob.glob(f'{RAW_ROBUST}/raw_C5*.csv'):
    r5 = cat(f'{RAW_ROBUST}/raw_C5*.csv'); rb = pd.concat([rb[~rb.config.isin(r5.config.unique())], r5], ignore_index=True)
    if glob.glob(f'{RAW_DFO}/raw_all5*.csv'):
        d5 = cat(f'{RAW_DFO}/raw_all5*.csv'); df = pd.concat([df[~df.config.isin(d5.config.unique())], d5], ignore_index=True)
    first = r5.groupby(['config', 'inst_hash'])[['hour', 'w', 'cut']].min()
    seen = r5.drop_duplicates(['config', 'hour', 'w', 'cut'])
    drop = set()
    for r in seen.itertuples():
        keep = r5[(r5.config == r.config) & (r5.inst_hash == r.inst_hash)].sort_values(['hour', 'w', 'cut']).iloc[0]
        if (keep.hour, keep.w, keep.cut) != (r.hour, r.w, r.cut):
            drop.add((r.config, r.hour, r.w, r.cut))
    k = lambda X: [(c, h, w, u) not in drop for c, h, w, u in zip(X.config, X.hour, X.w, X.cut)]
    rb = rb[k(rb)]; df = df[k(df)]
rb = rb.drop_duplicates(['config', 'hour', 'w', 'cut', 'method'])
check('NumMethodRuns', len(rb) + len(df), 0)
d = rb[rb.config == 'default']
check('NumInstDefault', d.drop_duplicates(['hour', 'w', 'cut']).shape[0], 0)
check('NumInstRandom', rb[rb.config == 'inc_random'].drop_duplicates(['hour', 'w', 'cut']).shape[0], 0)
for meth, nm in (('POLY', 'ShortPOLY'), ('CGM', 'ShortCGM'), ('GS', 'ShortGS'), ('DiCE', 'ShortDiCE')):
    x = d[d.method == meth]; check(nm, (x.G_fallback - x.G_ref).median(), 0.0005)
check('DiceWorseDefault', 100 * (d[d.method == 'DiCE'].G > 1 + 1e-9).mean(), 0.5)
ref = rb.drop_duplicates(['config', 'hour', 'w', 'cut']).set_index(['config', 'hour', 'w', 'cut']).G_ref
dfm = df.groupby(['config', 'hour', 'w', 'cut', 'method']).G_fallback.median().reset_index()
for meth, nm in (('COBYQA', 'ShortCOBYQA'), ('CMAES', 'ShortCMAES'), ('SLSQP_FD', 'ShortSLSQPFD')):
    x = dfm[(dfm.config == 'default') & (dfm.method == meth)].set_index(['config', 'hour', 'w', 'cut']).G_fallback
    check(nm, (x - ref.reindex(x.index)).median(), 0.0005)
allm = pd.concat([rb[['G_fallback', 'G_ref']],
                  dfm.merge(ref.reset_index(), on=['config', 'hour', 'w', 'cut'])[['G_fallback', 'G_ref']]])
check('NegShortfall', int(((allm.G_fallback - allm.G_ref) < -0.01).sum()), 0)
# family comparison: best repaired procedure vs best adapted generator, Wilcoxon (zsplit) + Holm + HL
wins = 0
for cfg_name, X in rb.groupby('config'):
    P = X.pivot_table(index=['hour', 'w', 'cut'], columns='method', values='G_fallback').dropna()
    a = P[['POLY', 'ELLP', 'CGM']].min(axis=1).values; b = P[['GS', 'FACE', 'NICE_R', 'DAACE_R', 'DiCE']].min(axis=1).values
    dd = a - b
    walsh = (dd[:, None] + dd[None, :]) / 2; hl = np.median(walsh[np.triu_indices(len(dd))])
    p = wilcoxon(dd, zero_method='zsplit').pvalue if np.any(dd != 0) else 1.0
    # Holm within the configuration uses 18 tests in the analysis; the family test is reproduced only
    # through its direction and raw significance here, so count wins with the analysis' adjusted p.
    T = pd.read_csv(f'{PROCESSED}/methods_paired_tests.csv')
    row = T[(T.config == cfg_name) & (T.A == 'proc') & (T.B == 'base')].iloc[0]
    if abs(row.HL - hl) > 1e-9 or abs(row.p - p) > 1e-12:
        checks.append((f'FamilyTest[{cfg_name}]', (hl, p), (row.HL, row.p), False))
    wins += bool(row.p_holm < 0.05 and hl <= -0.05)
check('ProcBeatsBaseCfg', wins, 0)
# superiority by a margin: one-sided signed-rank test of (best of family) - G_ref - 0.05 > 0, Holm over all tests
fams = {'gen': ['GS', 'FACE', 'NICE_R', 'DAACE_R', 'DiCE'], 'proc': ['POLY', 'ELLP', 'CGM'], 'dfo': ['COBYQA', 'CMAES', 'SLSQP_FD']}
wide = pd.concat([rb[['config', 'hour', 'w', 'cut', 'method', 'G_fallback']], dfm])
mres = []
for cfg_name, X in wide.groupby('config'):
    P = X.pivot_table(index=['hour', 'w', 'cut'], columns='method', values='G_fallback').dropna(axis=1, how='all').dropna()
    r_ = ref.loc[cfg_name].reindex(P.index).values
    for fam, ms in fams.items():
        if all(m_ in P.columns for m_ in ms):
            e_ = P[ms].min(axis=1).values - r_ - 0.05
            mres.append((fam, wilcoxon(e_, zero_method='zsplit', alternative='greater').pvalue if np.any(e_ != 0) else 1.0))
pv = np.array([x[1] for x in mres]); order = np.argsort(pv); adj = np.empty(len(pv)); run = 0.0
for k, i in enumerate(order):
    run = max(run, (len(pv) - k) * pv[i]); adj[i] = min(1.0, run)
for fam, nm in (('gen', 'MarginGenSig'), ('proc', 'MarginProcSig'), ('dfo', 'MarginDfoSig')):
    check(nm, sum(1 for (f_, _), q in zip(mres, adj) if f_ == fam and q < 0.05), 0)
check('MarginTests', len(mres), 0)

# ------------------------------------------------------------------ fifth round
import os
RV = f'{RAW_ATTAIN}/../additional'
if os.path.exists(f'{RV}/scip/anchors.csv'):
    an = pd.read_csv(f'{RV}/scip/anchors.csv')
    it = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{RV}/scip/incumbents_targets_p*.csv'))])
    i1 = it.drop_duplicates(['hour', 'w']); check('ScipIncN', len(i1), 0)
    tt = it.dropna(subset=['tar_lb']); check('ScipTarN', len(tt), 0)
    check('ScipTarBad', int(((tt.C_tar - tt.tar_lb) / tt.tar_lb > 1e-5).sum()), 0)
    check('ScipAnchStoredLocal', int(((an.Cmin_stored - an.Cmin_lb) / an.Cmin_lb > 1e-5).sum()), 0)
cfm = sorted(glob.glob(f'{RV}/cf_with_methods_p*.jsonl'))
if cfm:
    J = pd.concat([pd.read_json(f, lines=True) for f in cfm])
    J = J[J.method.isin(['REF', 'GS', 'FACE', 'NICE_R'])]
    wrong, n_ = 0, 0
    for (h, w_, c_), X in J[J.kind == 'rho'].groupby(['hour', 'w', 'cut']):
        P = X.pivot_table(index='value', columns='method', values='G')
        if len(P) < 10 or len(J[(J.hour == h) & (J.w == w_) & (J.cut == c_)].value.unique()) < 19:
            continue
        thr_ = X.thr.iloc[0]; g = P[['GS', 'FACE', 'NICE_R']].min(axis=1)
        ref_v = P.index[P.REF <= thr_ + 1e-9].min(); gen_v = P.index[g <= thr_ + 1e-9].min() if (g <= thr_ + 1e-9).any() else np.nan
        n_ += 1; wrong += bool(np.isnan(gen_v) or gen_v > ref_v)
    check('CfmRhoGenWrong', 100 * wrong / n_, 0.5)

# ------------------------------------------------------------------ fifth and sixth rounds (independent recomputation)
from eeldattain.model import System as _Sys
from eeldattain.paths import COEFFS as _CO, RAW as _RAW
_S = _Sys(str(_CO['10']))
_R = cat(f'{RAW_ATTAIN}/requests_e10*.csv'); _R = _R[_R.duplicate_of_w.isna() & ~_R.reachable.astype(bool)]
_V = {(v['hour'], v['w'], v['cut']): v for f in sorted(glob.glob(f'{RAW_ATTAIN}/vectors_e10*.jsonl')) for v in map(json.loads, open(f))}
_dev = []
for r in _R.itertuples():
    v = _V[(r.hour, r.w, r.cut)]; Pi, Ft, sc = np.array(v['Pinc']), np.array(v['Ft']), np.array(v['scale'])
    _dev.append(abs(r.G_phys - (r.E_min - Ft[1]) / (sc[1] * np.linalg.norm((_S.F(Pi) - Ft) / sc))))
check('ClosedPhysN', len(_dev), 0)
checks.append(('ClosedPhysMax', max(_dev), N.get('ClosedPhysMax'), max(_dev) <= 1e-9))
_I = pd.read_csv(_RAW / 'additional' / 'scip' / 'incumbents_inverse.csv')
check('ScipIncInvN', len(_I), 0)
checks.append(('ScipIncInvMax', float(_I.rel_excess.max()), N.get('ScipIncInvMax'), float(_I.rel_excess.max()) <= 2e-6))

# ------------------------------------------------------------------ certificate
de10 = cat(f'{RAW_CERT}/de_e10_p*.csv'); de40 = cat(f'{RAW_CERT}/de_e40_p*.csv')
check('CertSampleTen', len(de10), 0); check('CertFailTen', int((de10.improvement > 0.01).sum()), 0)
check('CertMaxImpTen', de10.improvement.max(), 0.00005)
check('CertSampleForty', len(de40), 0); check('CertFailForty', int((de40.improvement > 0.01).sum()), 0)
st10 = pd.read_csv(f'{RAW_CERT}/structural_e10.csv'); st40 = pd.read_csv(f'{RAW_CERT}/structural_e40.csv')
viol = ['viol_R', 'viol_rho', 'viol_frontier', 'viol_phys', 'out_of_range']
check('CertCellsTen', int(st10.cells.sum()), 0); check('CertViolTen', int(st10[viol].sum().sum()), 0)
check('CertCellsForty', int(st40.cells.sum()), 0); check('CertViolForty', int(st40[viol].sum().sum()), 0)

# ------------------------------------------------------------------ operational extension
oq = cat(f'{RAW_OPS}/ops_requests_*.csv', dtype={'S_units': str}); ou = oq[oq.duplicate_of_w.isna()]
og = cat(f'{RAW_OPS}/ops_grid_*.csv').merge(ou[['hour', 'w', 'cut']], on=['hour', 'w', 'cut'])
check('OpsNumReq', len(oq), 0); check('OpsNumReqUnique', len(ou), 0)
check('OpsLossPctMed', ou.loss_inc_pct.median(), 0.005)
hw = ou.groupby('hour').agg(W=('front_width', 'first'), eta=('eta_E', 'mean'))
check('OpsSpearman', spearmanr(hw.W, hw.eta)[0], 0.005)
mm = ou.merge(u[['hour', 'w', 'cut', 'G_ref']], on=['hour', 'w', 'cut'], suffixes=('', '_nl'))
check('OpsPairN', len(mm), 0)
check('OpsGrefDiff', (mm.G_ref_uniform - mm.G_ref).mean(), 0.0005)
check('OpsGrefRankR', spearmanr(mm.G_ref_uniform, mm.G_ref)[0], 0.005)
for band, lev, nm in (('uniform', [0.05, 0.1, 0.2], 'Uni'), ('ramp', [10, 15, 30], 'Ramp')):
    x = og[(og.band == band) & (og.R < 1e8) & og.R.isin([25, 50, 100]) & og.level.isin(lev)]
    mx = (x.G - x.G_phys).clip(lower=0).groupby([x.hour, x.w, x.cut]).mean()
    check(f'OpsAllowShare{nm}', 100 * (mx >= 0.1).mean(), 0.05)
check('OpsRampVsUniDiff', (ou.G_ref_ramp - ou.G_ref_uniform).mean(), 0.0005)
nd = ou[ou.G_ref_ramp > ou.G_phys + 0.05]
check('OpsNeedCF', len(nd), 0)
q = nd.tau_star.quantile([0.25, 0.5, 0.75])
check('OpsTauStarMed', q[0.5], 0.5); check('OpsTauStarQa', q[0.25], 0.5); check('OpsTauStarQc', q[0.75], 0.5)
check('OpsTauStarLeSixty', 100 * (nd.tau_star <= 60).mean(), 0.5)
check('OpsRStarMed', nd.R_star.median(), 0.5)
ss = nd.dropna(subset=['S_size']); check('OpsNeedSet', len(ss), 0); check('OpsSetLeTwo', 100 * (ss.S_size <= 2).mean(), 0.5)
wn = cat(f'{RAW_OPS}/window_*.csv').dropna(subset=['G_window'])
check('WinN', len(wn), 0)
check('WinFixedDiff', (wn.G_fixed - wn.G_single).mean(), 0.0005)
check('WinFixedMedAbs', (wn.G_fixed - wn.G_single).abs().median(), 0.0005)
check('WinFixedMax', (wn.G_fixed - wn.G_single).max(), 0.0005)
check('WinAdjDiff', (wn.G_window - wn.G_single).mean(), 0.0005)
check('WinAdjSmall', 100 * ((wn.G_window - wn.G_single).abs() < 0.05).mean(), 0.5)
x7 = cat(f'{RAW_OPS}/sets40_*.csv', dtype={'exact_size': str})
check('ExactN', len(x7), 0); x7n = x7[x7.needs_set]; check('ExactNeed', len(x7n), 0)
for k, nm in (('1', 'One'), ('2', 'Two'), ('>=3', 'Three')):
    check(f'Exact{nm}', int((x7n.exact_size == k).sum()), 0)
check('ExactLeTwo', 100 * x7n.exact_size.isin(['1', '2']).mean(), 0.5)
sd = cat(f'{RAW_OPS}/seeds_*.csv')
spread = sd.groupby(['hour', 'cut', 'R', 'tau']).G.agg(lambda v: v.max() - v.min())
check('SeedCells', len(spread), 0); check('SeedMaxDiff', spread.max(), 0.00005)

# ------------------------------------------------------------------ supplementary experiments (when present)
from eeldattain.paths import RAW_SUPP
if (RAW_SUPP / 'window_lock.csv').exists():
    L = pd.read_csv(RAW_SUPP / 'window_lock.csv'); e = L[L.construction == 'efficient_window']
    check('WinLockPairs', int(e.locked.sum()), 0)
    lk = set(zip(e[e.locked].hour, e[e.locked].w)); W0 = cat(f'{RAW_OPS}/window_*.csv')
    check('WinLockReq', int(sum((h, w) in lk for h, w in zip(W0.hour, W0.w))), 0)
    check('WinLockMatch', int(sum(((h, w) in lk) == (g > 0.999) for h, w, g in zip(W0.hour, W0.w, W0.G_fixed))), 0)
if (RAW_SUPP / 'oracle_a.csv').exists():
    O = pd.read_csv(RAW_SUPP / 'oracle_a.csv')
    O = O.groupby(['hour', 'w', 'cut', 'B', 'sigma', 'method'], as_index=False).shortfall.median()
    for B, sg, tag in ((25, 0.0, 'TwentyFive'), (100, 0.0, 'Hundred'), (100, 0.05, 'HundredNoisy')):
        x = O[(O.B == B) & (O.sigma == sg)]
        check(f'OracleBestMed{tag}', x.groupby('method').shortfall.median().min(), 0.0005)
        check(f'OraclePOLY{tag}', x[x.method == 'POLY'].shortfall.median(), 0.0005)
if glob.glob(f'{RAW_SUPP}/pairs40_p*.csv'):
    P = cat(f'{RAW_SUPP}/pairs40_p*.csv')
    check('PairsReq', len(P), 0); check('PairsClosing', int(P.pairs_closing.sum()), 0)
if (RAW_SUPP / 'anchor_impact_e10.csv').exists():
    D = pd.read_csv(RAW_SUPP / 'anchor_impact_e10.csv')
    D = D[~D.duplicate_stored.astype(bool)]          # distinct requests only, as for RQ1
    hw = D.groupby('hour').agg(Ws=('W_stored', 'first'), We=('W_ext', 'first'), es=('eta_stored', 'mean'), ee=('eta_ext', 'mean'))
    check('ImpactSpearStored', spearmanr(hw.Ws, hw.es)[0], 0.005); check('ImpactSpearExt', spearmanr(hw.We, hw.ee)[0], 0.005)

if glob.glob(f'{RAW_SUPP}/ops40full_p*.csv'):         # Phase 2: full stylized 40-unit replication
    F = cat(f'{RAW_SUPP}/ops40full_p*.csv', dtype={'S_class_loss': str})
    check('OpsFortyFullN', len(F), 0)
    nd = F.G_ref_ramp_loss > F.G_phys_loss + 0.05
    check('OpsFortyFullNeedCF', int(nd.sum()), 0)
    check('OpsFortyFullAllowShare', 100 * ((F.G_ref_ramp_loss - F.G_phys_loss) >= 0.10).mean(), 0.5)
    d = (F.G_ref_uniform_loss - F.G_ref_uniform_noloss).values
    check('OpsFortyFullLossDiff', d.mean(), 0.0005); check('OpsFortyFullLossP', wilcoxon(d).pvalue, 0.0005)
    check('OpsFortyFullTauMed', F[nd].tau_star_loss.median(), 0.5)
    cs = F[nd].S_class_loss.astype(str); cs = cs[cs.isin(['1', '2', '>=3'])]
    check('OpsFortyFullNeedSet', len(cs), 0); check('OpsFortyFullSLeTwo', 100 * cs.isin(['1', '2']).mean(), 0.5)
if glob.glob(f'{RAW_SUPP}/certificate_ops40_p*.csv'):
    C = cat(f'{RAW_SUPP}/certificate_ops40_p*.csv')
    check('OpsFortyCertCells', len(C), 0); check('OpsFortyCertMaxImp', (C.G_stored - C.G_DE).max(), 0.0005)
    check('OpsFortyCertFail', int(((C.G_stored - C.G_DE) > 0.01).sum()), 0)
if glob.glob(f'{RAW_SUPP}/ops40_seeds*.csv'):
    S7 = cat(f'{RAW_SUPP}/ops40_seeds*.csv')
    cl = ['G_phys', 'G_ref_uniform', 'G_ref_ramp', 'G_Rfree_ramp', 'G_bandfree']
    g = S7.groupby(['PD', 'w', 'cut', 'model'])[cl]
    check('OpsFortySeedMaxSpread', float((g.max() - g.min()).values.max()), 0.0005)
    check('OpsFortySeedCells', int(g.ngroups * len(cl)), 0)
# allowance ablation (10-unit): last budget step at the widest band, recomputed from the grid
G10 = cat(f'{RAW_ATTAIN}/grid_e10*.csv'); R10 = cat(f'{RAW_ATTAIN}/requests_e10*.csv')
u = R10[R10.duplicate_of_w.isna()][['hour', 'w', 'cut']]
Gu = G10[G10.kind == 'grid'].merge(u, on=['hour', 'w', 'cut']).pivot_table(index=['hour', 'w', 'cut'], columns=['R', 'rho'], values='G')
check('AblLastRGainTopRhoTen', (Gu[(100.0, 0.2)] - Gu[(250.0, 0.2)]).mean(), 0.0005)
check('AblLastRhoGainTopRTen', (Gu[(250.0, 0.1)] - Gu[(250.0, 0.2)]).mean(), 0.0005)

# extended method audit and its convex variant: share of instances whose family-best dispatch ends within 0.05
FAMS = {'Gen': ('GS', 'FACE', 'NICE_R', 'DAACE_R', 'DiCE'), 'Proc': ('POLY', 'ELLP', 'CGM'),
        'Dfo': ('COBYQA', 'SLSQP_FD', 'CMAES')}
for pre, sub in (('Ext', 'methods_extended'), ('Cvx', 'methods_extended_convex')):
    fs = sorted(glob.glob(f'{RAW}/additional/{sub}/raw_p*.csv'))
    if not fs:
        continue
    A = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    keys = A[['hour', 'w', 'cut', 'inst_hash']].drop_duplicates().sort_values(['hour', 'w', 'cut'])
    keys = keys[~keys.inst_hash.duplicated()]                       # repeated instances enter once
    A = A.merge(keys[['hour', 'w', 'cut']], on=['hour', 'w', 'cut'])
    check(f'{pre}N', len(keys), 0)
    for nm, ms in FAMS.items():
        shares, sr, su = [], [], []
        for sd in sorted(set(A.mseed) - {0}):
            x = A[A.method.isin(ms) & A.mseed.isin([0, sd])]
            b_ = x.groupby(['hour', 'w', 'cut']).agg(g=('G_fallback', 'min'), ref=('G_ref', 'first'), r=('reachable', 'first'))
            ok = (b_.g - b_.ref) <= 0.05
            shares.append(100 * ok.mean()); sr.append(100 * ok[b_.r.astype(bool)].mean()); su.append(100 * ok[~b_.r.astype(bool)].mean())
        check(f'{pre}{nm}Within', np.mean(shares), 0.5)
        check(f'{pre}{nm}WithinReach', np.mean(sr), 0.5); check(f'{pre}{nm}WithinUnreach', np.mean(su), 0.5)

# Proposition 1 (inclusion) on the calibrated compliance run: no compliance band or budget above the gap value by more than a step
fs = sorted(glob.glob(f'{RAW}/additional/compliance/compliance_calibrated_p*.csv'))
if fs:
    Cc = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    st = cat(f'{RAW_ATTAIN}/requests_e10*.csv', dtype={'S_units': str}); st = st[st.duplicate_of_w.isna()][['hour', 'w', 'cut', 'G_ref', 'G_phys', 'rho_star', 'R_star']]
    x = Cc[np.isclose(Cc.eps_gap, 0.05)].merge(st, on=['hour', 'w', 'cut'], suffixes=('', '_s'))
    both = (x.G_ref_s - x.G_phys_s > 0.05) & x.need_E.astype(bool)
    inv = both & ((x.rho_star_E - x.rho_star_s > 0.95 / 256 + 1e-9) | (x.R_star_E - x.R_star_s > 1000 / 256 + 1e-9))
    check('CalFiveBoth', both.sum(), 0); check('CalFiveInv', inv.sum(), 0)
    check('CalFiveOnlyE', (x.need_E.astype(bool) & ~(x.G_ref_s - x.G_phys_s > 0.05)).sum(), 0)

# consistency checks: direct band values off by more than a step lie below rho_0, window without locked schedules
fd = f'{RAW}/additional/direct_cf.csv'
if glob.glob(fd):
    X = pd.read_csv(fd); off = (X.rho_direct - X.rho_star).abs() > 0.95 / 256 + 1e-9
    check('DirOffN', off.sum(), 0); check('DirOffBelowRhoZero', (off & (X.rho_direct < 0.05)).sum(), 0)
if (RAW_SUPP / 'window_lock.csv').exists():
    e = pd.read_csv(RAW_SUPP / 'window_lock.csv'); e = e[e.construction == 'efficient_window']
    lk = set(zip(e[e.locked].hour, e[e.locked].w)); wn2 = cat(f'{RAW_OPS}/window_*.csv').dropna(subset=['G_window'])
    k = np.array([(h, w) not in lk for h, w in zip(wn2.hour, wn2.w)])
    check('WinNoLockMedAbs', (wn2.G_fixed - wn2.G_single)[k].abs().median(), 0.0005)

# one-sided distance (Proposition 2): allowance share on the study region
fo = f'{RAW}/additional/onesided/cells_p0.csv'
if glob.glob(fo):
    Co = pd.concat([pd.read_csv(f) for f in glob.glob(f'{RAW}/additional/onesided/cells_p*.csv')])
    pho = Co[Co.kind == 'physical'].set_index(['hour', 'w', 'cut']).Gp
    go = Co[(Co.kind == 'grid') & Co.R.isin([25, 50, 100])].copy()
    go['a'] = (go.Gp.values - pho.reindex(pd.MultiIndex.from_frame(go[['hour', 'w', 'cut']])).values).clip(min=0)
    check('OsAllow', 100 * (go.groupby(['hour', 'w', 'cut']).a.mean() >= 0.10).mean(), 0.05)
# SCIP on the 40-unit study region
ff = glob.glob(f'{RAW}/additional/scip/forty_region_p*.csv')
if ff:
    Xf = pd.concat([pd.read_csv(f) for f in ff])
    check('FrCert', ((Xf.G_stored - Xf.G_lb) <= 0.01 + 1e-9).sum(), 0); check('FrBetter', (Xf.G_ub < Xf.G_stored - 0.01).sum(), 0)
    check('FrBetterMax', (Xf.G_stored - Xf.G_ub).max(), 0.0005)

bad = [c for c in checks if not c[3]]
for c in checks:
    print(f'{"OK " if c[3] else "BAD"} {c[0]:<26} recomputed={c[1]!s:<26} reported={c[2]}')
print(f'{len(checks) - len(bad)} of {len(checks)} checks agree')
sys.exit(1 if bad else 0)
