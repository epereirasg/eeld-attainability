"""Criteria table for the six objectives of the evaluation (O1 to O6), generated from the processed numbers.

The objectives map onto the identifiers of docs/evaluation_criteria.md as O1=RQ1, O2=RQ2, O3=RQ3 and RQ7, O4=RQ4,
O5=RQ5 and RQ6, O6=RQ8 to RQ11.

Panel A holds the claims about the method (global bounds, the checks of Proposition 1, reproducibility and the method
comparison) with the verdicts Supported and Not supported. Panel B holds the characterizations of the test systems
(materiality, saturation, joint binding, sparsity and hour coupling) with the verdicts Rule met and Rule not met, and
one phrase on what the outcome tells the operator. No acceptance rule differs from docs/evaluation_criteria.md.
"""
import json
from eeldattain.paths import ROOT, TABLES
WS = str(ROOT)
TAB = str(TABLES)
N = {}
for f in ('attain', 'methods', 'cert', 'ops', 'supplementary', 'additional', 'additional2', 'network', 'compliance'):
    try: N.update(json.load(open(f'{WS}/results/processed/numbers_{f}.json')))
    except FileNotFoundError: pass
g = lambda k, d='n/a': N.get(k, d)
fl = lambda k, d=0.0: float(N.get(k, d))


def sup(ok): return 'Met' if ok else 'Not met'
def met(ok): return 'Met' if ok else 'Not met'
def parts(a, b, c): return f"(a) {a}, (b) {b}, (c) {c}"
def which(*ok, ok_sets=None):
    bad = [p for p, o in zip('abc', ok) if o is False] + (['sets'] if ok_sets is False else [])
    return 'Met' if not bad else 'Not met, (' + ', '.join(bad) + ')'


A, B = [], []
# ------------------------------------------------------------------ panel A, method claims
seed_ok = g('SeedCfOk') == g('SeedCfN')
A.append(('O3 (seeds)', '$\\rho^{*}$ and $R^{*}$ agree across solver seeds within 0.004 and 8~MW',
          f"{g('SeedCfOk')} of {g('SeedCfN')} pairs within tolerance", sup(seed_ok), 'The instruction does not depend on the solver seed'))
cert_ok = all(N.get(k, '1') == '0' for k in ('CertViolTen', 'CertFailTen')) and N.get('NegShortfall') == '0'
A.append(('O4', 'No structural check of Lemma~1 fails and no independent DE solve improves a sampled cell by more than $0.01$',
          f"{g('CertViolTen')} violations in {g('CertCellsTen')} cells, {g('CertFailTen')} DE failures in {g('CertSampleTen')}", sup(cert_ok), 'No missed optimum found by sampling'))
if 'ScipCells' in N:
    ok = N.get('ScipFail') == '0' and N.get('ScipOpen', '0') == '0'
    A.append(('O4 (global)', 'Same criterion, SCIP on every cell of the final 10-unit dataset', f"{g('ScipCert')} of {g('ScipCells')} cells globally bounded within 0.01",
              sup(ok), 'Every stored gap is a global optimum to 0.01'))
if 'CfbN' in N:
    A.append(('O4 (counterfactuals)', 'SCIP bounds propagated through each bisection step, no criterion (additional)',
              f"{g('CfbSuff')} of {g('CfbN')} proven sufficient, {g('CfbMinimalAll')} minimal to one step by the bounds", 'Reported', 'The minimal allowance is checked, not only the gap'))
if 'DirScipN' in N:
    A.append(('O4 (direct)', 'SCIP on the direct formulation for the undecided steps, no criterion (additional)',
              f"{g('DirScipMinimal')} of {g('DirScipN')} minimal, {g('DirScipSmaller')} replaced by the global minimum, so {g('CfbMinimalFinal')} of {g('CfbN')} minimal", 'Reported',
              'Bisection can miss the minimum by a step'))
if 'CompCertNRho' in N:
    A.append(('O4 (compliance)', 'Feasible dispatch at the value and SCIP bound one step below, no criterion (additional)',
              f"{int(g('CompCertSuffRho'))+int(g('CompCertSuffR'))} of {int(g('CompCertNRho'))+int(g('CompCertNR'))} sufficient, {int(g('CompCertMinRho'))+int(g('CompCertMinR'))} minimal",
              'Reported', 'The compliance counterfactuals are exact to one step'))
if 'ScipOpsN' in N:
    A.append(('O4 (losses, global)', 'Same criterion, SCIP on the physical and reference cells of the loss model', f"{g('ScipOpsCert')} of {g('ScipOpsN')} cells globally bounded",
              sup(N['ScipOpsCert'] == N['ScipOpsN']), 'Losses do not hide missed optima'))
if 'NetScipCells' in N:
    A.append(('O4 (30-bus, global)', 'Same criterion, SCIP on every stored cell (additional)', f"{g('NetScipCert')} of {g('NetScipCells')} cells globally bounded",
              sup(N['NetScipCert'] == N['NetScipCells']), 'Line limits do not hide missed optima'))
if 'CertFailForty' in N:
    ok40 = N.get('CertViolForty') == '0' and N.get('CertFailForty') == '0'
    A.append(('O4 (40-unit)', 'Same criterion, empirical check', f"{g('CertViolForty')} violations, {g('CertFailForty')} DE failures in {g('CertSampleForty')}", sup(ok40),
              'No missed optimum found by sampling'))
if 'ScipFortyN' in N:
    ok = N.get('ScipFortyBetter', '0') == '0'
    A.append(('O4 (40-unit, global)', 'Same criterion, SCIP for \\SI{600}{s} on every reference and physical cell (additional)',
              f"{g('ScipFortyCert')} of {g('ScipFortyN')} cells globally bounded, {g('ScipFortyOpenN')} open, {g('ScipFortyBetter')} improved by more than 0.01",
              sup(ok), 'Every 40-unit reference value is a global optimum to 0.01' if N.get('ScipFortyOpenN', '1') == '0' and ok
              else ('The open cell keeps' if N.get('ScipFortyOpenN') == '1' else 'Open cells keep') + ' the empirical check only'))
A.append(('O5 (procedures)', 'A difference holds in at least 7 of 9 configurations, with Holm-adjusted $p<0.05$ and a shift of at least $0.05$. The outcome applies this rule to request-level tests and has no inferential value. Procedures against generators',
          f"{g('ProcBeatsBaseCfg')} of 9", sup(int(g('ProcBeatsBaseCfg', 0)) >= 7), 'Repaired procedures beat the adapted generators'))
A.append(('O5 (DFO)', 'Same criterion, derivative-free optimizers against procedures', f"{g('DfoBeatsProcCfg')} of {g('DfoCfgTotal')}",
          sup(int(g('DfoBeatsProcCfg', 0)) >= 7 and g('DfoCfgTotal') == '9'), 'Derivative-free optimizers are the better inner solvers' if int(g('DfoBeatsProcCfg', 0)) >= 7 else 'Their advantage over the procedures depends on the configuration'))
if 'ExtN' in N:
    A.append(('O5 (audit)', 'Primary comparison added after the main campaign. Default configuration, all hours, reachable and unreachable requests, ten seeds, exact sign-flip test of hour means',
              f"generators within 0.05 in {g('ExtGenWithin')}\\%, procedures {g('ExtProcWithin')}\\%, DFO {g('ExtDfoWithin')}\\%, generators $p={g('ExtGenP')}$"
              + (f", generators {g('CvxGenWithin')}\\% with convex cost" if 'CvxGenWithin' in N else ''),
              'Reported', 'The adapted generators stop short of the attainable, also without the valve-point term'))
if 'CfmRhoN' in N:
    A.append(('O5 (counterfactuals)', 'Counterfactuals with each family as inner solver, no criterion (additional)',
              f"generators equal the reference band in {int(g('CfmRhoN')) - int(g('CfmRhoGenFailK'))} of {g('CfmRhoN')} sampled requests", 'Reported', 'A miss of the inner solver changes the instruction'))
if 'OracleBestTwentyFive' in N:
    A.append(('O5 (oracle)', 'Oracle-limited comparison, no criterion (additional)', f"best {g('OracleBestTwentyFive')} with 25 calls, {g('OracleBestHundredNoisy')} with noise", 'Reported',
              'With scarce calls an evolution strategy leads'))
A.append(('O5 (no gain)', 'A dispatch worse than the incumbent is scored as the incumbent, and its frequency is reported',
          f"DiCE-R worse than inc. in {g('DiceWorseDefault')}\\% (default)", 'Reported', 'Some generators return no improvement'))
A.append(('O3 (redispatch)', 'A fraction of the full re-solve is named only if the median supports it',
          f"$R^{{*}}$ = {g('RStarRatioTen')}\\% of a full re-solve (median)", 'Reported', 'A bounded instruction moves less than a full re-solve'))

# ------------------------------------------------------------------ panel B, characterizations of the test systems
B.append(('O1', 'Spearman correlation between $W_h$ and the mean $\\eta_E$ of at least $0.6$. A consistency check of \\eqref{eq:reach}',
          f"{g('SpearmanTen')}", met(fl('SpearmanTen') >= 0.6), 'Judge a request against the front of its hour'))
if 'SpearmanForty' in N:
    B.append(('O1 (40-unit)', 'Same criterion, six demand levels (descriptive)', f"{g('SpearmanForty')}", met(fl('SpearmanForty') >= 0.6), 'Six levels give only a coarse estimate'))
RULE2 = ('(a) Mean $G^{*}_{\\mathrm{allow}}\\ge0.10$ over the study region in at least half of the requests, lower 95\\% bound above $0.40$. '
         '(b) Mean change in $G^{*}$ from 100 to 250~MW at most $0.02$ for $\\rho\\le0.10$. (c) Budget alone and band alone each remove less than 70\\% of the gap in most requests')
okA = fl('AllowShareTen') >= 50 and fl('AllowShareLoTen') > 40; okB = fl('RqSatTen', 1) <= 0.02; okC = fl('RqJointTen') > 50
B.append(('O2', RULE2, parts(f"{g('AllowShareTen')}\\% [{g('AllowShareLoTen')}, {g('AllowShareHiTen')}]", g('RqSatTen'), f"{g('RqJointTen')}\\% of {g('RqJointNTen')}"),
          which(okA, okB, okC), 'State both a total budget and per-unit bands'))
if 'AllowShareForty' in N and 'RqSatForty' in N:
    okA = fl('AllowShareForty') >= 50 and fl('AllowShareLoForty') > 40; okB = fl('RqSatForty', 1) <= 0.02; okC = fl('RqJointForty') > 50
    reading = 'A larger budget still lowers the gap on a fleet of many similar units' if okB is False else 'Budgets saturate, bands carry the gap'
    B.append(('O2 (40-unit)', 'Same criterion, region $R\\in\\{100,250,500\\}$~MW', parts(f"{g('AllowShareForty')}\\% [{g('AllowShareLoForty')}, {g('AllowShareHiForty')}]",
              f"{g('RqSatForty')}", f"{g('RqJointForty')}\\% of {g('RqJointNForty')}"), which(okA, okB, okC), reading))
if 'NetAllowShareDc' in N:
    okA = fl('NetAllowShareDc') >= 50 and fl('NetAllowShareLoDc') > 40; okB = fl('NetSatDc', 1) <= 0.02; okC = fl('NetJointDc') > 50
    B.append(('O2 (30-bus)', 'Same criterion, DC line limits (additional)', parts(f"{g('NetAllowShareDc')}\\% [{g('NetAllowShareLoDc')}, {g('NetAllowShareHiDc')}]",
              g('NetSatDc'), f"{g('NetJointDc')}\\% of {g('NetNeedDc')}"), which(okA, okB, okC), 'One band carries the gap'))
B.append(('O3 (sparsity)', '$|S^{*}|\\le2$ for at least half of the requests that need a unit counterfactual',
          f"{g('SetLeTwoTen')}\\% (band-active search)", met(fl('SetLeTwoTen') >= 50), 'Lifting one or two bands usually suffices'))
if 'RevSetLeTwo' in N:
    B.append(('O3 (all units)', 'Same criterion, all sets of at most three units (additional)', f"{g('RevSetLeTwo')}\\% [{g('RevSetLeTwoCiLo')}, {g('RevSetLeTwoCiHi')}]",
              met(fl('RevSetLeTwo') >= 50), 'Same reading with an exhaustive search'))
if 'PairsFullCorrLeTwo' in N:
    ok = fl('PairsFullCorrLeTwo') >= 50
    B.append(('O3 (40-unit)', 'Same criterion, exhaustive pair search (O6)', f"{g('PairsFullCorrLeTwo')}\\% (greedy {g('SetLeTwoForty')}\\%)", met(ok),
              'Lifting one or two bands usually suffices' if fl('PairsFullCorrLeTwo') > 50
              else 'Lifting one or two bands suffices in half of the requests' if ok else 'Lift many bands or grant a response time'))
if 'OpsSpearman' in N:
    okA = fl('OpsAllowShareUni') >= 50 and fl('OpsAllowShareLoUni') > 40
    okB = fl('RqSatOpsUni', 1) <= 0.02 if 'RqSatOpsUni' in N else None
    okC = fl('RqJointOpsUni') > 50 if 'RqJointOpsUni' in N else None
    ok = fl('OpsSpearman') >= 0.6 and okA and okB is not False and okC is not False
    obs = f"O1 {g('OpsSpearman')}, O2 " + parts(f"{g('OpsAllowShareUni')}\\% [{g('OpsAllowShareLoUni')}, {g('OpsAllowShareHiUni')}]", g('RqSatOpsUni'),
                                                   f"{g('RqJointOpsUni')}\\% of {g('RqJointNOpsUni')}")
    B.append(('O6 (losses)', 'The O1 and O2 criteria hold in the loss model, uniform bands', obs, met(ok), 'Losses leave the reading unchanged'))
    okA = fl('OpsAllowShareRamp') >= 50 and fl('OpsAllowShareLoRamp') > 40
    okB = fl('OpsRampSat', 1) <= 0.02; okC = fl('OpsJoint') > 50; okS = fl('OpsSetLeTwo') >= 50
    obs = parts(f"{g('OpsAllowShareRamp')}\\% [{g('OpsAllowShareLoRamp')}, {g('OpsAllowShareHiRamp')}]", g('OpsRampSat'), f"{g('OpsJoint')}\\%") + f", $|S^{{*}}|\\le2$ in {g('OpsSetLeTwo')}\\%"
    v = which(okA, okB, okC, ok_sets=okS)
    B.append(('O6 (ramps)', 'The criteria of O2 and O3 hold with ramp bands ($\\tau\\le30$~min in (b))',
              obs, v, 'Grant response time, a larger budget still lowers the gap slightly'))
if 'OpsFortyFullAllowShare' in N:
    okA = fl('OpsFortyFullAllowShare') >= 50 and fl('OpsFortyFullAllowShareLo') > 40
    okC = fl('RqJointOpsForty') > 50 if 'RqJointOpsForty' in N else None
    okS = fl('OpsFortyFullSLeTwo') >= 50
    obs = parts(f"{g('OpsFortyFullAllowShare')}\\% [{g('OpsFortyFullAllowShareLo')}, {g('OpsFortyFullAllowShareHi')}] at the reference", 'no budget grid',
                f"{g('RqJointOpsForty')}\\% of {g('RqJointNOpsForty')}") + f", $|S^{{*}}|\\le2$ in {g('OpsFortyFullSLeTwo')}\\%"
    v = which(okA, None, okC, ok_sets=okS)
    B.append(('O6 (ramps, 40-unit)', 'Same criteria, stylized losses and ramps', obs, v,
              'Ramp bands dominate the budget' if fl('OpsFortyFullReliefBand') > fl('OpsFortyFullReliefBudget') else 'The budget dominates the ramp bands'))
if 'WinFixedMedAbs' in N:
    B.append(('O6 (window)', 'The single-hour analysis suffices if the median $|\\Delta G^{*}|<0.05$. Neighbors fixed',
              f"{g('WinFixedMedAbs')} (mean {g('WinFixedDiff')}, {g('WinFixedSmall')}\\% below 0.05, max {g('WinFixedMax')})", met(fl('WinFixedMedAbs') < 0.05) + ' (median only)', 'Frozen neighbors can lock schedules, so check the tail'))
    B.append(('O6 (window)', 'Same criterion, neighbors adjustable', f"{g('WinAdjMedAbs')} (mean {g('WinAdjDiff')}, {g('WinAdjSmall')}\\% below 0.05)", met(fl('WinAdjMedAbs') < 0.05),
              'A single-hour analysis suffices'))
if 'PairsFullCorrOne' in N:
    B.append(('O6 (pairs)', 'No criterion', f"{g('PairsFullCorrOne')}/{g('PairsFullCorrTwo')}/{g('PairsFullCorrThree')} of {g('ExactNeed')} (1/2/no pair found)",
              'Reported', 'Greedy screening understates sparsity'))

HEAD = ('\\begin{tabular}{@{}p{1.9cm}p{5.6cm}p{3.9cm}p{1.5cm}p{3.3cm}@{}}\n\\toprule\n'
        'Objective & Criterion & Observed & Outcome & Reading \\\\\n\\midrule\n')
for name, rows, title in (('tab_rq.tex', A, 'Panel A. Claims about the method'),
                          ('tab_rq_b.tex', B, 'Panel B. Characterizations of the test systems')):
    with open(f'{TAB}/{name}', 'w') as f:      # one float per panel, so that each fits on a page
        f.write(HEAD + f'\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\\n')
        for r in rows: f.write(' & '.join(r) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

# ------------------------------------------------------------------ compact table of the manuscript
# One short criterion per objective. The full panels above stay in the repository.
def mt(ok): return 'Met' if ok else 'Not met'
def w(v): return v.replace('Rule met', 'Met').replace('Rule not met', 'Not met')
PA, PB = [], []
ok10 = all(N.get(k, '1') == '0' for k in ('CertViolTen', 'CertFailTen')); ok40 = N.get('CertViolForty') == '0' and N.get('CertFailForty') == '0'
PA.append(('O3, seeds', '$\\rho^{*}$ and $R^{*}$ agree across solver seeds within 0.004 and 8~MW', f"{g('SeedCfOk')} of {g('SeedCfN')}",
           mt(g('SeedCfOk') == g('SeedCfN')), 'The instruction does not depend on the seed'))
PA.append(('O4', 'No structural violation and no DE improvement above 0.01 (10 / 40 units)',
           f"{g('CertViolTen')} and {g('CertFailTen')} / {g('CertViolForty')} and {g('CertFailForty')}", mt(ok10) + ' / ' + mt(ok40),
           'No missed optimum found by sampling, SCIP extends it (Table~\\ref{tab:scope})'))
PA.append(('O5, procedures', 'Procedures beat generators in 7 of 9 configurations ($p_{\\mathrm{Holm}}<0.05$, shift $\\ge0.05$, request level)',
           f"{g('ProcBeatsBaseCfg')} of 9", 'Reported, test assumptions not met', 'The 24-hour audit carries the family comparison'))
PA.append(('O5, DFO', 'Same criterion, derivative-free optimizers against procedures', f"{g('DfoBeatsProcCfg')} of {g('DfoCfgTotal')}",
           'Reported, test assumptions not met', 'Their lead over the procedures depends on the configuration'))
PA.append(('O5, no gain', 'Report how often a method returns no improvement', f"DiCE-R {g('DiceWorseDefault')}\\% (default)", 'Reported',
           'Some generators return the incumbent'))
PA.append(('O3, redispatch', 'Name a fraction of the full re-solve only if the median supports it', f"$R^{{*}}$ at {g('RStarRatioTen')}\\% (median)", 'Reported',
           'A bounded instruction moves less than a full re-solve'))
PB.append(('O1', 'Spearman of $W_h$ and mean $\\eta_E$ at least 0.6 (10 / 40 units)', f"{g('SpearmanTen')} / {g('SpearmanForty')}",
           mt(fl('SpearmanTen') >= 0.6) + ' / ' + mt(fl('SpearmanForty') >= 0.6), 'Judge a request against the front of its hour'))
okA = fl('AllowShareTen') >= 50 and fl('AllowShareLoTen') > 40; okB = fl('RqSatTen', 1) <= 0.02; okC = fl('RqJointTen') > 50
PB.append(('O2', '(a) Mean $G^{*}_{\\mathrm{allow}}\\ge0.10$ in half of the requests, lower bound above 0.40, (b) gain from 100 to 250~MW at most 0.02, (c) joint binding in most requests',
           parts(f"{g('AllowShareTen')}\\%", g('RqSatTen'), f"{g('RqJointTen')}\\%"), w(which(okA, okB, okC)), 'State both a total budget and per-unit bands'))
okA = fl('AllowShareForty') >= 50 and fl('AllowShareLoForty') > 40; okB = fl('RqSatForty', 1) <= 0.02; okC = fl('RqJointForty') > 50
PB.append(('O2, 40 units', 'Same criterion, $R\\in\\{100,250,500\\}$~MW', parts(f"{g('AllowShareForty')}\\%", g('RqSatForty'), f"{g('RqJointForty')}\\%"),
           w(which(okA, okB, okC)), 'A larger budget still lowers the gap on a large fleet' if okB is False else 'The budget saturates, bands carry the gap'))
okA = fl('NetAllowShareDc') >= 50 and fl('NetAllowShareLoDc') > 40; okB = fl('NetSatDc', 1) <= 0.02; okC = fl('NetJointDc') > 50
PB.append(('O2, 30-bus', 'Same criterion, DC line limits (additional)', parts(f"{g('NetAllowShareDc')}\\%", g('NetSatDc'), f"{g('NetJointDc')}\\%"),
           w(which(okA, okB, okC)), 'One band carries the gap'))
PB.append(('O3, unit sets', '$|S^{*}|\\le2$ in half of the requests that need a unit set (10 / 40 units)', f"{g('SetLeTwoTen')}\\% / {g('PairsFullCorrLeTwo')}\\%",
           mt(fl('SetLeTwoTen') >= 50) + ' / ' + mt(fl('PairsFullCorrLeTwo') >= 50),
           'Lifting one or two bands usually suffices' if fl('PairsFullCorrLeTwo') > 50
           else 'One or two bands suffice on 10 units, in half of the 40-unit requests' if fl('PairsFullCorrLeTwo') == 50
           else 'One or two bands suffice on 10 units, more on 40 units'))
okS = fl('OpsSpearman') >= 0.6; okA = fl('OpsAllowShareUni') >= 50 and fl('OpsAllowShareLoUni') > 40
okB = fl('RqSatOpsUni', 1) <= 0.02; okC = fl('RqJointOpsUni') > 50
PB.append(('O6, losses', 'The O1 and O2 criteria hold with losses', f"{g('OpsSpearman')}, " + parts(f"{g('OpsAllowShareUni')}\\%", g('RqSatOpsUni'), f"{g('RqJointOpsUni')}\\%"),
           mt(okS and okA and okB and okC), 'Losses leave the reading unchanged'))
okA = fl('OpsAllowShareRamp') >= 50 and fl('OpsAllowShareLoRamp') > 40; okB = fl('OpsRampSat', 1) <= 0.02; okC = fl('OpsJoint') > 50
PB.append(('O6, ramps', 'The O2 and O3 criteria hold with ramp bands ($\\tau\\le30$~min in (b))', parts(f"{g('OpsAllowShareRamp')}\\%", g('OpsRampSat'), f"{g('OpsJoint')}\\%")
           + f", {g('OpsSetLeTwo')}\\%", w(which(okA, okB, okC, ok_sets=fl('OpsSetLeTwo') >= 50)), 'Grant response time, a larger budget still lowers the gap slightly'))
okA = fl('OpsFortyFullAllowShare') >= 50 and fl('OpsFortyFullAllowShareLo') > 40
okC = fl('RqJointOpsForty') > 50 if 'RqJointOpsForty' in N else None
PB.append(('O6, ramps, 40 units', 'Same criteria, stylized losses and ramps, reference allowance only', parts(f"{g('OpsFortyFullAllowShare')}\\%", 'no grid', f"{g('RqJointOpsForty')}\\%")
           + f", {g('OpsFortyFullSLeTwo')}\\%", w(which(okA, None, okC, ok_sets=fl('OpsFortyFullSLeTwo') >= 50)),
           'Ramp bands dominate the budget' if fl('OpsFortyFullReliefBand') > fl('OpsFortyFullReliefBudget') else 'The budget dominates the ramp bands'))
PB.append(('O6, window', 'Median $|\\Delta G^{*}|<0.05$ in the window, neighbors fixed / adjustable', f"{g('WinFixedMedAbs')} / {g('WinAdjMedAbs')}, {g('WinNoLockMedAbs')} fixed without locks",
           mt(fl('WinFixedMedAbs') < 0.05) + ' / ' + mt(fl('WinAdjMedAbs') < 0.05), 'Check the tail when neighbors are frozen'))
PB.append(('O6, pairs', 'Exhaustive 40-unit pair search, no criterion', f"{g('PairsFullCorrOne')}, {g('PairsFullCorrTwo')} and {g('PairsFullCorrThree')} of {g('ExactNeed')} (1, 2, none)",
           'Reported', 'Greedy screening understates sparsity'))
with open(f'{TAB}/tab_rq_compact.tex', 'w') as f:
    f.write('\\begin{tabular}{@{}p{1.45cm}p{5.2cm}p{3.5cm}p{1.55cm}p{4.0cm}@{}}\n\\toprule\nObjective & Criterion & Observed & Outcome & Reading \\\\\n\\midrule\n')
    f.write('\\multicolumn{5}{@{}l}{\\emph{Panel A. Claims about the method}} \\\\\n')
    for r in PA: f.write(' & '.join(r) + ' \\\\\n')
    f.write('\\midrule\n\\multicolumn{5}{@{}l}{\\emph{Panel B. Characterizations of the test systems}} \\\\\n')
    for r in PB: f.write(' & '.join(r) + ' \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
print(open(f'{TAB}/tab_rq_compact.tex').read())
