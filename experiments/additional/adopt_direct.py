"""Store the global minima of the direct formulation (22) where SCIP finds a smaller sufficient allowance than bisection.

scip_direct.py solves (22) globally for the counterfactuals whose minimality the bisection bounds leave open. Where its
optimum lies more than one bisection step below the stored value, the stored rho* or R* was not minimal. This script
writes the SCIP optimum into results/raw/attainability/requests_e10*.csv and keeps the bisection value in the columns
rho_star_bisect and R_star_bisect, so the miss stays visible as an audit result.
Output: the updated campaign files and results/raw/additional/scip/direct_adopted.csv
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import RAW_ADD
from eeldattain.paths import RAW_ATTAIN


def main():
    D = pd.read_csv(RAW_ADD / 'scip' / 'direct.csv')
    A = D[D.status.isin(['optimal', 'gaplimit']) & (D.U < D.below - 1e-9)]
    A.to_csv(RAW_ADD / 'scip' / 'direct_adopted.csv', index=False)
    col = {'rho': 'rho_star', 'R': 'R_star'}
    for f in sorted(RAW_ATTAIN.glob('requests_e10*.csv')):
        R = pd.read_csv(f, dtype={'S_units': str})
        for c in ('rho_star', 'R_star'):
            if f'{c}_bisect' not in R:
                R[f'{c}_bisect'] = R[c]
        for a in A.itertuples():
            m = (R.hour == a.hour) & (R.w == a.w) & (R.cut == a.cut)
            R.loc[m, col[a.cf]] = a.U
            if m.any():
                print(f.name, a.hour, a.w, a.cut, a.cf, round(a.value, 4), '->', round(a.U, 4), flush=True)
        R.to_csv(f, index=False)


if __name__ == '__main__':
    main()
