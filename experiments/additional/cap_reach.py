"""Measure the bounded emission reach of the dense allowance grid against the cap.

allowance_grid.py divided by the emission of the target, F^tar_E. For a reachable request with a slack cap this
emission lies below the cap E^tar = (1 - c) E(P_inc), so the reach was measured against a deeper cut than the request
asks for. The grid stores the bounded minimum emission Emin_b and the emission of the bounded optimum E_opt of every
cell, so the reach against the cap follows without new solves. This script rewrites eta_b and eta_opt with the cap,
keeps the target-based values in eta_b_target and eta_opt_target, and sets E_tar to the cap. For every request whose
cap is active the two values agree within the solver tolerance.
Output: the updated results/raw/additional/allowance_grid/grid_p*.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import RAW_ADD


def main():
    for f in sorted((RAW_ADD / 'allowance_grid').glob('grid_p*.csv')):
        G = pd.read_csv(f)
        if 'eta_b_target' not in G:
            G['eta_b_target'] = G.eta_b; G['eta_opt_target'] = G.eta_opt; G['E_target'] = G.E_tar
        cap = (1.0 - G.cut) * G.E_inc
        G['E_tar'] = cap
        G['eta_b'] = np.clip((G.E_inc - G.Emin_b) / (G.E_inc - cap), 0.0, 1.0)
        G['eta_opt'] = (G.E_inc - G.E_opt) / (G.E_inc - cap)
        G.to_csv(f, index=False)
        d = (G.eta_b - G.eta_b_target).abs()
        print(f.name, 'cells changed by more than 1e-6:', int((d > 1e-6).sum()), 'max change', round(float(d.max()), 4), flush=True)


if __name__ == '__main__':
    main()
