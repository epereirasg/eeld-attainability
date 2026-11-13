"""The 40-unit campaign with four random starts and SCIP checks, for one emission convention.

The data source of the 40-unit system (data/README.md) prints the emission of unit i as
1e-2 (alpha_i + beta_i P_i + gamma_i P_i^2) + zeta_i exp(lambda_i P_i). The published optima of this system
(Bhattacharjee et al., Int. J. Electr. Power Energy Syst. 59, 2014, at 10,500 MW) agree with the form without the
factor, which is therefore the primary convention (--escale 1, the default). The printed form (--escale 0.01) is the
variant of Appendix B. No single factor converts one convention into the other, because the factor changes the weight
of the polynomial against the exponential term, so each convention has its own campaign (anchors, incumbents,
targets, grid, counterfactuals and greedy unit sets). The script uses four random starts per solve (kb = 4), eight
bisection steps (bis = 8), as on the 10-unit system, and
passes the anchors, incumbents and targets through the SCIP check of src/eeldattain/globalopt.py (120 s per solve).
Everything else is the configuration of experiments/attainability/run.py (case 40, same seeds).
Output: results/raw/attainability/*_e40* for the primary convention, results/raw/additional/forty_printed/ for the
variant.
Run:  python experiments/additional/forty_main.py --hours 1,3,5 --tag e40a   (and 2,4,6 with e40b)
      python experiments/additional/forty_main.py --escale 0.01 --hours 1,3,5 --tag e40a   (variant, likewise)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'attainability'))
import run as A                                            # experiments/attainability/run.py
from eeldattain.model import System
from eeldattain.paths import RAW


def main():
    k = 1.0
    if '--escale' in sys.argv:
        i = sys.argv.index('--escale'); k = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
    out = RAW / 'attainability' if k == 1.0 else RAW / 'additional' / 'forty_printed'
    out.mkdir(parents=True, exist_ok=True)
    A.System = lambda path, escale=k: System(path, escale=k)
    A.CFG['40'] = dict(A.CFG['40'], escale=k, kb=4, bis=8, certify=True)   # recorded in the run configuration
    if '--out' not in sys.argv:
        sys.argv += ['--out', str(out)]
    if '--case' not in sys.argv:
        sys.argv += ['--case', '40']
    A.main()


if __name__ == '__main__':
    main()
