"""Project-relative locations of inputs and outputs.

Every script resolves its paths through this module, so the repository can be cloned anywhere.
The root is the repository directory that contains ``src/``. It can be overridden with the
environment variable ``EELD_ROOT`` (for example, to point the analysis at a copy of the results).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get('EELD_ROOT', Path(__file__).resolve().parents[2])).resolve()

DATA = ROOT / 'data'
EXTERNAL = DATA / 'external'
RESULTS = ROOT / 'results'
RAW = Path(os.environ.get('EELD_RAW', RESULTS / 'raw')).resolve()   # EELD_RAW points the analysis at a rerun
PROCESSED = RESULTS / 'processed'
LOGS = RESULTS / 'logs'
FIGURES = RESULTS / 'figures'     # PDF figures of the manuscript
TABLES = RESULTS / 'tables'       # LaTeX tables of the manuscript
MACROS = RESULTS / 'macros'       # LaTeX number macros (numbers_*.tex) of the manuscript

RAW_ATTAIN = RAW / 'attainability'
RAW_CERT = RAW / 'certificate'
RAW_ROBUST = RAW / 'robustness'
RAW_DFO = RAW / 'methods_dfo'
RAW_OPS = RAW / 'operational'
RAW_SUPP = RAW / 'supplementary'
RAW_ADD = RAW / 'additional'

COEFFS = {'10': DATA / 'eeld10_coeffs.csv', '40': DATA / 'eeld40_coeffs.csv'}
DEMAND = {'10': DATA / 'demand_eeld10.csv', '40': DATA / 'demand_eeld40.csv'}

# Earlier code base (public repository) used only by the method comparison.
LEGACY_REPO_URL = 'https://github.com/epereirasg/eeld-cf'
LEGACY_REPO_COMMIT = '07e983a'


def legacy_repo() -> Path:
    """Location of a checkout of the earlier eeld-cf repository (see README.md)."""
    p = Path(os.environ.get('EELD_LEGACY_REPO', ROOT / 'external' / 'eeld-cf')).resolve()
    if not (p / 'src' / 'cf_eeld').is_dir():
        raise FileNotFoundError(
            f'The method comparison needs the earlier eeld-cf code base at {p}. '
            f'Run scripts/fetch_legacy_repo.sh or set EELD_LEGACY_REPO.')
    return p
