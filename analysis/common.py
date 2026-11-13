"""Shared paths, loaders and statistics for the analysis scripts."""
import glob
import numpy as np
import pandas as pd
from eeldattain import paths

WS = str(paths.ROOT)
ATT = str(paths.RAW_ATTAIN)
TAB = str(paths.TABLES)
MAC = str(paths.MACROS)
PROC = str(paths.PROCESSED)

LABEL = {'POLY': 'POLY', 'ELLP': 'ELLP', 'CGM': 'CGM', 'GS': 'GS-R', 'FACE': 'FACE-R', 'NICE_R': 'NICE-R',
         'DAACE_R': 'DAACE-R', 'DiCE': 'DiCE-R', 'COBYQA': 'COBYQA', 'CMAES': 'CMA-ES', 'SLSQP_FD': 'SLSQP-FD'}


def load_case(case):
    R = pd.concat([pd.read_csv(f, dtype={'S_units': str}) for f in sorted(glob.glob(f'{ATT}/requests_e{case}*.csv'))], ignore_index=True)
    if 'S_units' in R: R['S_units'] = R.S_units.fillna('')
    G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{ATT}/grid_e{case}*.csv'))], ignore_index=True)
    R['unique'] = R.duplicate_of_w.isna()
    key = ['hour', 'w', 'cut']
    G = G.merge(R[key + ['unique']], on=key, how='left')
    return R, G


def boot_ci(x, stat=np.mean, B=10000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    s = np.array([stat(rng.choice(x, len(x))) for _ in range(B)])
    return np.quantile(s, [0.025, 0.975])
