"""Data loading and provenance checks for the operational extension."""
import hashlib
import numpy as np
import pandas as pd

from eeldattain.paths import DATA as _DATA

DATA = str(_DATA)
EXPECTED_SHA256 = {
    'eeld10_coeffs.csv': 'ea9ff88e46de8ee7e18c4d8ca542c476a35129f3e08857397a7ea8006c9aa42c',
    'external/eeld10_B_matrix.csv': '091a2d12945c6830881b72d03d5d683815538dd595e4c1a4959619a94e0a9c5d',
    'external/eeld10_ramp_limits.csv': '68c553b6080a905d1d1d73ef3bf5789796d0ffe3f69131ce6dd2777beede4e67',
    'external/eeld10_gen_external.csv': '643289aa03e8abb873415d3c9e13cdf4ac60d43c46ef7347989b6c08cc6262cf',
}
# column correspondence between the external transcription and our coefficient file
COLMAP = {'Pmin': 'Pmin', 'Pmax': 'Pmax', 'const': 'c', 'lin': 'b', 'quad': 'a', 'vp_e': 'e', 'vp_f': 'f',
          'alpha': 'alpha', 'beta': 'beta', 'gamma': 'gamma', 'zeta': 'zeta', 'lambda': 'lambda'}


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def verify_files():
    """Raise if any input file differs from the hashes recorded at extraction time."""
    for rel, h in EXPECTED_SHA256.items():
        got = sha256(f'{DATA}/{rel}')
        if got != h:
            raise RuntimeError(f'{rel}: sha256 {got} differs from recorded {h}')
    return True


def load_eeld10():
    """Coefficients, loss matrix, ramp limits and demand of the 10-unit system."""
    verify_files()
    coef = pd.read_csv(f'{DATA}/eeld10_coeffs.csv')
    B = np.loadtxt(f'{DATA}/external/eeld10_B_matrix.csv', delimiter=',')
    ramp = pd.read_csv(f'{DATA}/external/eeld10_ramp_limits.csv')
    dem = pd.read_csv(f'{DATA}/demand_eeld10.csv')
    ext = pd.read_csv(f'{DATA}/external/eeld10_gen_external.csv')
    for k, v in COLMAP.items():                  # external transcription agrees with our file in every shared field
        if not np.allclose(ext[k].values, coef[v].values):
            raise ValueError(f'external generator table differs from eeld10_coeffs.csv in column {k}')
    if not (np.allclose(ramp.Pmin.values, coef.Pmin.values) and np.allclose(ramp.Pmax.values, coef.Pmax.values)):
        raise ValueError('ramp table limits differ from eeld10_coeffs.csv')
    if not (B.shape == (10, 10) and np.allclose(B, B.T) and np.all(np.linalg.eigvalsh(B) > 0)):
        raise ValueError('loss matrix must be 10 x 10, symmetric and positive definite')
    if (ramp[['UR_MW_per_h', 'DR_MW_per_h']].to_numpy() <= 0).any():
        raise ValueError('ramp rates must be positive')
    return coef, B, ramp, dem
