"""Compare rerun outputs with the stored raw outputs, row by row.

Usage: python analysis/compare_reruns.py RERUN_DIR [--report FILE]

RERUN_DIR mirrors results/raw (subdirectories attain10, attain40, ops, cert, dfo, robust are mapped to
the stored files below). Rows are matched on their keys; for every numeric column the script reports
the largest absolute difference, and it lists rows that are missing from the stored outputs.
"""
import argparse, glob, json, os, sys
import pandas as pd
from eeldattain.paths import RAW_ATTAIN, RAW_OPS, RAW_CERT, RAW_DFO, RAW_ROBUST

SPECS = [  # (rerun subdir, rerun glob, stored glob, keys, columns ignored)
    ('attain10', 'requests_e10*.csv', f'{RAW_ATTAIN}/requests_e10*.csv', ['hour', 'w', 'cut'], ['seed', 'solver_fallbacks']),
    ('attain10', 'grid_e10*.csv', f'{RAW_ATTAIN}/grid_e10*.csv', ['hour', 'w', 'cut', 'kind', 'R', 'rho'], ['seed']),
    ('attain40', 'requests_e40*.csv', f'{RAW_ATTAIN}/requests_e40*.csv', ['hour', 'w', 'cut'], ['seed', 'solver_fallbacks']),
    ('attain40', 'grid_e40*.csv', f'{RAW_ATTAIN}/grid_e40*.csv', ['hour', 'w', 'cut', 'kind', 'R', 'rho'], ['seed']),
    ('ops', 'ops_requests_*.csv', f'{RAW_OPS}/ops_requests_*.csv', ['hour', 'w', 'cut'], ['runtime_s', 'solver_fallbacks']),
    ('ops', 'ops_grid_*.csv', f'{RAW_OPS}/ops_grid_*.csv', ['hour', 'w', 'cut', 'band', 'R', 'level'], []),
    ('ops', 'window_*.csv', f'{RAW_OPS}/window_*.csv', ['hour', 'w', 'cut'], ['runtime_s', 'solver_fallbacks', 'window_solve_failed']),
    ('ops', 'seeds_*.csv', f'{RAW_OPS}/seeds_*.csv', ['hour', 'cut', 'rep', 'R', 'tau'], []),
    ('ops', 'sets40_*.csv', f'{RAW_OPS}/sets40_*.csv', ['PD', 'w', 'cut'], ['runtime_s', 'solver_fallbacks']),
    ('cert', 'de_e10_*.csv', f'{RAW_CERT}/de_e10_*.csv', ['hour', 'w', 'cut', 'kind', 'R', 'rho'], ['de_seed', 'de_status']),
    ('cert', 'de_e40_*.csv', f'{RAW_CERT}/de_e40_*.csv', ['hour', 'w', 'cut', 'kind', 'R', 'rho'], ['de_seed', 'de_status']),
    ('cert', 'structural_e*.csv', f'{RAW_CERT}/structural_e*.csv', ['case', 'hour', 'w', 'cut'], []),
    ('dfo', 'raw_*.csv', f'{RAW_DFO}/raw_*.csv', ['config', 'hour', 'w', 'cut', 'method', 'mseed'], ['runtime']),
    ('robust', 'raw_*.csv', f'{RAW_ROBUST}/raw_*.csv', ['config', 'hour', 'w', 'cut', 'method'], ['runtime']),
]


def load(pattern):
    fs = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f, dtype={'S_units': str, 'exact_size': str}) for f in fs], ignore_index=True) if fs else None


def compare(new, old, keys, ignore):
    new = new.drop_duplicates(keys); old = old.drop_duplicates(keys)
    m = new.merge(old, on=keys, how='left', suffixes=('', '__old'), indicator=True)
    missing = int((m._merge != 'both').sum())
    res = {}
    for c in new.columns:
        if c in keys or c in ignore or f'{c}__old' not in m:
            continue
        a, b = m[c], m[f'{c}__old']
        if pd.api.types.is_bool_dtype(a) or pd.api.types.is_bool_dtype(b):
            a = a.astype(float); b = b.astype(float)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            both_nan = a.isna() & b.isna(); one_nan = a.isna() ^ b.isna()
            d = (a - b).abs()[~both_nan & ~one_nan]
            res[c] = dict(max_abs_diff=float(d.max()) if len(d) else 0.0, nan_mismatch=int(one_nan.sum()))
        else:
            neq = (a.fillna('').astype(str) != b.fillna('').astype(str)).sum()
            res[c] = dict(text_mismatch=int(neq))
    return len(m), missing, res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('rerun_dir'); ap.add_argument('--report', default='')
    a = ap.parse_args()
    report = {}
    for sub, pat, stored, keys, ignore in SPECS:
        new = load(os.path.join(a.rerun_dir, sub, pat)); old = load(stored)
        if new is None or old is None:
            continue
        n, missing, res = compare(new, old, keys, ignore)
        worst = max((v.get('max_abs_diff', 0.0) for v in res.values()), default=0.0)
        bad = {k: v for k, v in res.items() if v.get('max_abs_diff', 0) > 0 or v.get('nan_mismatch', 0) or v.get('text_mismatch', 0)}
        report[f'{sub}/{pat}'] = dict(rows=n, missing_in_stored=missing, worst_abs_diff=worst, differing_columns=bad)
        print(f'{sub + "/" + pat:28s} rows={n:4d} missing={missing} worst |diff|={worst:.3g} differing columns={len(bad)}')
        for k, v in bad.items():
            print(f'    {k}: {v}')
    if a.report:
        json.dump(report, open(a.report, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
