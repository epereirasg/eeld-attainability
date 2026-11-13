"""Integrity of the stored results and of the reporting chain (raw outputs -> processed files -> tables and macros).

These tests read the committed raw outputs; they do not rerun the optimization campaign.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from eeldattain import paths
from eeldattain.runio import guard_config, load_cache, save_cache, append_rows, sha256

ROOT = paths.ROOT
PY = sys.executable
needs_runners = pytest.mark.skipif(not (ROOT / 'experiments').is_dir(), reason='experiment runners not in this archive')


def run(script, env_root=None, args=()):
    env = dict(os.environ, PYTHONPATH=f'{ROOT / "src"}', MPLBACKEND='Agg')
    if env_root is not None:
        env['EELD_ROOT'] = str(env_root)
    return subprocess.run([PY, str(ROOT / script), *args], capture_output=True, text=True, env=env, timeout=1200)


def test_input_data_hashes_match_manifest():
    for line in (paths.DATA / 'SHA256SUMS.txt').read_text().splitlines():
        h, name = line.split()
        assert sha256(paths.DATA / name) == h, name


def test_stored_campaign_is_complete_and_unique():
    R = pd.concat([pd.read_csv(f) for f in sorted(paths.RAW_ATTAIN.glob('requests_e10*.csv'))])
    assert len(R) == 24 * 3 * 3 and not R.duplicated(['hour', 'w', 'cut']).any()
    G = pd.concat([pd.read_csv(f) for f in sorted(paths.RAW_ATTAIN.glob('grid_e10*.csv'))])
    assert G.groupby(['hour', 'w', 'cut']).size().eq(20 + 7).all()      # 20 grid cells + 7 frontier points
    R4 = pd.concat([pd.read_csv(f) for f in sorted(paths.RAW_ATTAIN.glob('requests_e40*.csv'))])
    assert len(R4) == 6 * 2 * 2 and not R4.duplicated(['hour', 'w', 'cut']).any()
    O = pd.read_csv(paths.RAW_OPS / 'ops_requests_a.csv')
    assert len(O) == 24 * 3 * 2 and not O.duplicated(['hour', 'w', 'cut']).any()
    W = pd.read_csv(paths.RAW_OPS / 'window_a.csv')
    assert len(W) == 22 * 3 * 2 and W.G_window.notna().all()
    S = pd.concat([pd.read_csv(f) for f in sorted(paths.RAW_OPS.glob('seeds_*.csv'))])
    assert S.groupby(['hour', 'cut']).rep.nunique().eq(3).all() and len(S) == 8 * 2 * 3 * 16


def test_stored_gaps_satisfy_the_guaranteed_properties():
    G = pd.concat([pd.read_csv(f) for f in sorted(paths.RAW_ATTAIN.glob('grid_e*.csv'))])
    assert G.G.between(0, 1).all() and (G.G >= G.G_phys - 0.01).all()
    for _, X in G[G.kind == 'grid'].groupby(['case', 'hour', 'w', 'cut']):
        p = X.pivot_table(index='R', columns='rho', values='G').values
        assert (np.diff(p, axis=0) <= 0.01).all() and (np.diff(p, axis=1) <= 0.01).all()


def test_caches_round_trip_exactly(tmp_path):
    A, INC = load_cache(paths.RAW_OPS / 'cache' / 'ops_h1.json')
    save_cache(tmp_path / 'c.json', (A, INC))
    B, J = load_cache(tmp_path / 'c.json')
    assert all(np.array_equal(A[k], B[k]) for k in ('Pc', 'Pe', 'ideal', 'nadir'))
    assert list(INC) == [0.25, 0.5, 0.75] and all(np.array_equal(INC[w], J[w]) for w in INC)


def test_guard_config_refuses_a_different_configuration(tmp_path):
    guard_config(tmp_path, 'x', dict(a=1, b=(1, 2)))
    guard_config(tmp_path, 'x', dict(a=1, b=[1, 2]))           # same configuration: resume allowed
    with pytest.raises(RuntimeError, match='different configuration'):
        guard_config(tmp_path, 'x', dict(a=2, b=[1, 2]))
    meta = json.loads((tmp_path / 'config_x.json').read_text())
    assert 'data_sha256' in meta['metadata'] and len(meta['resumed']) == 1


def test_append_rows_keeps_columns_aligned(tmp_path):
    """Regression: rows with extra or missing keys used to shift later columns."""
    f = tmp_path / 'o.csv'
    append_rows(f, [dict(a=1, b=2, z=9)])
    append_rows(f, [dict(a=3, extra=7, b=4, z=8)])         # new key in the middle
    append_rows(f, [dict(a=5, z=6)])                        # missing key
    d = pd.read_csv(f)
    assert list(d.columns) == ['a', 'b', 'z', 'extra']
    assert d.z.tolist() == [9, 8, 6] and d.b.tolist()[:2] == [2, 4] and np.isnan(d.b.iloc[2])
    assert np.isnan(d.extra.iloc[0]) and d.extra.iloc[1] == 7


def test_stored_csv_files_have_consistent_rows():
    import csv
    for f in sorted(paths.RAW.rglob('*.csv')):
        with open(f) as fh:
            r = csv.reader(fh); n = len(next(r))
            assert all(len(row) == n for row in r), f


def test_append_rows_writes_header_once(tmp_path):
    f = tmp_path / 'o.csv'
    append_rows(f, [dict(a=1, b=2)]); append_rows(f, [dict(a=3, b=4)]); append_rows(f, [])
    assert pd.read_csv(f).values.tolist() == [[1, 2], [3, 4]]


@needs_runners
def test_stored_configs_match_the_runner_configuration():
    sys.path.insert(0, str(ROOT / 'experiments' / 'attainability'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('att_run', ROOT / 'experiments' / 'attainability' / 'run.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for tag, case in (('e10a', '10'), ('e10b', '10'), ('e40a', '40'), ('e40b', '40')):
        stored = json.loads((paths.RAW_ATTAIN / f'config_{tag}.json').read_text())['config']['cfg']
        cur = {k: v for k, v in mod.CFG[case].items() if k not in mod.PATH_KEYS}
        if case == '10':     # the stored 10-unit runs predate the SCIP check; certified_rerun.json documents its hours
            cur.pop('certify', None)
        else:                # the primary 40-unit campaign runs through experiments/additional/forty_main.py
            cur.update(escale=1.0, kb=4, bis=8, certify=True)
        assert json.loads(json.dumps(cur)) == stored, tag


def test_certificate_sample_order_reproduces_stored_cells():
    """The EELD40 sample is drawn with the grid files in the order e40b, e40a."""
    for case, order, parts in (('10', ['e10a', 'e10b'], 1), ('40', ['e40b', 'e40a'], 2)):
        G = pd.concat([pd.read_csv(paths.RAW_ATTAIN / f'grid_{t}.csv') for t in order]).reset_index(drop=True)
        smp = G.groupby(['kind', 'R', 'rho']).sample(frac=0.05, random_state=7)
        cells = pd.concat([smp.iloc[p::parts] for p in range(parts)])[['hour', 'w', 'cut', 'kind', 'R', 'rho']]
        stored = pd.concat([pd.read_csv(f) for f in sorted(paths.RAW_CERT.glob(f'de_e{case}_p*.csv'))])
        key = list(cells.columns)        # rows of rerun hours were merged at the end of the files, so compare as sets
        assert cells.sort_values(key).reset_index(drop=True).equals(stored[key].sort_values(key).reset_index(drop=True))


def test_seed_rule_collisions_are_the_documented_ones():
    """The operational seed rule maps some (hour, weight) pairs to one seed (documented, kept for reproducibility)."""
    from eeldops.experiments import seed_of
    seeds = {}
    for h in range(1, 25):
        for w in (0.25, 0.5, 0.75):
            for c in (0.05, 0.10):
                seeds.setdefault(seed_of(2, h, w, c), []).append((h, w, c))
    coll = {k: v for k, v in seeds.items() if len(v) > 1}
    assert ((8, 0.25, 0.05), (3, 0.75, 0.05)) in [tuple(sorted(v, reverse=True)) for v in coll.values()]
    assert all(len({x[1] for x in v}) > 1 for v in coll.values())   # collisions only across different weights


def test_no_heatmaps_and_no_hardcoded_paths_in_code():
    code = [p for d in ('src', 'analysis', 'experiments', 'scripts', 'tests') for p in (ROOT / d).glob('**/*.py')
            if p.name != 'test_pipeline.py']
    assert len(code) > 20
    for p in code:
        t = p.read_text()
        assert 'imshow(' not in t and 'pcolormesh(' not in t and 'heatmap(' not in t, p
        assert '/home/' not in t and '/mnt/' not in t and '/root/' not in t, p


@pytest.mark.slow
def test_reporting_chain_is_reproducible_and_verified(tmp_path):
    """Rebuild macros and tables from the raw outputs in a copy and compare them with the committed ones."""
    for d in ('data', 'results', 'src'):
        shutil.copytree(ROOT / d, tmp_path / d)
    for script in ('analysis/methods.py', 'analysis/certificate.py', 'analysis/rq_summary.py'):
        r = run(script, env_root=tmp_path); assert r.returncode == 0, r.stderr[-2000:]
    for f in ('results/macros/numbers_methods.tex', 'results/macros/numbers_cert.tex', 'results/tables/tab_family_tests.tex',
              'results/tables/tab_methods_default.tex', 'results/tables/tab_rq.tex'):
        assert (tmp_path / f).read_bytes() == (ROOT / f).read_bytes(), f
    r = run('analysis/verify_numbers.py', env_root=tmp_path)
    assert r.returncode == 0 and 'checks agree' in r.stdout
    # a corrupted macro must make the verification fail (regression: failures used to be ignored)
    f = tmp_path / 'results' / 'processed' / 'numbers_ops.json'
    N = json.loads(f.read_text()); N['OpsTauStarMed'] = '999'; f.write_text(json.dumps(N))
    r = run('analysis/verify_numbers.py', env_root=tmp_path)
    assert r.returncode == 1 and 'BAD OpsTauStarMed' in r.stdout


def test_compare_reruns_reports_zero_difference_on_identical_data(tmp_path):
    sub = tmp_path / 'ops'; sub.mkdir()
    shutil.copy(paths.RAW_OPS / 'seeds_a.csv', sub / 'seeds_a.csv')
    r = run('analysis/compare_reruns.py', args=(str(tmp_path),))
    assert r.returncode == 0 and 'worst |diff|=0 ' in r.stdout
