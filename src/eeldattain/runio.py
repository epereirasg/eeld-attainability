"""Output handling shared by the experiment runners.

* Run metadata (package versions, data hashes, git commit) is written next to every raw output.
* A runner that resumes into an existing output directory must use the same configuration. A
  different configuration aborts the run instead of silently mixing results.
* Anchor and incumbent caches are stored as JSON (exact float64 round trip), not pickle.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA, ROOT


def sha256(path) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ('-dirty' if dirty else '') if out.returncode == 0 else 'unknown'
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def run_metadata() -> dict:
    import scipy
    data = {str(p.relative_to(DATA)): sha256(p) for p in sorted(DATA.rglob('*.csv'))}
    return dict(started=time.strftime('%Y-%m-%dT%H:%M:%S'), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                platform=platform.platform(), git_commit=git_commit(), data_sha256=data,
                omp_num_threads=os.environ.get('OMP_NUM_THREADS', ''),
                openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS', ''), cpu_count=os.cpu_count())


def _canonical(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def guard_config(outdir, name: str, config: dict) -> Path:
    """Write ``config_<name>.json`` or check that an existing one matches ``config``.

    Raises RuntimeError when resuming with a different configuration.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f'config_{name}.json'
    if f.exists():
        old = json.loads(f.read_text())
        if _canonical(old.get('config')) != _canonical(config):
            raise RuntimeError(f'{f} holds a different configuration. Use a new output directory '
                               f'(--out) instead of mixing results from two configurations.')
        meta = old.setdefault('resumed', [])
        meta.append(run_metadata())
    else:
        old = dict(config=_canonical(config), metadata=run_metadata())
    f.write_text(json.dumps(old, indent=1, default=str))
    return f


def append_rows(path, rows) -> None:
    """Append rows to a CSV with a stable schema.

    Rows are aligned to the existing header; keys a row lacks are written as empty cells. If a row
    brings new keys, the file is rewritten once with the union of the columns, so a column can never
    shift into its neighbour's place.
    """
    if not rows:
        return
    path = Path(path)
    new = pd.DataFrame(rows)
    if not path.exists():
        new.to_csv(path, index=False)
        return
    header = list(pd.read_csv(path, nrows=0).columns)
    extra = [c for c in new.columns if c not in header]
    if extra:
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        pd.concat([old, new.astype(object)], ignore_index=True)[header + extra].to_csv(path, index=False)
    else:
        new.reindex(columns=header).to_csv(path, mode='a', header=False, index=False)


def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return {'__nd__': x.tolist()}
    if isinstance(x, dict):
        return {'__dict__': [[_to_jsonable(k), _to_jsonable(v)] for k, v in x.items()]}
    if isinstance(x, (list, tuple)):
        return {'__seq__': [_to_jsonable(v) for v in x], 'tuple': isinstance(x, tuple)}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _from_jsonable(x):
    if isinstance(x, dict):
        if '__nd__' in x:
            return np.asarray(x['__nd__'], dtype=float)
        if '__dict__' in x:
            return {_from_jsonable(k): _from_jsonable(v) for k, v in x['__dict__']}
        if '__seq__' in x:
            seq = [_from_jsonable(v) for v in x['__seq__']]
            return tuple(seq) if x.get('tuple') else seq
    return x


def save_cache(path, obj) -> None:
    Path(path).write_text(json.dumps(_to_jsonable(obj)))


def load_cache(path):
    return _from_jsonable(json.loads(Path(path).read_text()))


def campaign_escale(case: str, outdir=None) -> float:
    """Emission convention (factor on the emission polynomial) recorded in the configuration of a stored campaign.

    Both primary campaigns use 1, and the 40-unit variant of Appendix B keeps the factor 1e-2 that its data source
    prints. Scripts that re-solve stored requests read the factor here, so that they evaluate the same objective.
    """
    from .paths import RAW_ATTAIN
    d = Path(outdir) if outdir is not None else RAW_ATTAIN
    for f in sorted(d.glob(f'config_e{case}*.json')):
        cfg = json.load(open(f)).get('config', {}).get('cfg', {})
        return float(cfg.get('escale', 1.0))
    return 1.0
