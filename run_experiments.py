"""Single entry point for the experiments and every table of the manuscript.

  python run_experiments.py --exp tables      # all tables, number macros and processed results from results/raw (minutes)
  python run_experiments.py --exp campaign    # main campaign, certificate, method comparison, operational extension
  python run_experiments.py --exp supplementary  # supplementary experiments (window, oracle, 40-unit operational, pairs, scales)
  python run_experiments.py --exp all         # campaign + supplementary + tables

Experiments write to --out (default results/rerun) and never overwrite results/raw. With --exp all, the
tables are then rebuilt from the rerun outputs (EELD_RAW=<out>). Figures are not part of this entry point.
scripts/make_figures.sh builds them.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLE_STEPS = ['analysis/attainability.py', 'analysis/methods.py', 'analysis/certificate.py', 'analysis/operational.py',
               'analysis/supplementary.py', 'analysis/ablation.py', 'analysis/additional.py', 'analysis/additional2.py', 'analysis/compliance.py', 'analysis/network.py', 'analysis/rq_summary.py', 'analysis/audit_counts.py', 'analysis/verify_numbers.py']


def run(cmd, env):
    print(f'[{time.strftime("%H:%M:%S")}] $ {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f'step failed with status {r.returncode}: {" ".join(cmd)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--exp', default='tables', choices=['tables', 'campaign', 'supplementary', 'all'])
    ap.add_argument('--out', default='results/rerun', help='output root for experiments (default results/rerun)')
    a = ap.parse_args()
    need = {'tables': ['analysis', 'results/raw'], 'campaign': ['experiments'], 'supplementary': ['experiments'],
            'all': ['analysis', 'experiments']}[a.exp]
    absent = [d for d in need if not (ROOT / d).is_dir()]
    if absent:
        raise SystemExit(f'--exp {a.exp} needs {", ".join(absent)}/, which this archive does not contain (see README.md)')
    env = dict(os.environ, PYTHONPATH=f'{ROOT / "src"}{os.pathsep}{os.environ.get("PYTHONPATH", "")}', MPLBACKEND='Agg')
    out = str((ROOT / a.out).resolve())
    if a.exp in ('campaign', 'all'):
        run(['bash', 'scripts/run_campaign.sh'], dict(env, OUT=out, LOG=f'{out}/logs'))
    if a.exp in ('supplementary', 'all'):
        run(['bash', 'scripts/run_supplementary_experiments.sh'], dict(env, OUT=f'{out}/supplementary', LOG=f'{out}/logs/supplementary'))
    if a.exp in ('tables', 'all'):
        tenv = dict(env)
        if a.exp == 'all':
            tenv['EELD_RAW'] = out
        for step in TABLE_STEPS:
            run([sys.executable, step] + (['--concept'] if step.endswith('attainability.py') else []), tenv)
    print('done', flush=True)


if __name__ == '__main__':
    main()
