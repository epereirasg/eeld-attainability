# Attainability analysis and allowance counterfactuals in emission redispatch

Code, data and results for the article

> E. Pereiras, G. Nápoles and Y. Salgueiro, "Attainability Analysis and Allowance Counterfactuals in
> Emission Redispatch," submitted to *IEEE Access*, 2026.

An operator runs an efficient economic–emission dispatch and receives a request to cut emissions, but may move only a
limited amount of generation, a total redispatch budget *R* and a band ρ per unit. The code measures how much of the
request any admissible redispatch can deliver (the attainability gap *G\**), splits the shortfall into a physical part
and an allowance part, and computes the smallest authorization that recovers what is physically attainable (the band
ρ\*, the budget *R\**, the response time τ\* and the unit set *S\**). It also computes the smallest authorization that
meets the emission cap alone (the compliance counterfactuals ρ\*_E, *R\**_E, τ\*_E and *S\**_E). The study covers the
10-unit and 40-unit valve-point benchmarks, an operational extension with Kron losses, ramp-derived bands and a
three-hour window, a 30-bus network case, global bounds with SCIP and a comparison of search methods against the
attainable value of the same admissible set.

## Repository structure

```
data/          benchmark inputs and the external 10-unit loss and ramp tables, with provenance (data/README.md)
src/           importable model code
  eeldattain/    lossless dispatch model, multistart SLSQP solver, bounded gap, SCIP check, paths, run records
  eeldops/       operational model with Kron losses, ramp-derived bands and the three-hour window
  pilotcore/     frozen solver core used to regenerate the method-comparison instances
experiments/   one runner per experiment, each writes to results/raw/
analysis/      turns raw outputs into processed files, tables, number macros and figures
scripts/       shell drivers for the main campaign, the supplementary and additional runs and the figures
results/
  raw/           stored raw outputs of every run, with their config_*.json records
  processed/     derived CSV and JSON files, including the number files numbers_*.json
  tables/        LaTeX tables of the manuscript
  macros/        LaTeX number macros of the manuscript (numbers_*.tex)
  figures/       PDF figures of the manuscript
docs/          evaluation criteria of the campaign (docs/evaluation_criteria.md)
tests/         model, constraint, data and reporting-chain tests
run_experiments.py   entry point for experiments and tables
```

The article presents the evaluation as six objectives. They map onto the identifiers of `docs/evaluation_criteria.md`
as O1 = RQ1, O2 = RQ2, O3 = RQ3 and RQ7, O4 = RQ4, O5 = RQ5 and RQ6, and O6 = RQ8 to RQ11. The folders
`experiments/supplementary/` and `experiments/additional/`, with their outputs under `results/raw/`, hold the
experiments that followed the main campaign.

## Requirements and setup

The stored runs used Python 3.13 (3.13.16 for the campaign, 3.13.17 and 3.13.18 for the checks). Exact package versions
are pinned in `requirements.txt` (numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2, seaborn 0.13.2, cma 4.5.0,
scikit-learn 1.9.1, pyscipopt 6.3.0, pytest 9.1.1). The PySCIPOpt wheel bundles SCIP 10, which the global bounds use.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest            # 42 tests, about one minute
```

`make install`, `make test`, `make tables`, `make figures` and `make verify` wrap the same commands.

The method comparison (Tables 9 to 11) imports the authors' earlier public code base
(github.com/epereirasg/eeld-cf, commit `07e983a`). `scripts/fetch_legacy_repo.sh` clones it into `external/eeld-cf/`,
and the run scripts call it automatically. It is needed only to rerun those experiments, not to rebuild tables, macros
or figures from the stored raw outputs.

## Reproducing the manuscript

There are three levels. Levels 1 and 2 read only `results/raw/` and `data/` and take a few minutes on a laptop.

### Level 1. Tables and number macros

```bash
python run_experiments.py --exp tables
```

The command runs every script of `analysis/` that writes a table or a macro file, then `analysis/verify_numbers.py`. It
rewrites `results/processed/`, `results/tables/` and `results/macros/` and ends with `181 of 181 checks agree`.
`verify_numbers.py` recomputes 181 macros from `results/raw/` with separate code and exits with an error on any
mismatch. The rebuilt files are identical to the stored ones.

### Level 2. Figures

```bash
scripts/make_figures.sh
```

It rewrites the PDF figures of `results/figures/` in about a minute. Fig. 2 is a TikZ drawing in the manuscript. PDF
bytes and tight bounding boxes depend on the platform (fonts, text metrics and stream compression), so a redrawn
figure can differ from the stored one by a fraction of a point at text edges while plotting the same data. The stored
figures are the files of the manuscript.

### Checking a manuscript source

```bash
python analysis/check_manuscript.py path/to/main.tex
```

The script compares the manuscript with the outputs of this repository. Every figure next to `main.tex` must equal the
file in `results/figures/`, every generated table that the manuscript reproduces must appear in full, and every
decimal number of the text, captions and tables must be a value of a number macro or of a generated table. Design
inputs such as the weights, cuts and tolerances are listed in the script. For the submitted manuscript it reports 312
numbers, 5 generated tables and 8 figures without a difference.

### Level 3. Rerunning the experiments

```bash
python run_experiments.py --exp campaign        # main campaign, certificate, method comparison, operational extension
python run_experiments.py --exp supplementary   # supplementary experiments
python run_experiments.py --exp all             # both, then every table rebuilt from the rerun outputs
```

These commands write to `results/rerun/` (option `--out`) and never touch `results/raw/`. On the two-core machine of
the stored runs, the campaign took about 12 CPU-hours and the supplementary experiments about 9 CPU-hours.
`scripts/run_campaign.sh`, `scripts/run_supplementary_experiments.sh` and `scripts/run_additional_experiments.sh` list
every command in the order of the stored runs, so single experiments can be rerun alone. For example,

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  python experiments/attainability/run.py --case 10 --hours 2 --tag e10x --out results/rerun/attain10
```

reruns the nine requests of hour 2 of the 10-unit campaign in about five minutes. Its rows match the stored
`requests_e10*.csv` and `grid_e10*.csv`, except in cells where a later SCIP check stored a better dispatch
(`results/raw/certificate/scip_adopted.csv`). `python analysis/compare_reruns.py results/rerun` compares rerun files
row by row with the stored ones.

## Where each result comes from

| Manuscript item | Script | Output |
|---|---|---|
| Fig. 1 | `analysis/figure_problem.py` | `results/figures/fig_problem.pdf` |
| Fig. 3 (physical reach) | `analysis/figures_seaborn.py` | `fig_hourly_reach.pdf` |
| Fig. 4 (one request in objective space) | `analysis/figures_seaborn.py` | `fig_concept.pdf` |
| Figs. 5 and 6 | `analysis/figures_additional.py` | `fig_attainability_map.pdf`, `fig_frontier_counterfactuals.pdf` |
| Fig. 7 (losses and ramp bands) | `analysis/figures_seaborn.py` | `fig_ops_losses_ramp.pdf` |
| Fig. 8 (40-unit system) | `analysis/figures_seaborn.py` | `fig_eeld40_and_sets.pdf` |
| Fig. 9 (window and unit sets) | `analysis/figures_seaborn.py` | `fig_ops_window_sets.pdf` |
| Tables 3, 5 to 8, 12 and 15, and the numbers in the text | every `analysis/*.py` | `results/macros/numbers_*.tex` |
| Audit instance counts and 40-unit anchors of Appendix B | `analysis/audit_counts.py` | `numbers_audit_counts.tex` |
| Table 9 | `analysis/additional2.py` | `tab_audit.tex` |
| Table 10 | `analysis/additional2.py` | `tab_decomp.tex` |
| Table 11 | `analysis/additional2.py` | `tab_cf_with_methods.tex` |
| Table 13 | `analysis/network.py` | `tab_network.tex` |
| Table 14 (criteria of the objectives O1 to O6) | `analysis/rq_summary.py` | `tab_rq_compact.tex` |
| Daily optima of Appendix B | `experiments/auxiliary/daily_optima.py` | `results/raw/daily_optima.json` |

Tables 1, 2 and 4 are text of the manuscript. The scripts also write `tab_oracle.tex`, `tab_ablation.tex`, `tab_rq.tex`,
`tab_rq_b.tex`, `tab_methods_default.tex`, `tab_family_tests.tex`, `tab_proposal_tests.tex` and `tab_proposal_desc.tex`,
which give further detail and are not part of the article.

| Experiment | Runner | Raw output | Objectives |
|---|---|---|---|
| 10-unit attainability campaign | `experiments/attainability/run.py` | `results/raw/attainability/*_e10*` | O1 to O3 |
| 40-unit attainability campaign | `experiments/additional/forty_main.py` | `results/raw/attainability/*_e40*` | O1 to O3, O6 |
| Empirical solver check | `experiments/certificate/run.py` | `results/raw/certificate/` | O4 |
| Global bounds | `experiments/additional/scip_*.py`, `experiments/certificate/adopt_scip.py` | `results/raw/additional/scip/` | O4 |
| Method comparison and extended audit | `experiments/robustness/run.py`, `experiments/methods_dfo/run.py`, `experiments/additional/methods_extended.py` | `results/raw/robustness/`, `methods_dfo/`, `additional/methods_extended*/` | O5 |
| Operational extension (losses, ramp bands, window, seeds, 40-unit unit sets) | `experiments/operational/run.py` | `results/raw/operational/` | O6 |
| 30-bus network case | `experiments/additional/network30.py`, `network_split.py`, `scip_network.py` | `results/raw/additional/network/` | O6 |
| Supplementary experiments | `experiments/supplementary/*.py` | `results/raw/supplementary/` | O4 to O6 |

## Global bounds and the emission convention

The anchors P^C and P^E, the incumbents of Eq. (4) and the targets of Eq. (5) pass through SCIP 10
(`src/eeldattain/globalopt.py`, relative gap 1e-6, 120 s per solve). The multistart point is polished with SLSQP and the
SCIP point is kept when it is better by more than a relative 1e-7. SCIP draws no random numbers, so the random stream of
a run does not change. The stored 10-unit rows of hours 1, 3, 12 and 24 come from a run of this pipeline, because SCIP
improved the multistart minimum-cost anchor of these hours. Hours 2 and 13, rerun the same way as controls, reproduce
the stored rows within 4e-7 in every gap and exactly in every counterfactual (`results/raw/attainability/certified_rerun.json`).
`experiments/certificate/adopt_scip.py` then stores the polished SCIP value in every 10-unit cell where branch and bound
found a better dispatch by more than 1e-4, so the bounds hold for the final dataset.

The 40-unit data source prints a factor 1e-2 on the polynomial part of the emission. The published minimum-emission
optimum of this system at 10,500 MW (Bhattacharjee et al., Int. J. Electr. Power Energy Syst. 59, 2014) agrees with the
form without the factor, so the primary 40-unit campaign uses `escale=1` (`experiments/additional/forty_main.py`, four
random starts, eight bisection steps, SCIP check). The campaign under the printed factor is stored in
`results/raw/additional/forty_printed/` and enters Table 15. Every script that re-solves stored 40-unit requests reads
the factor from the campaign configuration (`eeldattain.runio.campaign_escale`). `scip_forty.py` (600 s per cell) and
`scip_forty_region.py` (120 s per cell) bound the reference, physical and study-region cells. A cell whose stored value
is at most 0.01 needs no solve, because G\* ≥ 0 already bounds it within 0.01.

The SCIP model enters the valve-point term through the sine and absolute-value expressions of SCIP with an explicitly
bounded sine argument, and the objective uses scaled residuals. A formulation without these safeguards returned lower
bounds above attainable values in some cells. Time limits are CPU time.

## Data

All inputs are in `data/`, and nothing has to be downloaded. `data/README.md` gives the source, licence and SHA-256 sum
of every file, the emission conventions, the assumptions behind the stylized 40-unit operational data and the sources
and assumptions of the 30-bus network case.

## Seeds, threads and runtime

Every random stream has a fixed seed derived from the experiment, hour, weight and cut. Each runner writes its
configuration to a `config_*.json` file next to its output. Multistart solves on the valve-point cost can end in
different local optima when the number of linear-algebra threads changes, so the run scripts set the thread counts of
the stored runs. The main 10-unit campaign, the empirical check and the method comparison ran with two threads. The
40-unit campaign, the global bounds and the additional experiments ran with one thread.

## Known limitations

* The response-time and unit counterfactuals carry no global bound. A reported set size shows that a sufficient set
  exists, not that no smaller one does. The references of the method comparison are multistart optima that no compared
  method beats by more than 0.01.
* SCIP bounds every cell of the lossless 10-unit campaign and of the network case within 0.01, the physical and
  reference cells of the loss model, 2,728 of the 2,760 cells of its ramp-band grid, 44 of the 48 reference and
  physical cells of the 40-unit system and 146 of its 216 study-region cells. No SCIP dispatch improves a stored 40-unit
  value by more than 0.008. The 40-unit minimum-cost anchors are bounded only within 3.1%, so the 40-unit fronts are
  exploratory.
* Every band and budget counterfactual of the 10-unit campaign is proven sufficient. The propagated bounds prove 307 of
  the 322 minimal to one bisection step. SCIP on the direct formulation confirms 13 more and replaces two by slightly
  smaller global minima (`adopt_direct.py`).
* In the 40-unit operational replication with stylized losses and ramps, differential evolution improves 2 of 240
  checked cells by more than 0.01 (at most 0.037). Both are band-free relaxations of the lossless model, and no
  reference or physical cell is affected (`results/raw/supplementary/certificate_ops40_p*.csv`).
* The attainability gap uses a two-sided distance, which can prefer a dispatch that another admissible dispatch
  dominates. The efficiency loss of the bounded optimum bounds this effect, and `experiments/additional/onesided.py`
  recomputes the 10-unit campaign with the one-sided distance of Proposition 2.
* Reruns on other hardware can end in different local optima, for instance in the daily optima of Appendix B. The stored
  anchors, incumbents and targets are part of the raw results, so every reported number can be rebuilt exactly.
* The 40-unit operational replication uses stylized losses and ramp rates (`data/README.md`). No 40-unit three-hour
  window was run. The network case is lossless and uses DC line limits.
* The run scripts write their logs to `results/logs/`, which is not part of the repository.

## Citation and licence

Citation metadata are in `CITATION.cff`. The code is released under the MIT licence (`LICENSE`). The benchmark data
come from the publications cited in `data/README.md`, and the external loss and ramp tables keep the MIT licence of
their source repository (`data/external/LICENSE`).
