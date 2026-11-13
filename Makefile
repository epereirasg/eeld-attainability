# Convenience targets. Each one calls a documented command of README.md.
PY ?= python3
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: install test test-fast tables figures verify campaign supplementary clean

install:            ## pinned packages (Python 3.13), then this package in editable mode
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps

test:               ## all tests (under 1 min)
	$(PY) -m pytest

test-fast:          ## skip the reporting-chain test
	$(PY) -m pytest -m "not slow"

tables:             ## tables, number macros and processed files from results/raw
	$(PY) run_experiments.py --exp tables

figures:            ## Figs. 1 and 3 to 9 from results/raw
	scripts/make_figures.sh

verify:             ## independent recomputation of the manuscript numbers
	$(PY) analysis/verify_numbers.py

campaign:           ## main campaign into results/rerun (does not touch results/raw)
	$(PY) run_experiments.py --exp campaign

supplementary:      ## supplementary experiments into results/rerun/supplementary
	$(PY) run_experiments.py --exp supplementary

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache
