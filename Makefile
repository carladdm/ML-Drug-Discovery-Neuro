# ==============================================================================
# Cross-platform Makefile (GNU Make).
# Works under bash/zsh (Linux, macOS, WSL, Git-Bash) and cmd/PowerShell on
# Windows (install GNU Make via `choco install make` / `scoop install make`).
# Every recipe runs `python -m ...` from the project venv — no shell activation.
# ==============================================================================

.DEFAULT_GOAL := help

PY  ?= python
VENV := .venv

# Resolve the venv interpreter for the current platform.
ifeq ($(OS),Windows_NT)
	VENV_PY := $(VENV)/Scripts/python.exe
else
	VENV_PY := $(VENV)/bin/python
endif

.PHONY: help setup install lint format typecheck test cov ingest featurize train predict report docker-build docker-run clean

help: ## Show this help
	@grep -E "^[a-zA-Z_-]+:.*?## .*$$" $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv and install the package with dev extras
	$(PY) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -e ".[dev]"

install: ## Install dev extras into the ACTIVE interpreter
	$(PY) -m pip install -e ".[dev]"

lint: ## Ruff lint + format check
	$(VENV_PY) -m ruff check src tests scripts
	$(VENV_PY) -m ruff format --check src tests scripts

format: ## Auto-fix lint issues and format the code
	$(VENV_PY) -m ruff check --fix src tests scripts
	$(VENV_PY) -m ruff format src tests scripts

typecheck: ## Neutralise broken rdkit stubs, then run mypy in strict mode
	$(VENV_PY) scripts/ensure_rdkit_stubs_ok.py
	$(VENV_PY) -m mypy

test: ## Run the test suite
	$(VENV_PY) -m pytest -q

cov: ## Run tests with coverage report (terminal + HTML)
	$(VENV_PY) -m pytest -q --cov=mldd --cov-report=term-missing --cov-report=html

ingest: ## Fetch + curate one target: make ingest ARGS="target=maob"
	$(VENV_PY) -m mldd.cli ingest $(ARGS)

featurize: ## Build the model matrix: make featurize ARGS="target=maob"
	$(VENV_PY) -m mldd.cli featurize $(ARGS)

train: ## Tune the registry + train (MLflow): make train ARGS="target=maob task=regression"
	$(VENV_PY) -m mldd.cli train $(ARGS)

predict: ## Batch inference: make predict ARGS="-m models/m.joblib -i cands.csv -o preds.csv"
	$(VENV_PY) -m mldd.cli predict $(ARGS)

report: ## Regenerate figures/reports (Fase 7)
	$(VENV_PY) -m mldd.cli report

docker-build: ## Build the production image
	docker build -t ml-drug-discovery-neuro:latest .

docker-run: ## Run the container with mounted data/models volumes
	docker run --rm -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data ml-drug-discovery-neuro:latest

clean: ## Remove caches and build artifacts
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', 'dist', 'build')]"; $(PY) -c "import glob, os; [os.remove(f) for f in glob.glob('src/*.egg-info/*')] or None"
