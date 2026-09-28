.PHONY: help install test test-ci lint train-moons train-california evaluate baselines clean

RUN ?=

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install with dev extras (uv)
	uv sync --extra dev

test: ## Run the test suite
	uv run pytest -q

test-ci: ## Test gate for CI (deselect known failures here, never skip silently)
	uv run pytest -q -m "not slow"

lint: ## Ruff lint + format check
	uv run ruff check . && uv run ruff format --check .

train-moons: ## Train on the 2-D moons toy set (minutes on CPU)
	uv run datadiffusion train --config configs/moons.toml

train-california: ## Train on California Housing (CPU; long)
	caffeinate -i uv run datadiffusion train --config configs/california.toml

evaluate: ## Evaluate a run against baselines: make evaluate RUN=runs/california/<ts>
	@if [ -z "$(RUN)" ]; then echo "Usage: make evaluate RUN=runs/<name>/<timestamp>"; exit 1; fi
	uv run datadiffusion evaluate $(RUN) --method ddpm

baselines: ## Score only the baselines on California Housing
	uv run datadiffusion baselines --dataset california

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__
