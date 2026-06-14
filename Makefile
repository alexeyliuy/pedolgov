PY ?= python

.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "install  install runtime package"
	@echo "dev      install with dev extras and pre-commit"
	@echo "lint     ruff + black --check + isort --check"
	@echo "type     mypy --strict"
	@echo "test     pytest"
	@echo "smoke    two-step fit + clear on the smoke config"
	@echo "docker   build the container image"
	@echo "clean    remove caches and build artefacts"

install:
	$(PY) -m pip install .

dev:
	$(PY) -m pip install -e .[dev]
	pre-commit install

lint:
	ruff check .
	black --check pedolgov tests
	isort --check-only pedolgov tests

type:
	$(PY) -m mypy

test:
	$(PY) -m pytest

smoke:
	$(PY) -m pedolgov.gateway stratify --config configs/experiment/_smoke.yaml
	$(PY) -m pedolgov.gateway clear --config configs/experiment/_smoke.yaml

docker:
	docker build -t pedolgov:0.1.0 .

clean:
	rm -rf build dist *.egg-info .mypy_cache .ruff_cache .pytest_cache runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
