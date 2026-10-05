.PHONY: setup data features eda test lint

setup:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

data:
	bash scripts/download_data.sh

features:
	.venv/bin/python -m fraud.data

eda:
	.venv/bin/python -m fraud.eda

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests
