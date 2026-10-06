.PHONY: setup data features eda train mlflow test lint

setup:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

data:
	bash scripts/download_data.sh

features:
	.venv/bin/python -m fraud.data

eda:
	.venv/bin/python -m fraud.eda

train:
	.venv/bin/python -m fraud.train

mlflow:
	MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests
