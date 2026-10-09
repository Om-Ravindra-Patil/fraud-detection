.PHONY: setup data features eda train evaluate explain fairness notes drift app-data app mlflow test lint

setup:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev,app]"

data:
	bash scripts/download_data.sh

features:
	.venv/bin/python -m fraud.data

eda:
	.venv/bin/python -m fraud.eda

train:
	.venv/bin/python -m fraud.train

evaluate:
	.venv/bin/python -m fraud.evaluate

explain:
	.venv/bin/python -m fraud.explain

fairness:
	.venv/bin/python -m fraud.fairness

notes:
	.venv/bin/python -m fraud.notes

drift:
	.venv/bin/python -m fraud.drift

app-data:
	.venv/bin/python -m fraud.app_data

app:
	.venv/bin/streamlit run app/streamlit_app.py --server.port 8501

mlflow:
	MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests
