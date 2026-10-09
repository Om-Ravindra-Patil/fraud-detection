# Payment Fraud Detection (PaySim)

End-to-end fraud detection project: SQL feature engineering in DuckDB, a time-based
model evaluation, a decision threshold chosen by business cost, SHAP explanations,
LLM-written analyst notes, MLOps tooling and a Streamlit dashboard.

Full write-up to follow. Current status: **Phase 6 (dashboard).** See
[docs/threshold_decision.md](docs/threshold_decision.md) and [docs/responsible_ai.md](docs/responsible_ai.md).

## Data

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (licence CC BY-SA 4.0) is a
synthetic mobile money dataset generated from a month of real transaction logs from an
African mobile money service. It has 6.3 million transactions over 744 hourly steps.

Two scoping decisions, both made to keep results honest:

- **Only TRANSFER and CASH_OUT transactions are modelled.** All fraud in PaySim falls in
  these two types.
- **Post-transaction balances are dropped.** `newbalanceOrig` and `newbalanceDest` describe
  the account after the payment, and the simulator's quirks in these columns give fraud away
  almost perfectly. A bank scoring a payment before approving it would not have them, so
  only pre-payment balances are used.

## Getting started

```bash
make setup      # create .venv and install dependencies
make data       # download PaySim from Kaggle (log in once first, see below)
make features   # build data/processed/fraud.duckdb and features.parquet
make eda        # run the SQL EDA queries, save results and charts
make train      # train and compare models on the time-based split, logged to MLflow
make evaluate   # choose the alert threshold on validation, then score the test period once
make explain    # SHAP: overall importance and the top reasons for each flagged payment
make fairness   # false alert rates by payment type, time of day and amount
make notes      # LLM analyst notes via a local Ollama model (falls back to a template)
make app-data   # export the model and build the small data bundle the dashboard reads
make app        # run the Streamlit dashboard at http://localhost:8501
make mlflow     # open the MLflow UI at http://127.0.0.1:5001
make test       # run the test suite (uses small synthetic data, no download needed)
```

Before `make data`, log in to Kaggle once with `.venv/bin/kaggle auth login`.
Before `make notes`, install Ollama and pull the model:
`brew install ollama && brew services start ollama && ollama pull llama3.2:3b`.
The raw data is not committed to this repo.

## Project layout

| Path | Contents |
|---|---|
| `sql/` | All data loading, cleaning, feature and EDA logic, written in SQL |
| `src/fraud/` | Python package that runs the SQL and (later) trains and explains the model |
| `tests/` | pytest suite, including leakage tests for the feature SQL |
| `app/` | Streamlit dashboard and its committed data bundle (`app/data/`) |
| `models/` | Exported model (`fraud_model.txt`) and frozen alert threshold (`decision.json`) |
| `configs/config.yaml` | Paths and settings |
