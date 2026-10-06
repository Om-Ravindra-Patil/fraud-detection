# Payment Fraud Detection (PaySim)

End-to-end fraud detection project: SQL feature engineering in DuckDB, a time-based
model evaluation, a decision threshold chosen by business cost, SHAP explanations,
LLM-written analyst notes, MLOps tooling and a Streamlit dashboard.

Full write-up to follow. Current status: **Phase 2 (data and SQL features).**

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
make test       # run the test suite (uses small synthetic data, no download needed)
```

Before `make data`, log in to Kaggle once with `.venv/bin/kaggle auth login`.
The raw data is not committed to this repo.

## Project layout

| Path | Contents |
|---|---|
| `sql/` | All data loading, cleaning, feature and EDA logic, written in SQL |
| `src/fraud/` | Python package that runs the SQL and (later) trains and explains the model |
| `tests/` | pytest suite, including leakage tests for the feature SQL |
| `configs/config.yaml` | Paths and settings |
