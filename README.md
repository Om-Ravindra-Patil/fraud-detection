# Card Fraud Detection (IEEE-CIS)

End-to-end fraud detection project: SQL feature engineering in DuckDB, a time-based
model evaluation, a decision threshold chosen by business cost, SHAP explanations,
LLM-written analyst notes, MLOps tooling and a Streamlit dashboard.

Full write-up to follow. Current status: **Phase 2 (data and SQL features).**

## Getting started

```bash
make setup      # create .venv and install dependencies
make data       # download the Kaggle files (see below first)
make features   # build data/processed/fraud.duckdb and features.parquet
make eda        # run the SQL EDA queries, save results and charts
make test       # run the test suite (uses small synthetic data, no download needed)
```

Before `make data`: accept the competition rules at
https://www.kaggle.com/competitions/ieee-fraud-detection/rules and place your Kaggle
API token at `~/.kaggle/kaggle.json`. The raw data is not committed to this repo.

## Project layout

| Path | Contents |
|---|---|
| `sql/` | All data loading, cleaning, feature and EDA logic, written in SQL |
| `src/fraud/` | Python package that runs the SQL and (later) trains and explains the model |
| `tests/` | pytest suite, including leakage tests for the feature SQL |
| `configs/config.yaml` | Paths and settings |
