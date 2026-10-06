import numpy as np
import pandas as pd
import pytest

from fraud.config import load_config
from fraud.metrics import evaluate, precision_recall_at, recall_at_precision
from fraud.split import feature_columns
from fraud.train import fit_lightgbm, fit_logreg


def test_precision_recall_at_threshold():
    y = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.3, 0.8, 0.1, 0.2])
    assert precision_recall_at(y, score, 0.5) == (0.5, 0.5)
    assert precision_recall_at(y, score, 0.95) == (0.0, 0.0)  # nothing flagged


def test_recall_at_precision():
    y = np.array([1, 1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    assert recall_at_precision(y, score, 1.0) == pytest.approx(2 / 3)
    assert recall_at_precision(y, score, 0.75) == 1.0


def test_evaluate_perfect_ranking():
    m = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["fraud_rate"] == 0.5


@pytest.fixture
def toy_splits():
    """Small random data with the real column names, where fraud depends on two features."""
    rng = np.random.default_rng(0)

    def make(n):
        df = pd.DataFrame({
            "type": rng.choice(["TRANSFER", "CASH_OUT"], n),
            "log_amount": rng.normal(10, 2, n),
            "hour_of_day": rng.integers(0, 24, n),
            "empties_orig_account": rng.integers(0, 2, n),
            "orig_balance_zero": rng.integers(0, 2, n),
            "dest_balance_zero": rng.integers(0, 2, n),
            "old_balance_dest": rng.exponential(1e5, n),
            "dest_txn_count_24h": rng.poisson(2, n),
            "dest_amount_sum_24h": rng.exponential(1e5, n),
            "dest_prior_txn_count": rng.poisson(5, n),
            "hours_since_prev_dest_txn": np.where(rng.random(n) < 0.3, np.nan,
                                                  rng.exponential(20, n)),
        })
        logit = -4 + 2 * df["empties_orig_account"] + 0.5 * (df["log_amount"] - 10)
        df["is_fraud"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        return df

    return make(4000), make(1000)


def test_both_models_train_and_beat_random(toy_splits):
    train, valid = toy_splits
    cfg = load_config()
    cfg["model"]["lightgbm"]["n_estimators"] = 50
    columns = feature_columns(cfg["features"], "realistic")
    for fit in (fit_logreg, lambda *a: fit_lightgbm(*a, weighting="none")):
        _, scores, _ = fit(train, valid, columns, cfg)
        assert scores.shape == (len(valid),)
        assert np.all((scores >= 0) & (scores <= 1))
        assert evaluate(valid["is_fraud"], scores)["roc_auc"] > 0.7
