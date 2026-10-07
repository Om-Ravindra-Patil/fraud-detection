import numpy as np
import pandas as pd
import pytest

from fraud.threshold import choose_threshold, cost_at_threshold, cost_curve

# Two frauds (amounts 1,000 and 50) and three genuine payments
Y = np.array([1, 1, 0, 0, 0])
SCORE = np.array([0.9, 0.4, 0.6, 0.2, 0.1])
AMOUNT = np.array([1_000.0, 50.0, 300.0, 80.0, 20.0])


def test_cost_counts_missed_fraud_and_every_alert():
    r = cost_at_threshold(Y, SCORE, AMOUNT, threshold=0.5, review_cost=10)
    # Alerts on 0.9 (fraud) and 0.6 (genuine); misses the 50 fraud
    assert (r["true_alerts"], r["false_alerts"], r["missed_frauds"]) == (1, 1, 1)
    assert r["missed_fraud_loss"] == 50
    assert r["review_cost"] == 20
    assert r["total_cost"] == 70
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert r["fraud_money_caught"] == pytest.approx(1000 / 1050)


def test_loss_rate_scales_missed_fraud():
    r = cost_at_threshold(Y, SCORE, AMOUNT, threshold=0.95, review_cost=10, loss_rate=0.5)
    assert r["alerts"] == 0
    assert r["total_cost"] == 525


def test_cheap_reviews_mean_more_alerts():
    cheap = choose_threshold(cost_curve(Y, SCORE, AMOUNT, review_cost=1))
    pricey = choose_threshold(cost_curve(Y, SCORE, AMOUNT, review_cost=600))
    assert cheap["recall"] == 1.0  # catching the 50 fraud is worth a few cheap reviews
    assert pricey["alerts"] <= cheap["alerts"]
    assert pricey["threshold"] >= cheap["threshold"]


def test_capacity_limits_alert_volume():
    curve = cost_curve(Y, SCORE, AMOUNT, review_cost=1, n_days=1)
    unconstrained = choose_threshold(curve)
    capped = choose_threshold(curve, daily_capacity=1)
    assert unconstrained["alerts_per_day"] > 1
    assert capped["alerts_per_day"] <= 1
    assert capped["true_alerts"] == 1  # the single alert goes to the highest score


def test_curve_includes_alerting_on_nothing():
    curve = cost_curve(Y, SCORE, AMOUNT, review_cost=10)
    assert (curve["alerts"] == 0).any()
    assert curve.loc[curve["alerts"] == 0, "total_cost"].iloc[0] == 1_050


def test_impossible_capacity_raises():
    curve = pd.DataFrame({"alerts_per_day": [5.0], "total_cost": [1.0], "threshold": [0.5]})
    with pytest.raises(ValueError):
        choose_threshold(curve, daily_capacity=1)
