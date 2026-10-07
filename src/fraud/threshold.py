"""Choose the alert threshold from business costs and the fraud team's review capacity.

Total cost at a threshold =
    money lost to fraud we did not alert on
  + review cost for every alert raised (true or false)
"""

import numpy as np
import pandas as pd


def cost_at_threshold(y, score, amount, threshold: float, review_cost: float,
                      loss_rate: float = 1.0) -> dict:
    y, score, amount = np.asarray(y), np.asarray(score), np.asarray(amount)
    alert = score >= threshold
    fraud = y == 1
    tp, fp = int(np.sum(alert & fraud)), int(np.sum(alert & ~fraud))
    fn = int(np.sum(~alert & fraud))
    missed_loss = float(amount[~alert & fraud].sum() * loss_rate)
    review_total = float(alert.sum() * review_cost)
    return {
        "threshold": float(threshold),
        "alerts": tp + fp,
        "true_alerts": tp,
        "false_alerts": fp,
        "missed_frauds": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fraud_money_caught": float(amount[alert & fraud].sum() / max(amount[fraud].sum(), 1e-9)),
        "missed_fraud_loss": missed_loss,
        "review_cost": review_total,
        "total_cost": missed_loss + review_total,
    }


def candidate_thresholds(score, n: int = 2000) -> np.ndarray:
    """Score quantiles, plus a threshold above every score (alert on nothing)."""
    qs = np.unique(np.quantile(np.asarray(score), np.linspace(0, 1, n + 1)))
    return np.append(qs, np.nextafter(qs.max(), np.inf))


def cost_curve(y, score, amount, review_cost: float, loss_rate: float = 1.0,
               n_days: int = 1) -> pd.DataFrame:
    rows = [cost_at_threshold(y, score, amount, t, review_cost, loss_rate)
            for t in candidate_thresholds(score)]
    curve = pd.DataFrame(rows)
    curve["alerts_per_day"] = curve["alerts"] / n_days
    return curve


def choose_threshold(curve: pd.DataFrame, daily_capacity: float | None = None) -> pd.Series:
    """Cheapest threshold whose alert volume fits within the team's daily capacity."""
    feasible = curve if daily_capacity is None else curve[curve["alerts_per_day"] <= daily_capacity]
    if feasible.empty:
        raise ValueError("No threshold keeps alerts within capacity")
    # Ties go to the higher threshold, which raises fewer alerts
    best = feasible.sort_values(["total_cost", "threshold"], ascending=[True, False]).iloc[0]
    return best
