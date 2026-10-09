"""Score a single payment in the dashboard, without the training stack.

build_features() mirrors sql/03_features.sql for one payment. tests/test_serving.py checks
the two give the same values, so live scores cannot silently drift from training.
Per-feature contributions come from LightGBM's built-in TreeSHAP (pred_contrib=True), the
same algorithm as shap.TreeExplainer, so the container does not need the shap package.
"""

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud.models import to_lightgbm_frame
from fraud.reasons import top_reasons

MODEL_FILE = "fraud_model.txt"
DECISION_FILE = "decision.json"


def build_features(payment: dict) -> dict:
    """Model features for one payment. Keys of `payment`:

    type, amount, old_balance_orig, old_balance_dest, step, and the receiving account's
    history: dest_txn_count_24h, dest_amount_sum_24h, dest_prior_txn_count,
    hours_since_prev_dest_txn (None if it has never received money).
    """
    amount = float(payment["amount"])
    orig = float(payment["old_balance_orig"])
    dest = float(payment["old_balance_dest"])
    return {
        "type": payment["type"],
        "amount": amount,
        "log_amount": math.log1p(amount),
        "hour_of_day": int(payment["step"]) % 24,
        "empties_orig_account": int(orig > 0 and amount >= orig),
        "orig_balance_zero": int(orig == 0),
        "dest_balance_zero": int(dest == 0),
        "old_balance_dest": dest,
        "old_balance_orig": orig,
        "amount_to_orig_balance": amount / orig if orig != 0 else np.nan,
        "dest_txn_count_24h": int(payment["dest_txn_count_24h"]),
        "dest_amount_sum_24h": float(payment["dest_amount_sum_24h"]),
        "dest_prior_txn_count": int(payment["dest_prior_txn_count"]),
        "hours_since_prev_dest_txn": (np.nan if payment.get("hours_since_prev_dest_txn") is None
                                      else float(payment["hours_since_prev_dest_txn"])),
    }


class Scorer:
    """Loads the exported LightGBM model and the frozen alert threshold."""

    def __init__(self, models_dir: Path):
        self.booster = lgb.Booster(model_file=str(models_dir / MODEL_FILE))
        self.columns = self.booster.feature_name()
        self.decision = json.loads((models_dir / DECISION_FILE).read_text())
        self.threshold = float(self.decision["threshold"])

    def score(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Risk scores and per-feature contributions (last column of pred_contrib is the base)."""
        X = to_lightgbm_frame(rows, self.columns)
        probs = self.booster.predict(X)
        contrib = self.booster.predict(X, pred_contrib=True)[:, :-1]
        return probs, contrib

    def explain_payment(self, payment: dict) -> dict:
        row = pd.DataFrame([build_features(payment)])
        probs, contrib = self.score(row)
        return {
            "risk_score": float(probs[0]),
            "alert": bool(probs[0] >= self.threshold),
            "reasons": top_reasons(row.iloc[0], contrib[0], self.columns),
        }
