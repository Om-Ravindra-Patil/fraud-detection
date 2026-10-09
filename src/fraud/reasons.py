"""Plain-English reasons from per-feature contributions (SHAP values).

Kept free of heavy imports (no shap, no duckdb) so the dashboard container can use it.
"""

import numpy as np
import pandas as pd

TOP_N_REASONS = 5

FEATURE_LABELS = {
    "type": "Payment type",
    "log_amount": "Payment amount",
    "hour_of_day": "Hour of day",
    "empties_orig_account": "Empties sender's account",
    "orig_balance_zero": "Sender's balance was zero",
    "dest_balance_zero": "Receiving account's balance was zero",
    "old_balance_dest": "Receiving account's balance",
    "dest_txn_count_24h": "Payments into receiving account, last 24h",
    "dest_amount_sum_24h": "Money into receiving account, last 24h",
    "dest_prior_txn_count": "Earlier payments into receiving account",
    "hours_since_prev_dest_txn": "Hours since receiving account last got money",
}


def describe(feature: str, row: pd.Series) -> str:
    """One plain-English fact about a feature's value for one transaction."""
    v = row[feature]
    missing = bool(pd.isna(v))
    match feature:
        case "type":
            return "Payment is a transfer" if v == "TRANSFER" else "Payment is a cash-out"
        case "log_amount":
            return f"Amount is {row['amount']:,.0f}"
        case "hour_of_day":
            return f"Made at {int(v):02d}:00 on the simulation clock"
        case "empties_orig_account":
            return ("Payment empties the sender's account" if v
                    else "Sender keeps money after the payment")
        case "orig_balance_zero":
            return ("Sender's balance was zero before the payment" if v
                    else "Sender had money before the payment")
        case "dest_balance_zero":
            return ("Receiving account held no money before the payment" if v
                    else "Receiving account already held money")
        case "old_balance_dest":
            return f"Receiving account held {v:,.0f} before the payment"
        case "dest_txn_count_24h":
            return f"Receiving account got {int(v)} payment(s) in the previous 24 hours"
        case "dest_amount_sum_24h":
            return f"Receiving account got {v:,.0f} in the previous 24 hours"
        case "dest_prior_txn_count":
            return ("First payment ever seen into this receiving account" if v == 0
                    else f"Receiving account has {int(v)} earlier payment(s)")
        case "hours_since_prev_dest_txn":
            return ("No earlier payment into this receiving account" if missing
                    else f"{int(v)} hours since the receiving account last got money")
    return f"{feature} = {v}"


def top_reasons(row: pd.Series, contributions: np.ndarray, columns: list[str],
                n: int = TOP_N_REASONS) -> list[dict]:
    """The n features that moved this transaction's score the most, either way."""
    order = np.argsort(-np.abs(contributions))[:n]
    return [{
        "feature": columns[i],
        "label": FEATURE_LABELS.get(columns[i], columns[i]),
        "fact": describe(columns[i], row),
        "shap": round(float(contributions[i]), 4),
        "direction": "raises risk" if contributions[i] > 0 else "lowers risk",
    } for i in order]
