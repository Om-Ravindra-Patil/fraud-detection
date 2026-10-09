"""Live scoring must match training: same features as the SQL, same scores as the model."""

import json
import math

import pandas as pd
import pytest

from fraud.config import PROJECT_ROOT, load_config
from fraud.serving import MODEL_FILE, Scorer, build_features
from fraud.split import feature_columns

MODELS = PROJECT_ROOT / "models"
needs_model = pytest.mark.skipif(not (MODELS / MODEL_FILE).exists(),
                                 reason="run make app-data to export the model")


def test_python_features_match_sql_features(make_db):
    """Rebuild every feature row from raw inputs in Python and compare with the SQL output."""
    con = make_db()
    sql = con.execute("""
        SELECT f.*, t.old_balance_orig AS raw_orig
        FROM features f JOIN transactions t USING (transaction_id)
    """).df()
    cols = feature_columns(load_config()["features"], "full")
    for _, r in sql.iterrows():
        payment = {
            "type": r["type"], "amount": r["amount"], "old_balance_orig": r["raw_orig"],
            "old_balance_dest": r["old_balance_dest"], "step": r["step"],
            "dest_txn_count_24h": r["dest_txn_count_24h"],
            "dest_amount_sum_24h": r["dest_amount_sum_24h"],
            "dest_prior_txn_count": r["dest_prior_txn_count"],
            "hours_since_prev_dest_txn": (None if pd.isna(r["hours_since_prev_dest_txn"])
                                          else r["hours_since_prev_dest_txn"]),
        }
        py = build_features(payment)
        for c in cols:
            a, b = py[c], r[c]
            if isinstance(a, str):
                assert a == b, c
            elif pd.isna(b):
                assert math.isnan(a), c
            else:
                assert a == pytest.approx(float(b)), c


@needs_model
def test_contributions_add_up_to_the_score():
    scorer = Scorer(MODELS)
    row = pd.DataFrame([build_features({
        "type": "TRANSFER", "amount": 250_000, "old_balance_orig": 250_000,
        "old_balance_dest": 0, "step": 2, "dest_txn_count_24h": 0, "dest_amount_sum_24h": 0,
        "dest_prior_txn_count": 0, "hours_since_prev_dest_txn": None})])
    prob, contrib = scorer.score(row)
    raw = scorer.booster.predict(row[scorer.columns].assign(
        type=pd.Categorical(row["type"], categories=["CASH_OUT", "TRANSFER"])), raw_score=True)
    base = scorer.booster.predict(row[scorer.columns].assign(
        type=pd.Categorical(row["type"], categories=["CASH_OUT", "TRANSFER"])),
        pred_contrib=True)[0, -1]
    assert contrib.sum() + base == pytest.approx(raw[0])
    assert 0 <= prob[0] <= 1


@needs_model
def test_explain_payment_flags_an_account_draining_transfer():
    result = Scorer(MODELS).explain_payment({
        "type": "TRANSFER", "amount": 250_000, "old_balance_orig": 250_000,
        "old_balance_dest": 0, "step": 2, "dest_txn_count_24h": 0, "dest_amount_sum_24h": 0,
        "dest_prior_txn_count": 0, "hours_since_prev_dest_txn": None})
    assert result["alert"] is True
    assert len(result["reasons"]) == 5
    assert any("empties the sender" in r["fact"] for r in result["reasons"])


@needs_model
def test_app_bundle_matches_decision_and_has_no_account_ids():
    alerts = pd.read_parquet(PROJECT_ROOT / "app" / "data" / "alerts.parquet")
    decision = json.loads((MODELS / "decision.json").read_text())
    assert alerts["risk_score"].min() >= decision["threshold"]
    assert not {"name_orig", "name_dest", "nameOrig", "nameDest"} & set(alerts.columns)
    assert alerts["note"].notna().all()
