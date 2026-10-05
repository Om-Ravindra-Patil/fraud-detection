"""The feature SQL must only use information available before each transaction."""

import math

import pandas as pd
from conftest import BASE_IDENTITY, BASE_TRANSACTIONS, CARD_A, txn

HISTORY_COLUMNS = [
    "card_txn_count_24h", "card_amount_sum_24h", "card_txn_count_7d",
    "card_prior_txn_count", "card_prior_avg_amount", "secs_since_prev_card_txn",
    "amount_vs_card_avg",
]


def features(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM features").df().set_index("transaction_id").sort_index()


def test_velocity_features_match_hand_calculation(make_db):
    f = features(make_db())

    # First transaction on card A: no history at all
    assert f.loc[1, "card_txn_count_24h"] == 0
    assert f.loc[1, "card_amount_sum_24h"] == 0
    assert pd.isna(f.loc[1, "secs_since_prev_card_txn"])
    assert pd.isna(f.loc[1, "amount_vs_card_avg"])

    # Ids 2 and 3 happen in the same second: each sees only id 1, never each other
    for tid in (2, 3):
        assert f.loc[tid, "card_txn_count_24h"] == 1
        assert f.loc[tid, "card_amount_sum_24h"] == 100
        assert f.loc[tid, "secs_since_prev_card_txn"] == 100

    # Id 4 is 24h + 1s after id 1, so id 1 drops out of the 24h window but not the 7d one
    assert f.loc[4, "card_txn_count_24h"] == 2
    assert f.loc[4, "card_amount_sum_24h"] == 80
    assert f.loc[4, "card_txn_count_7d"] == 3
    assert f.loc[4, "card_prior_txn_count"] == 3
    assert math.isclose(f.loc[4, "card_prior_avg_amount"], 60)
    assert math.isclose(f.loc[4, "amount_vs_card_avg"], 200 / 60)

    # Card B is a different card, so card A's activity must not leak into it
    assert f.loc[5, "card_prior_txn_count"] == 0


def test_simple_features(make_db):
    f = features(make_db())
    assert f.loc[1, "hour_of_day"] == 0  # 86,400s is exactly one day in
    assert f.loc[1, "email_domains_match"] == 1
    assert f.loc[2, "email_domains_match"] == 0
    assert pd.isna(f.loc[3, "email_domains_match"])
    assert f.loc[1, "device_type"] == "desktop"


def test_future_transactions_do_not_change_past_features(make_db):
    before = features(make_db())
    later = txn(6, 172_900, 9_999.0, CARD_A, fraud=1)
    after = features(make_db(transactions=BASE_TRANSACTIONS + [later], identity=BASE_IDENTITY))
    pd.testing.assert_frame_equal(before[HISTORY_COLUMNS], after.loc[before.index, HISTORY_COLUMNS])


def test_labels_do_not_change_features(make_db):
    before = features(make_db())
    flipped = [{**t, "isFraud": 1 - t["isFraud"]} for t in BASE_TRANSACTIONS]
    after = features(make_db(transactions=flipped))
    cols = [c for c in before.columns if c != "is_fraud"]
    pd.testing.assert_frame_equal(before[cols], after[cols])
