"""The feature SQL must only use information available before each transaction."""

import pandas as pd
from conftest import BASE_ROWS, txn

HISTORY_COLUMNS = [
    "dest_txn_count_24h", "dest_amount_sum_24h", "dest_prior_txn_count",
    "hours_since_prev_dest_txn", "orig_prior_txn_count",
]


def features(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM features").df().set_index("transaction_id").sort_index()


def test_history_features_match_hand_calculation(make_db):
    f = features(make_db())

    # First transfer into D1: no history at all
    assert f.loc[1, "dest_txn_count_24h"] == 0
    assert f.loc[1, "dest_amount_sum_24h"] == 0
    assert pd.isna(f.loc[1, "hours_since_prev_dest_txn"])

    # Ids 2 and 3 happen in the same hour: each sees only id 1, never each other
    for tid in (2, 3):
        assert f.loc[tid, "dest_txn_count_24h"] == 1
        assert f.loc[tid, "dest_amount_sum_24h"] == 100
        assert f.loc[tid, "hours_since_prev_dest_txn"] == 1

    # Id 4 is at step 26, so step 1 falls outside the 24-hour window but counts as history
    assert f.loc[4, "dest_txn_count_24h"] == 2
    assert f.loc[4, "dest_amount_sum_24h"] == 80
    assert f.loc[4, "dest_prior_txn_count"] == 3
    assert f.loc[4, "hours_since_prev_dest_txn"] == 24
    assert f.loc[4, "orig_prior_txn_count"] == 1  # C1 sent id 1 earlier

    # D2 is a different account, so D1's activity must not leak into it
    assert f.loc[6, "dest_prior_txn_count"] == 0


def test_balance_features(make_db):
    f = features(make_db())
    assert f.loc[1, "empties_orig_account"] == 0  # 100 out of 1000
    assert f.loc[2, "empties_orig_account"] == 1  # 50 out of 50
    assert f.loc[3, "orig_balance_zero"] == 1
    assert pd.isna(f.loc[3, "amount_to_orig_balance"])  # no division by zero
    assert f.loc[1, "dest_balance_zero"] == 1
    assert f.loc[4, "dest_balance_zero"] == 0
    assert f.loc[4, "hour_of_day"] == 2 and f.loc[4, "day"] == 1


def test_future_transactions_do_not_change_past_features(make_db):
    before = features(make_db())
    later = txn(27, 9_999.0, "C9", "D1", fraud=1)
    after = features(make_db(BASE_ROWS + [later]))
    pd.testing.assert_frame_equal(before[HISTORY_COLUMNS], after.loc[before.index, HISTORY_COLUMNS])


def test_labels_do_not_change_features(make_db):
    before = features(make_db())
    flipped = [{**r, "isFraud": 1 - r["isFraud"]} for r in BASE_ROWS]
    after = features(make_db(flipped))
    cols = [c for c in before.columns if c != "is_fraud"]
    pd.testing.assert_frame_equal(before[cols], after[cols])
