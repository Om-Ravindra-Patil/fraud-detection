"""Tiny hand-built version of the IEEE-CIS files, so the SQL can be tested without Kaggle data.

Card A (card1=1000) has four transactions. Two share the same second (ids 2 and 3).
Card B (card1=2000) has one transaction.
"""

import csv
from pathlib import Path

import pytest

from fraud.data import build_database

TXN_COLUMNS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "dist1",
    "P_emaildomain", "R_emaildomain", "V1",
]
ID_COLUMNS = ["TransactionID", "DeviceType", "DeviceInfo", "id_01"]
CARD_A = {"card1": 1000, "card2": 111, "card3": 150, "card4": "visa", "card5": 226,
          "card6": "debit", "addr1": 315}
CARD_B = {**CARD_A, "card1": 2000}


def txn(tid, dt, amt, card, fraud=0, p_email="gmail.com", r_email=""):
    return {"TransactionID": tid, "isFraud": fraud, "TransactionDT": dt, "TransactionAmt": amt,
            "ProductCD": "W", **card, "dist1": "", "P_emaildomain": p_email,
            "R_emaildomain": r_email, "V1": 1}


BASE_TRANSACTIONS = [
    txn(1, 86_400, 100.0, CARD_A, p_email="gmail.com", r_email="gmail.com"),
    txn(2, 86_500, 50.0, CARD_A, fraud=1, p_email="gmail.com", r_email="yahoo.com"),
    txn(3, 86_500, 30.0, CARD_A),
    txn(4, 172_801, 200.0, CARD_A, fraud=1),
    txn(5, 86_450, 10.0, CARD_B),
]
BASE_IDENTITY = [
    {"TransactionID": 1, "DeviceType": "desktop", "DeviceInfo": "Windows", "id_01": -5},
    {"TransactionID": 5, "DeviceType": "mobile", "DeviceInfo": "iOS", "id_01": 0},
]


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def make_db(tmp_path):
    """Return a function that builds a DuckDB database from the given rows."""
    counter = iter(range(1000))

    def _make(transactions=BASE_TRANSACTIONS, identity=BASE_IDENTITY):
        run_dir = tmp_path / f"run{next(counter)}"
        raw = run_dir / "raw"
        raw.mkdir(parents=True)
        _write_csv(raw / "train_transaction.csv", TXN_COLUMNS, transactions)
        _write_csv(raw / "train_identity.csv", ID_COLUMNS, identity)
        return build_database(raw, run_dir / "fraud.duckdb", reference_date="2017-11-30")

    return _make
