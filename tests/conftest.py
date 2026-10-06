"""Tiny hand-built version of the PaySim file, so the SQL can be tested without Kaggle data.

Destination account D1 receives four transfers. Ids 2 and 3 happen in the same hour (step 2).
Id 5 is a PAYMENT to a merchant, which the model table should exclude.
Id 6 sends money to a different account, D2.
"""

import csv
from pathlib import Path

import pytest

from fraud.data import build_database

COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def txn(step, amount, orig, dest, type_="TRANSFER", old_orig=1000.0, old_dest=0.0, fraud=0):
    return {"step": step, "type": type_, "amount": amount, "nameOrig": orig,
            "oldbalanceOrg": old_orig, "newbalanceOrig": max(old_orig - amount, 0),
            "nameDest": dest, "oldbalanceDest": old_dest, "newbalanceDest": old_dest + amount,
            "isFraud": fraud, "isFlaggedFraud": 0}


# File order defines transaction_id: 1, 2, 3, ...
BASE_ROWS = [
    txn(1, 100.0, "C1", "D1"),
    txn(2, 50.0, "C2", "D1", fraud=1, old_orig=50.0),
    txn(2, 30.0, "C3", "D1", type_="CASH_OUT", old_orig=0.0),
    txn(26, 200.0, "C1", "D1", fraud=1, old_dest=500.0),
    txn(26, 20.0, "C4", "M1", type_="PAYMENT"),
    txn(27, 10.0, "C5", "D2"),
]


def write_paysim(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def make_db(tmp_path):
    """Return a function that builds a DuckDB database from the given rows."""
    counter = iter(range(1000))

    def _make(rows=BASE_ROWS):
        run_dir = tmp_path / f"run{next(counter)}"
        raw = run_dir / "raw"
        raw.mkdir(parents=True)
        write_paysim(raw / "paysim.csv", rows)
        return build_database(raw, run_dir / "fraud.duckdb", raw_file="paysim.csv")

    return _make
