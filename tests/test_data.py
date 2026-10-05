import pytest

from fraud.data import read_named_queries, render_sql


def test_render_sql_fills_placeholders():
    sql = render_sql("01_load_raw.sql", raw_dir="/some/dir")
    assert "/some/dir/train_transaction.csv" in sql
    assert "{{" not in sql


def test_render_sql_fails_on_missing_parameter():
    with pytest.raises(KeyError, match="raw_dir"):
        render_sql("01_load_raw.sql")


def test_named_queries_are_parsed():
    queries = read_named_queries("04_eda.sql")
    assert {"overview", "fraud_by_week", "fraud_by_card_velocity"} <= set(queries)
    for q in queries.values():
        first_code_line = next(ln for ln in q.splitlines() if not ln.startswith("--"))
        assert first_code_line.upper().startswith(("SELECT", "WITH"))


def test_identity_join_keeps_one_row_per_transaction(make_db):
    con = make_db()
    n_raw = con.execute("SELECT COUNT(*) FROM raw_transaction").fetchone()[0]
    n_txn, n_unique = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT transaction_id) FROM transactions"
    ).fetchone()
    n_feat = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    assert n_raw == n_txn == n_unique == n_feat == 5


def test_identity_fields(make_db):
    con = make_db()
    rows = dict(con.execute("SELECT transaction_id, has_identity FROM transactions").fetchall())
    assert rows == {1: 1, 2: 0, 3: 0, 4: 0, 5: 1}


def test_all_eda_queries_run(make_db):
    con = make_db()
    for query in read_named_queries("04_eda.sql").values():
        con.execute(query).fetchall()  # raises if any query is broken
