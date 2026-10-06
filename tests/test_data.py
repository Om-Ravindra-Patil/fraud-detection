import pytest

from fraud.data import read_named_queries, render_sql


def test_render_sql_fills_placeholders():
    sql = render_sql("01_load_raw.sql", raw_dir="/some/dir", raw_file="paysim.csv")
    assert "/some/dir/paysim.csv" in sql
    assert "{{" not in sql


def test_render_sql_fails_on_missing_parameter():
    with pytest.raises(KeyError, match="raw_file"):
        render_sql("01_load_raw.sql", raw_dir="/some/dir")


def test_named_queries_are_parsed():
    queries = read_named_queries("04_eda.sql")
    assert {"overview", "fraud_by_day", "existing_rule_performance"} <= set(queries)
    for q in queries.values():
        first_code_line = next(ln for ln in q.splitlines() if not ln.startswith("--"))
        assert first_code_line.upper().startswith(("SELECT", "WITH"))


def test_transaction_ids_follow_file_order(make_db):
    con = make_db()
    rows = con.execute("SELECT transaction_id, amount FROM transactions ORDER BY 1").fetchall()
    assert rows == [(1, 100.0), (2, 50.0), (3, 30.0), (4, 200.0), (5, 20.0), (6, 10.0)]


def test_features_keep_only_transfer_and_cash_out(make_db):
    con = make_db()
    types = {t for (t,) in con.execute("SELECT DISTINCT type FROM features").fetchall()}
    n = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    assert types == {"TRANSFER", "CASH_OUT"}
    assert n == 5


def test_post_transaction_balances_never_reach_features(make_db):
    con = make_db()
    cols = {c for (c,) in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'features'"
    ).fetchall()}
    assert not {c for c in cols if "new" in c.lower()}


def test_all_eda_queries_run(make_db):
    con = make_db()
    for query in read_named_queries("04_eda.sql").values():
        con.execute(query).fetchall()  # raises if any query is broken
