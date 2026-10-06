"""Build the DuckDB database from the raw Kaggle CSVs by running the files in sql/."""

import re
from pathlib import Path

import duckdb

from fraud.config import SQL_DIR, load_config

PIPELINE = ["01_load_raw.sql", "02_transactions.sql", "03_features.sql"]
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render_sql(filename: str, **params) -> str:
    """Read a SQL file and fill in {{ name }} placeholders. Fails loudly on a missing one."""
    sql = (SQL_DIR / filename).read_text()

    def substitute(match: re.Match) -> str:
        key = match.group(1)
        if key not in params:
            raise KeyError(f"{filename} needs a value for '{key}'")
        return str(params[key])

    return _PLACEHOLDER.sub(substitute, sql)


def read_named_queries(filename: str) -> dict[str, str]:
    """Split a SQL file into {name: query} using '-- name: <name>' markers."""
    queries: dict[str, str] = {}
    name = None
    for line in (SQL_DIR / filename).read_text().splitlines():
        marker = re.match(r"--\s*name:\s*(\w+)", line)
        if marker:
            name = marker.group(1)
            queries[name] = ""
        elif name:
            queries[name] += line + "\n"
    return {k: v.strip() for k, v in queries.items()}


def build_database(raw_dir: Path, db_path: Path, raw_file: str) -> duckdb.DuckDBPyConnection:
    """Run the SQL pipeline and return an open connection to the finished database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    params = {"raw_dir": raw_dir.as_posix(), "raw_file": raw_file}
    for filename in PIPELINE:
        con.execute(render_sql(filename, **params))
    return con


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    raw_file = cfg["data"]["raw_file"]
    if not (paths["raw_dir"] / raw_file).exists():
        raise SystemExit(
            f"Missing {raw_file} in {paths['raw_dir']}. "
            "Run 'make data' first (see README for Kaggle setup)."
        )
    con = build_database(paths["raw_dir"], paths["duckdb"], raw_file)
    con.execute(f"COPY features TO '{paths['features_parquet'].as_posix()}' (FORMAT parquet)")

    for table in ["raw_paysim", "transactions", "features"]:
        rows, cols = con.execute(
            f"SELECT (SELECT COUNT(*) FROM {table}), "
            f"(SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{table}')"
        ).fetchone()
        print(f"{table:<16} {rows:>9,} rows  {cols:>4} columns")
    print(f"Features written to {paths['features_parquet']}")


if __name__ == "__main__":
    main()
