"""Time-based train / validation / test split, read from the features table with SQL."""

from pathlib import Path

import duckdb
import pandas as pd

SPLITS = ("train", "valid", "test")


def check_split_config(split_cfg: dict) -> None:
    """Fail if the day ranges are inverted, out of order or overlapping."""
    ranges = [split_cfg[f"{name}_days"] for name in SPLITS]
    for start, end in ranges:
        if start > end:
            raise ValueError(f"Day range {start}-{end} is inverted")
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:], strict=False):
        if next_start <= prev_end:
            raise ValueError("Split day ranges must be in time order and must not overlap")


def load_splits(features_path: Path, split_cfg: dict) -> dict[str, pd.DataFrame]:
    """Return {'train': df, 'valid': df, 'test': df}, each sorted by time."""
    check_split_config(split_cfg)
    con = duckdb.connect()
    splits = {}
    for name in SPLITS:
        start, end = split_cfg[f"{name}_days"]
        splits[name] = con.execute(
            """
            SELECT *
            FROM read_parquet(?)
            WHERE day BETWEEN ? AND ?
            ORDER BY step, transaction_id
            """,
            [features_path.as_posix(), start, end],
        ).df()
    return splits


def feature_columns(features_cfg: dict, feature_set: str) -> list[str]:
    """Model input columns for a named feature set ('realistic' or 'full')."""
    cols = features_cfg["categorical"] + features_cfg["realistic"]
    if feature_set == "full":
        cols = cols + features_cfg["full_extra"]
    elif feature_set != "realistic":
        raise ValueError(f"Unknown feature set: {feature_set}")
    return cols
