"""The split must be strictly time-ordered, and leaky columns must never be model inputs."""

import numpy as np
import pandas as pd
import pytest

from fraud.config import load_config
from fraud.split import check_split_config, feature_columns, load_splits

SPLIT = {"train_days": [0, 19], "valid_days": [20, 24], "test_days": [25, 29]}


@pytest.fixture
def features_parquet(tmp_path):
    steps = np.arange(0, 31 * 24)  # days 0 to 30, one row per hour
    df = pd.DataFrame({
        "transaction_id": steps + 1, "step": steps, "day": steps // 24,
        "is_fraud": (steps % 7 == 0).astype(int),
    })
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    return path


def test_splits_are_time_ordered_and_disjoint(features_parquet):
    s = load_splits(features_parquet, SPLIT)
    assert s["train"]["step"].max() < s["valid"]["step"].min()
    assert s["valid"]["step"].max() < s["test"]["step"].min()
    ids = [set(s[name]["transaction_id"]) for name in ("train", "valid", "test")]
    assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])


def test_day_30_artifact_is_excluded(features_parquet):
    s = load_splits(features_parquet, SPLIT)
    assert max(df["day"].max() for df in s.values()) == 29


@pytest.mark.parametrize("bad", [
    {**SPLIT, "valid_days": [19, 24]},   # overlaps train
    {**SPLIT, "test_days": [10, 15]},    # earlier than validation
    {**SPLIT, "train_days": [19, 0]},    # inverted
])
def test_bad_split_config_is_rejected(bad):
    with pytest.raises(ValueError):
        check_split_config(bad)


def test_project_config_is_valid():
    check_split_config(load_config()["split"])


def test_feature_sets_exclude_labels_ids_and_post_payment_data():
    cfg = load_config()["features"]
    forbidden = {"is_fraud", "is_flagged_fraud", "transaction_id", "step", "day"}
    for name in ("realistic", "full"):
        cols = set(feature_columns(cfg, name))
        assert not cols & forbidden
        assert not any("new" in c for c in cols)


def test_realistic_set_excludes_the_balance_artifact():
    cfg = load_config()["features"]
    realistic = set(feature_columns(cfg, "realistic"))
    assert not realistic & {"old_balance_orig", "amount_to_orig_balance"}
    assert set(feature_columns(cfg, "full")) - realistic == set(cfg["full_extra"])


def test_unknown_feature_set_is_rejected():
    with pytest.raises(ValueError):
        feature_columns(load_config()["features"], "everything")
