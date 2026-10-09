import json

import numpy as np
import pandas as pd
import pytest

from fraud.drift import build_profile, drift_table, psi, status

rng = np.random.default_rng(0)
REFERENCE = pd.DataFrame({
    "amount": rng.lognormal(10, 1, 20_000),
    "type": rng.choice(["TRANSFER", "CASH_OUT"], 20_000, p=[0.2, 0.8]),
    "flag": rng.integers(0, 2, 20_000),
    "hours": np.where(rng.random(20_000) < 0.3, np.nan, rng.exponential(20, 20_000)),
})
PROFILE = build_profile(REFERENCE, list(REFERENCE.columns))


def test_same_distribution_is_stable():
    fresh = pd.DataFrame({
        "amount": rng.lognormal(10, 1, 5_000),
        "type": rng.choice(["TRANSFER", "CASH_OUT"], 5_000, p=[0.2, 0.8]),
        "flag": rng.integers(0, 2, 5_000),
        "hours": np.where(rng.random(5_000) < 0.3, np.nan, rng.exponential(20, 5_000)),
    })
    assert (drift_table(PROFILE, fresh)["status"] == "stable").all()


def test_shifted_numeric_feature_is_drift():
    shifted = pd.Series(rng.lognormal(11, 1, 5_000))  # amounts about 2.7 times larger
    assert psi(PROFILE["amount"], shifted) > 0.25


def test_category_mix_change_is_detected():
    flipped = pd.Series(rng.choice(["TRANSFER", "CASH_OUT"], 5_000, p=[0.8, 0.2]))
    assert psi(PROFILE["type"], flipped) > 0.25


def test_unseen_category_counts_as_drift():
    assert psi(PROFILE["type"], pd.Series(["PAYMENT"] * 1_000)) > 0.25


def test_change_in_missing_values_is_detected():
    no_missing = pd.Series(rng.exponential(20, 5_000))  # reference had 30% missing
    assert psi(PROFILE["hours"], no_missing) > 0.25


def test_values_outside_training_range_still_land_in_a_bin():
    extreme = pd.Series(np.full(1_000, 1e12))
    assert np.isfinite(psi(PROFILE["amount"], extreme))


def test_low_cardinality_numeric_is_profiled_as_categories():
    assert PROFILE["flag"]["kind"] == "categorical"
    assert PROFILE["amount"]["kind"] == "numeric"


def test_profile_survives_json_round_trip():
    reloaded = json.loads(json.dumps(PROFILE))
    sample = REFERENCE.sample(2_000, random_state=1)
    assert psi(reloaded["amount"], sample["amount"]) == pytest.approx(
        psi(PROFILE["amount"], sample["amount"]))


@pytest.mark.parametrize("value, expected", [(0.05, "stable"), (0.1, "watch"),
                                             (0.25, "watch"), (0.3, "drift")])
def test_status_thresholds(value, expected):
    assert status(value) == expected
