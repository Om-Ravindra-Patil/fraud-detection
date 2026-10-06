"""Model definitions: a logistic regression baseline and LightGBM."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Skewed, non-negative columns that a linear model handles better on a log scale
LOG_COLUMNS = {
    "old_balance_dest", "old_balance_orig", "dest_amount_sum_24h", "dest_txn_count_24h",
    "dest_prior_txn_count", "hours_since_prev_dest_txn", "amount_to_orig_balance",
}
# Columns a linear model should treat as categories rather than numbers
ONE_HOT_COLUMNS = {"type", "hour_of_day"}


def build_logistic_regression(columns: list[str], params: dict, seed: int) -> Pipeline:
    """Scaled logistic regression. class_weight='balanced' handles the class imbalance."""
    one_hot = [c for c in columns if c in ONE_HOT_COLUMNS]
    logged = [c for c in columns if c in LOG_COLUMNS]
    plain = [c for c in columns if c not in ONE_HOT_COLUMNS | LOG_COLUMNS]
    impute_and_scale = [SimpleImputer(strategy="median", add_indicator=True), StandardScaler()]
    preprocess = ColumnTransformer([
        ("one_hot", OneHotEncoder(handle_unknown="ignore"), one_hot),
        ("log", make_pipeline(FunctionTransformer(np.log1p), *impute_and_scale), logged),
        ("plain", make_pipeline(*impute_and_scale), plain),
    ])
    model = LogisticRegression(
        C=params["C"], max_iter=params["max_iter"], class_weight="balanced", random_state=seed
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def build_lightgbm(params: dict, seed: int, scale_pos_weight: float = 1.0) -> lgb.LGBMClassifier:
    """LightGBM handles missing values and the categorical 'type' column natively."""
    params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    return lgb.LGBMClassifier(
        **params, scale_pos_weight=scale_pos_weight, random_state=seed, verbose=-1
    )


def to_lightgbm_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select model columns and give 'type' a fixed category set so codes match across splits."""
    X = df[columns].copy()
    if "type" in X:
        X["type"] = pd.Categorical(X["type"], categories=["CASH_OUT", "TRANSFER"])
    return X
