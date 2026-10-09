"""Shared DataFrame helpers, kept free of heavy imports for the dashboard container."""

import pandas as pd


def to_lightgbm_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select model columns and give 'type' a fixed category set so codes match across splits."""
    X = df[columns].copy()
    if "type" in X:
        X["type"] = pd.Categorical(X["type"], categories=["CASH_OUT", "TRANSFER"])
    return X
