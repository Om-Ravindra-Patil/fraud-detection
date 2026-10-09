"""Data drift monitoring with the Population Stability Index (PSI).

A reference profile of the training data (bin edges and the share of rows in each bin) is saved
once to models/reference_profile.json. Any later batch can then be checked against it without
the training data.

PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%).
Usual reading in credit and fraud model monitoring: < 0.1 stable, 0.1 to 0.25 watch, > 0.25 drift.
"""

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

N_BINS = 10
MAX_CATEGORIES = 10  # numeric columns with this many distinct values or fewer are binned by value
EPS = 1e-4           # floor on bin shares, so an empty bin does not give log(0)
WATCH, DRIFT = 0.1, 0.25
MISSING = "__missing__"


def status(psi: float) -> str:
    return "drift" if psi > DRIFT else "watch" if psi >= WATCH else "stable"


def _profile_column(values: pd.Series) -> dict:
    present = values.dropna()
    missing_share = 1 - len(present) / len(values) if len(values) else 0.0
    is_numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    if not is_numeric or present.nunique() <= MAX_CATEGORIES:
        shares = present.astype(str).value_counts(normalize=True) * (1 - missing_share)
        return {"kind": "categorical", "shares": shares.round(6).to_dict(),
                "missing": round(missing_share, 6)}
    edges = np.unique(np.quantile(present.astype(float), np.linspace(0, 1, N_BINS + 1)))
    inner = edges[1:-1].tolist()  # outer bins are open-ended, so new extremes still land in a bin
    counts = np.bincount(np.searchsorted(inner, present.astype(float), side="right"),
                         minlength=len(inner) + 1)
    return {"kind": "numeric", "edges": inner,
            "shares": (counts / len(values)).round(6).tolist(), "missing": round(missing_share, 6)}


def build_profile(df: pd.DataFrame, columns: list[str]) -> dict:
    return {c: _profile_column(df[c]) for c in columns}


def _actual_shares(entry: dict, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Expected and actual shares over the same bins, including a missing-value bin."""
    n = max(len(values), 1)
    present = values.dropna()
    missing_actual = 1 - len(present) / n
    if entry["kind"] == "categorical":
        cats = list(entry["shares"])
        actual_counts = present.astype(str).value_counts()
        expected = [entry["shares"][c] for c in cats]
        actual = [actual_counts.get(c, 0) / n for c in cats]
        # Categories never seen in training form one extra bin
        expected.append(0.0)
        actual.append(actual_counts[~actual_counts.index.isin(cats)].sum() / n)
    else:
        counts = np.bincount(np.searchsorted(entry["edges"], present.astype(float), side="right"),
                             minlength=len(entry["edges"]) + 1)
        expected, actual = list(entry["shares"]), list(counts / n)
    expected.append(entry["missing"])
    actual.append(missing_actual)
    return np.array(expected, dtype=float), np.array(actual, dtype=float)


def psi(entry: dict, values: pd.Series) -> float:
    expected, actual = _actual_shares(entry, values)
    keep = (expected > 0) | (actual > 0)
    e = np.clip(expected[keep], EPS, None)
    a = np.clip(actual[keep], EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_table(profile: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"feature": c, "psi": round(psi(entry, df[c]), 4)} for c, entry in profile.items()]
    table = pd.DataFrame(rows)
    table["status"] = table["psi"].map(status)
    return table.sort_values("psi", ascending=False).reset_index(drop=True)


def plot_daily(daily: pd.DataFrame, path) -> None:
    """Two stacked panels sharing the day axis: score drift on top, alert rate below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    top.plot(daily["day"], daily["psi_risk_score"], color=series, linewidth=2, marker="o",
             markersize=4)
    for level, label in ((WATCH, "watch"), (DRIFT, "drift")):
        top.axhline(level, color=muted, linestyle="--", linewidth=1)
        top.text(daily["day"].min(), level, f" {label} ({level})", color=muted, fontsize=8,
                 va="bottom")
    top.set_title("Drift in the model's risk score vs training (PSI)", loc="left", fontsize=11,
                  color=ink)
    bottom.plot(daily["day"], daily["alert_rate"] * 100, color=series, linewidth=2, marker="o",
                markersize=4)
    bottom.set_title("Share of payments raising an alert (%)", loc="left", fontsize=11, color=ink)
    bottom.set_xlabel("Day", color=muted)
    for ax in (top, bottom):
        ax.axvline(16.5, color=muted, linewidth=1)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
    top.text(16.6, top.get_ylim()[1] * 0.9, "genuine volume\ncollapses (day 17)", color=muted,
             fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import joblib

    from fraud.config import PROJECT_ROOT, load_config
    from fraud.models import to_lightgbm_frame
    from fraud.split import load_splits

    cfg = load_config()
    decision = json.loads((PROJECT_ROOT / "models" / "decision.json").read_text())
    bundle = joblib.load(PROJECT_ROOT / "models" / f"{decision['model']}.joblib")
    model, columns = bundle["model"], bundle["columns"]
    splits = load_splits(cfg["paths"]["features_parquet"], cfg["split"])
    for df in splits.values():
        df["risk_score"] = model.predict_proba(to_lightgbm_frame(df, columns))[:, 1]

    monitored = columns + ["risk_score"]
    profile = build_profile(splits["train"], monitored)
    (PROJECT_ROOT / "models" / "reference_profile.json").write_text(json.dumps(profile, indent=1))

    # Drift of each later period against training
    by_period = {name: drift_table(profile, splits[name]) for name in ("valid", "test")}
    table = by_period["valid"][["feature", "psi"]].rename(columns={"psi": "psi_valid"}).merge(
        by_period["test"][["feature", "psi", "status"]].rename(
            columns={"psi": "psi_test", "status": "status_test"}), on="feature")
    table = table.sort_values("psi_test", ascending=False)

    # Day by day, for the score and the most drifted inputs: what a daily monitoring job would see
    later = pd.concat([splits["train"][splits["train"]["day"] >= 14], splits["valid"],
                       splits["test"]])
    watch = ["risk_score"] + [f for f in table["feature"] if f != "risk_score"][:3]
    daily = []
    for day, g in later.groupby("day"):
        row = {"day": int(day), "payments": len(g),
               "alert_rate": round(float((g["risk_score"] >= decision["threshold"]).mean()), 4)}
        row.update({f"psi_{f}": round(psi(profile[f], g[f]), 4) for f in watch})
        daily.append(row)
    daily = pd.DataFrame(daily)

    reports = PROJECT_ROOT / "reports"
    table.to_csv(reports / "drift_by_period.csv", index=False)
    daily.to_csv(reports / "drift_daily.csv", index=False)
    plot_daily(daily, reports / "figures" / "drift_daily.png")
    (PROJECT_ROOT / "app" / "data" / "drift.json").write_text(json.dumps(
        {"thresholds": {"watch": WATCH, "drift": DRIFT},
         "by_period": table.to_dict("records"), "daily": daily.to_dict("records"),
         "watched": watch}, indent=1))

    print("PSI against the training period (days 0 to 19):")
    print(table.to_string(index=False))
    print("\nDaily view (days 14 onwards):")
    print(daily.to_string(index=False))
    drifted = table[table["status_test"] == "drift"]["feature"].tolist()
    print(f"\nFeatures drifting in the test period: {drifted or 'none'}")


if __name__ == "__main__":
    main()
