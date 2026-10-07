"""SHAP explanations: overall feature importance and per-transaction reasons.

SHAP values here are in log-odds: positive values push a payment towards "fraud",
negative values push it towards "genuine".
"""

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fraud.config import PROJECT_ROOT, load_config  # noqa: E402
from fraud.models import to_lightgbm_frame  # noqa: E402
from fraud.split import load_splits  # noqa: E402

TOP_N_REASONS = 5
SERIES, INK, INK_MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"

FEATURE_LABELS = {
    "type": "Payment type",
    "log_amount": "Payment amount",
    "hour_of_day": "Hour of day",
    "empties_orig_account": "Empties sender's account",
    "orig_balance_zero": "Sender's balance was zero",
    "dest_balance_zero": "Receiving account's balance was zero",
    "old_balance_dest": "Receiving account's balance",
    "dest_txn_count_24h": "Payments into receiving account, last 24h",
    "dest_amount_sum_24h": "Money into receiving account, last 24h",
    "dest_prior_txn_count": "Earlier payments into receiving account",
    "hours_since_prev_dest_txn": "Hours since receiving account last got money",
}


def describe(feature: str, row: pd.Series) -> str:
    """One plain-English fact about a feature's value for one transaction."""
    v = row[feature]
    missing = bool(pd.isna(v))
    match feature:
        case "type":
            return "Payment is a transfer" if v == "TRANSFER" else "Payment is a cash-out"
        case "log_amount":
            return f"Amount is {row['amount']:,.0f}"
        case "hour_of_day":
            return f"Made at {int(v):02d}:00 on the simulation clock"
        case "empties_orig_account":
            return ("Payment empties the sender's account" if v
                    else "Sender keeps money after the payment")
        case "orig_balance_zero":
            return ("Sender's balance was zero before the payment" if v
                    else "Sender had money before the payment")
        case "dest_balance_zero":
            return ("Receiving account held no money before the payment" if v
                    else "Receiving account already held money")
        case "old_balance_dest":
            return f"Receiving account held {v:,.0f} before the payment"
        case "dest_txn_count_24h":
            return f"Receiving account got {int(v)} payment(s) in the previous 24 hours"
        case "dest_amount_sum_24h":
            return f"Receiving account got {v:,.0f} in the previous 24 hours"
        case "dest_prior_txn_count":
            return ("First payment ever seen into this receiving account" if v == 0
                    else f"Receiving account has {int(v)} earlier payment(s)")
        case "hours_since_prev_dest_txn":
            return ("No earlier payment into this receiving account" if missing
                    else f"{int(v)} hours since the receiving account last got money")
    return f"{feature} = {v}"


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older SHAP returns one array per class
        values = values[1]
    base = explainer.expected_value
    base = float(base[1] if np.ndim(base) else base)
    return np.asarray(values), base


def top_reasons(row: pd.Series, contributions: np.ndarray, columns: list[str],
                n: int = TOP_N_REASONS) -> list[dict]:
    """The n features that moved this transaction's score the most, either way."""
    order = np.argsort(-np.abs(contributions))[:n]
    return [{
        "feature": columns[i],
        "label": FEATURE_LABELS.get(columns[i], columns[i]),
        "fact": describe(columns[i], row),
        "shap": round(float(contributions[i]), 4),
        "direction": "raises risk" if contributions[i] > 0 else "lowers risk",
    } for i in order]


def _style(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)


def plot_global_importance(values, columns, path):
    importance = pd.Series(np.abs(values).mean(axis=0), index=columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [FEATURE_LABELS.get(c, c) for c in importance.index]
    ax.barh(labels, importance.values, color=SERIES, height=0.65)
    for y, v in enumerate(importance.values):
        ax.text(v, y, f" {v:.2f}", va="center", fontsize=8, color=INK_MUTED)
    _style(ax, "What drives the risk score overall (test period)",
           "Mean |SHAP value| (average impact on log-odds of fraud)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return importance.sort_values(ascending=False)


def plot_beeswarm(values, X, columns, path):
    display = X.copy()
    display["type"] = (display["type"] == "TRANSFER").astype(int)
    explanation = shap.Explanation(values=values, data=display.values,
                                   feature_names=[FEATURE_LABELS.get(c, c) for c in columns])
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=len(columns), show=False)
    plt.title("How each feature's value moves the score (test period)", loc="left", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_reasons(reasons: list[dict], title: str, path):
    """A simple waterfall-style bar chart of one transaction's top reasons."""
    r = list(reversed(reasons))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = ["#d03b3b" if x["shap"] > 0 else SERIES for x in r]
    ax.barh([x["fact"] for x in r], [x["shap"] for x in r], color=colors, height=0.6)
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    _style(ax, "", "SHAP value (red raises risk, blue lowers it)")
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    figures = cfg["paths"]["figures_dir"]
    decision = json.loads((PROJECT_ROOT / "models" / "decision.json").read_text())
    bundle = joblib.load(PROJECT_ROOT / "models" / f"{decision['model']}.joblib")
    model, columns = bundle["model"], bundle["columns"]

    test = load_splits(cfg["paths"]["features_parquet"], cfg["split"])["test"]
    X = to_lightgbm_frame(test, columns)
    test["risk_score"] = model.predict_proba(X)[:, 1]
    values, base = shap_values(model, X)

    # Check: SHAP values plus the base value must rebuild the model's raw score
    raw = model.predict(X, raw_score=True)
    assert np.allclose(values.sum(axis=1) + base, raw, atol=1e-4), "SHAP values do not add up"

    importance = plot_global_importance(values, columns, figures / "shap_global_importance.png")
    plot_beeswarm(values, X, columns, figures / "shap_beeswarm.png")
    importance.round(4).to_csv(PROJECT_ROOT / "reports" / "shap_global_importance.csv",
                               header=["mean_abs_shap"])

    flagged_mask = test["risk_score"].values >= decision["threshold"]
    flagged = test[flagged_mask].copy()
    flagged_values = values[flagged_mask]
    flagged["reasons"] = [
        json.dumps(top_reasons(row, contrib, columns))
        for (_, row), contrib in zip(flagged.iterrows(), flagged_values, strict=True)
    ]
    keep = ["transaction_id", "step", "day", "hour_of_day", "type", "amount", "risk_score",
            "is_fraud", "reasons"]
    out = PROJECT_ROOT / "data" / "processed" / "flagged_test.parquet"
    flagged = flagged.sort_values("risk_score", ascending=False)
    flagged[keep].to_parquet(out, index=False)

    # Example explanations: one caught fraud and one false alert, each the highest-scoring
    examples = {"fraud": flagged[flagged["is_fraud"] == 1].iloc[0],
                "false_alert": flagged[flagged["is_fraud"] == 0].iloc[0]}
    for name, row in examples.items():
        reasons = json.loads(row["reasons"])
        kind = "caught fraud" if name == "fraud" else "false alert"
        plot_reasons(reasons, f"Why transaction {row['transaction_id']} was flagged\n"
                              f"{kind}, risk score {row['risk_score']:.3f}",
                     figures / f"shap_example_{name}.png")

    print(f"Base rate (log-odds): {base:.3f}. SHAP values add up to the model output: OK")
    print("\nOverall importance (mean |SHAP|):")
    print(importance.round(3).rename(index=FEATURE_LABELS).to_string())
    print(f"\n{len(flagged):,} flagged test transactions with reasons saved to {out}")
    for name, row in examples.items():
        print(f"\nExample {name}: transaction {row['transaction_id']}, "
              f"score {row['risk_score']:.3f}")
        for r in json.loads(row["reasons"]):
            print(f"  {r['shap']:+.3f}  {r['fact']}")


if __name__ == "__main__":
    main()
