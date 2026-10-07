"""Phase 4: pick the alert threshold on validation, then score the test period once.

Everything that is decided (model, threshold, costs, capacity) is fixed using the validation
period. The test period is only scored at the end, with those decisions frozen.
"""

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import matplotlib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fraud.config import PROJECT_ROOT, load_config  # noqa: E402
from fraud.metrics import evaluate  # noqa: E402
from fraud.models import to_lightgbm_frame  # noqa: E402
from fraud.split import load_splits  # noqa: E402
from fraud.threshold import choose_threshold, cost_at_threshold, cost_curve  # noqa: E402
from fraud.train import setup_mlflow  # noqa: E402

SERIES = ["#2a78d6", "#eb6834"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def score(run_name: str, df: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(PROJECT_ROOT / "models" / f"{run_name}.joblib")
    model, columns = bundle["model"], bundle["columns"]
    X = to_lightgbm_frame(df, columns) if run_name.startswith("lightgbm") else df[columns]
    return model.predict_proba(X)[:, 1]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)


def plot_cost_curve(curve, chosen, capacity, path):
    c = curve[curve["alerts_per_day"] > 0].sort_values("alerts_per_day")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(c["alerts_per_day"], c["total_cost"] / 1e6, color=SERIES[0], linewidth=2)
    ax.axvline(capacity, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(capacity * 1.08, ax.get_ylim()[1] * 0.92, f"Team capacity\n{capacity} alerts/day",
            color=INK_MUTED, fontsize=9, va="top")
    ax.scatter([chosen["alerts_per_day"]], [chosen["total_cost"] / 1e6], s=64, color=INK, zorder=5)
    ax.annotate(f"Chosen threshold {chosen['threshold']:.3f}",
                (chosen["alerts_per_day"], chosen["total_cost"] / 1e6),
                xytext=(12, 12), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    _style(ax, "Total cost vs alerts raised per day (validation)", "Alerts per day (log scale)",
           "Total cost (millions)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_precision_recall_vs_threshold(curve, chosen, path):
    c = curve[curve["alerts"] > 0].sort_values("threshold")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(c["threshold"], c["recall"], color=SERIES[0], linewidth=2, label="Recall")
    ax.plot(c["threshold"], c["precision"], color=SERIES[1], linewidth=2, label="Precision")
    ax.axvline(chosen["threshold"], color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(chosen["threshold"] * 1.15, 0.05, "Chosen", color=INK_MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.05)
    ax.set_ylim(0, 1.02)
    _style(ax, "Precision and recall as the threshold rises (validation)",
           "Threshold on risk score (log scale)", "Share")
    ax.legend(frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sensitivity_table(y, s, amount, n_days, loss_rate) -> pd.DataFrame:
    """How the chosen threshold moves if the cost or capacity assumptions change."""
    rows = []
    for review_cost in (100, 1_000, 5_000, 20_000):
        curve = cost_curve(y, s, amount, review_cost, loss_rate, n_days)
        for capacity in (None, 250, 500, 1_000):
            best = choose_threshold(curve, capacity)
            rows.append({"review_cost": review_cost, "capacity_per_day": capacity or "none",
                         "threshold": best["threshold"],
                         "alerts_per_day": best["alerts_per_day"],
                         "precision": best["precision"], "recall": best["recall"],
                         "fraud_money_caught": best["fraud_money_caught"]})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    costs = cfg["costs"]
    figures = cfg["paths"]["figures_dir"]
    reports = PROJECT_ROOT / "reports"
    splits = load_splits(cfg["paths"]["features_parquet"], cfg["split"])
    valid, test = splits["valid"], splits["test"]
    days = {"valid": valid["day"].nunique(), "test": test["day"].nunique()}
    review, loss_rate = costs["review_cost_per_alert"], costs["missed_fraud_loss_rate"]
    capacity = costs["daily_alert_capacity"]

    # ---- 1. Decide everything on the validation period ----
    decisions = {}
    for role in ("main_model", "baseline_model"):
        run = costs[role]
        s = score(run, valid)
        curve = cost_curve(valid["is_fraud"], s, valid["amount"], review, loss_rate,
                           days["valid"])
        chosen = choose_threshold(curve, capacity)
        unconstrained = choose_threshold(curve, None)
        decisions[run] = float(chosen["threshold"])
        print(f"\n[validation] {run}")
        print(f"  cheapest with no capacity limit: threshold {unconstrained['threshold']:.4f}, "
              f"{unconstrained['alerts_per_day']:.0f} alerts/day")
        print(f"  chosen within {capacity}/day:     threshold {chosen['threshold']:.4f}, "
              f"{chosen['alerts_per_day']:.0f} alerts/day, precision {chosen['precision']:.3f}, "
              f"recall {chosen['recall']:.3f}")
        if role == "main_model":
            curve.round(6).to_csv(reports / "threshold_curve_validation.csv", index=False)
            plot_cost_curve(curve, chosen, capacity, figures / "cost_curve_validation.png")
            plot_precision_recall_vs_threshold(curve, chosen,
                                               figures / "precision_recall_vs_threshold.png")
            sens = sensitivity_table(valid["is_fraud"].values, s, valid["amount"].values,
                                     days["valid"], loss_rate)
            sens.round(4).to_csv(reports / "threshold_sensitivity_validation.csv", index=False)

    decision = {"model": costs["main_model"], "threshold": decisions[costs["main_model"]],
                "review_cost_per_alert": review, "daily_alert_capacity": capacity,
                "missed_fraud_loss_rate": loss_rate}
    (PROJECT_ROOT / "models" / "decision.json").write_text(json.dumps(decision, indent=2))

    # ---- 2. Score the test period once, with every decision frozen ----
    y, amount = test["is_fraud"].values, test["amount"].values
    results = {}
    for run, threshold in decisions.items():
        s = score(run, test)
        r = cost_at_threshold(y, s, amount, threshold, review, loss_rate)
        r.update({k: v for k, v in evaluate(y, s).items() if k in ("pr_auc", "roc_auc")})
        r["alerts_per_day"] = r["alerts"] / days["test"]
        results[run] = r
    rule = cost_at_threshold(y, test["is_flagged_fraud"].values, amount, 0.5, review, loss_rate)
    rule["alerts_per_day"] = rule["alerts"] / days["test"]
    results["rule_isFlaggedFraud"] = rule
    no_model = cost_at_threshold(y, np.zeros(len(y)), amount, 1.0, review, loss_rate)
    no_model["alerts_per_day"] = 0.0
    results["no_alerts"] = no_model

    table = pd.DataFrame(results).T
    cols = ["threshold", "alerts_per_day", "precision", "recall", "fraud_money_caught",
            "pr_auc", "roc_auc", "true_alerts", "false_alerts", "missed_frauds", "total_cost"]
    table = table[[c for c in cols if c in table]]
    table.to_csv(reports / "test_results.csv")
    print(f"\n[test, days {cfg['split']['test_days']}, scored once]")
    print(table.to_string(float_format=lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.0f}"))

    main = results[costs["main_model"]]
    saving = 1 - main["total_cost"] / no_model["total_cost"]
    print(f"\nMain model cuts total cost by {saving:.1%} vs raising no alerts")

    setup_mlflow(cfg)
    with mlflow.start_run(run_name="phase4_threshold_and_test"):
        mlflow.log_params(decision)
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in main.items()
                            if isinstance(v, int | float)})
        mlflow.log_metric("test_cost_saving_vs_no_alerts", saving)
        for f in ("cost_curve_validation.png", "precision_recall_vs_threshold.png"):
            mlflow.log_artifact(str(figures / f))
        mlflow.log_artifact(str(reports / "test_results.csv"))


if __name__ == "__main__":
    main()
