"""Build the small, self-contained bundle the dashboard and its Docker image need.

Outputs (all committed to git, so the app runs without Kaggle access or the raw data):
- models/fraud_model.txt   LightGBM model in its portable text format
- app/data/alerts.parquet  flagged test payments with score, SHAP reasons and analyst note
- app/data/summary.json    headline results, feature importance and fairness table

PaySim is synthetic and licensed CC BY-SA 4.0; no real customer data is included.
"""

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraud.config import PROJECT_ROOT, load_config  # noqa: E402
from fraud.models import to_lightgbm_frame  # noqa: E402
from fraud.notes import risk_band, template_note  # noqa: E402
from fraud.serving import MODEL_FILE, Scorer  # noqa: E402
from fraud.split import load_splits  # noqa: E402

APP_DATA = PROJECT_ROOT / "app" / "data"


def export_model(models_dir) -> None:
    decision = json.loads((models_dir / "decision.json").read_text())
    bundle = joblib.load(models_dir / f"{decision['model']}.joblib")
    bundle["model"].booster_.save_model(str(models_dir / MODEL_FILE),
                                        num_iteration=bundle["model"].best_iteration_)


def check_export(models_dir, test: pd.DataFrame) -> None:
    """The exported text model must score exactly like the trained model."""
    decision = json.loads((models_dir / "decision.json").read_text())
    bundle = joblib.load(models_dir / f"{decision['model']}.joblib")
    expected = bundle["model"].predict_proba(to_lightgbm_frame(test, bundle["columns"]))[:, 1]
    actual, _ = Scorer(models_dir).score(test)
    assert np.allclose(expected, actual, atol=1e-9), "Exported model scores differ"


def build_alerts(processed, threshold: float) -> pd.DataFrame:
    flagged = pd.read_parquet(processed / "flagged_test.parquet")
    notes = pd.read_parquet(processed / "analyst_notes.parquet")
    alerts = flagged.merge(notes[["transaction_id", "note", "source", "model"]],
                           on="transaction_id", how="left")
    missing = alerts["note"].isna()
    alerts.loc[missing, "note"] = [template_note(r, threshold)
                                   for r in alerts[missing].to_dict("records")]
    alerts.loc[missing, "source"] = "template"
    alerts.loc[missing, "model"] = "template"
    alerts["risk_band"] = [risk_band(s, threshold) for s in alerts["risk_score"]]
    alerts = alerts.rename(columns={"source": "note_source", "model": "note_model",
                                    "is_fraud": "actual_fraud"})
    cols = ["transaction_id", "day", "hour_of_day", "type", "amount", "risk_score",
            "risk_band", "actual_fraud", "reasons", "note", "note_source", "note_model"]
    return alerts[cols].sort_values("risk_score", ascending=False).reset_index(drop=True)


def build_summary(cfg, decision, n_test_days: int) -> dict:
    reports = PROJECT_ROOT / "reports"
    results = pd.read_csv(reports / "test_results.csv", index_col=0)
    importance = pd.read_csv(reports / "shap_global_importance.csv", index_col=0)
    fairness = pd.read_csv(reports / "fairness_segments_test.csv")
    keep = ["alerts_per_day", "precision", "recall", "fraud_money_caught", "pr_auc",
            "total_cost"]
    return {
        "threshold": decision["threshold"],
        "review_cost_per_alert": decision["review_cost_per_alert"],
        "daily_alert_capacity": decision["daily_alert_capacity"],
        "test_days": cfg["split"]["test_days"],
        "n_test_days": n_test_days,
        "results": {name: {k: (None if pd.isna(row[k]) else float(row[k])) for k in keep}
                    for name, row in results.iterrows()},
        "global_importance": importance["mean_abs_shap"].round(4).to_dict(),
        "fairness": fairness.to_dict("records"),
    }


def main() -> None:
    cfg = load_config()
    models_dir = PROJECT_ROOT / "models"
    processed = PROJECT_ROOT / "data" / "processed"
    decision = json.loads((models_dir / "decision.json").read_text())
    test = load_splits(cfg["paths"]["features_parquet"], cfg["split"])["test"]

    export_model(models_dir)
    check_export(models_dir, test)
    APP_DATA.mkdir(parents=True, exist_ok=True)
    alerts = build_alerts(processed, decision["threshold"])
    alerts.to_parquet(APP_DATA / "alerts.parquet", index=False)
    summary = build_summary(cfg, decision, test["day"].nunique())
    (APP_DATA / "summary.json").write_text(json.dumps(summary, indent=2))

    size_kb = (APP_DATA / "alerts.parquet").stat().st_size / 1024
    print(f"Exported model to models/{MODEL_FILE} (scores match the trained model)")
    print(f"{len(alerts):,} alerts ({size_kb:.0f} KB), notes: "
          f"{alerts['note_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
