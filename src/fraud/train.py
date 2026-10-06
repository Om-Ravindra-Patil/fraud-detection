"""Train and compare models on the time-based split, logging every run to MLflow.

The test set is NOT used here. Models are compared on the validation period only;
the test period is scored once, in Phase 4, after the threshold has been chosen.
"""

import os
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import matplotlib  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from fraud.config import PROJECT_ROOT, load_config  # noqa: E402
from fraud.metrics import evaluate, precision_recall_at  # noqa: E402
from fraud.models import build_lightgbm, build_logistic_regression, to_lightgbm_frame  # noqa: E402
from fraud.split import feature_columns, load_splits  # noqa: E402

# (model, feature set, class weighting). Weighting for LightGBM: none, sqrt of the
# genuine-to-fraud ratio (about 22), or the full ratio (about 493).
EXPERIMENTS = [
    ("logreg", "realistic", "balanced"),
    ("lightgbm", "realistic", "none"),
    ("lightgbm", "realistic", "sqrt"),
    ("lightgbm", "realistic", "balanced"),
    ("logreg", "full", "balanced"),
    ("lightgbm", "full", "none"),
]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup_mlflow(cfg: dict) -> None:
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{PROJECT_ROOT / uri.removeprefix('sqlite:///')}"
    mlflow.set_tracking_uri(uri)
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(name)


def fit_logreg(train, valid, columns, cfg):
    model = build_logistic_regression(columns, cfg["model"]["logistic_regression"],
                                      cfg["model"]["seed"])
    model.fit(train[columns], train["is_fraud"])
    return model, model.predict_proba(valid[columns])[:, 1], {}


def fit_lightgbm(train, valid, columns, cfg, weighting):
    params = cfg["model"]["lightgbm"]
    y = train["is_fraud"]
    ratio = float((y == 0).sum() / (y == 1).sum())
    spw = {"none": 1.0, "sqrt": ratio**0.5, "balanced": ratio}[weighting]
    model = build_lightgbm(params, cfg["model"]["seed"], scale_pos_weight=spw)
    X_train, X_valid = to_lightgbm_frame(train, columns), to_lightgbm_frame(valid, columns)
    model.fit(
        X_train, y,
        eval_X=(X_valid,),
        eval_y=(valid["is_fraud"],),
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False)],
    )
    extra = {"best_iteration": model.best_iteration_, "scale_pos_weight": spw}
    return model, model.predict_proba(X_valid)[:, 1], extra


def plot_pr_curves(curves: dict, rule_point: tuple[float, float], fraud_rate: float, path):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for (label, (y, score)), color in zip(curves.items(), SERIES, strict=False):
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, color=color, linewidth=2, label=label)
    ax.scatter([rule_point[1]], [rule_point[0]], s=64, color=INK, zorder=5,
               label="Existing rule (isFlaggedFraud)")
    ax.axhline(fraud_rate, color=INK_MUTED, linewidth=1, linestyle="--",
               label=f"Random guessing ({fraud_rate:.1%})")
    ax.set_title("Precision vs recall on the validation period (days 20 to 24)",
                 loc="left", fontsize=12, color=INK)
    ax.set_xlabel("Recall (share of fraud caught)", color=INK_MUTED)
    ax.set_ylabel("Precision (share of alerts that are fraud)", color=INK_MUTED)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    splits = load_splits(paths["features_parquet"], cfg["split"])
    train, valid = splits["train"], splits["valid"]
    for name, df in splits.items():
        print(f"{name:<6} days {df['day'].min()}-{df['day'].max()}  "
              f"{len(df):>9,} rows  {int(df['is_fraud'].sum()):>5,} frauds")

    setup_mlflow(cfg)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    results, curves = [], {}

    # The existing rule is the bar to beat
    rule_p, rule_r = precision_recall_at(valid["is_fraud"].values,
                                         valid["is_flagged_fraud"].values, 0.5)
    with mlflow.start_run(run_name="rule_isFlaggedFraud"):
        mlflow.log_params({"model": "rule", "feature_set": "n/a"})
        mlflow.log_metrics({"precision": rule_p, "recall": rule_r})
    results.append({"run": "rule_isFlaggedFraud", "precision_at_0.5": rule_p,
                    "recall_at_0.5": rule_r})

    for model_name, feature_set, weighting in EXPERIMENTS:
        run_name = f"{model_name}_{feature_set}_{weighting}"
        columns = feature_columns(cfg["features"], feature_set)
        print(f"\nTraining {run_name} on {len(columns)} features...")
        with mlflow.start_run(run_name=run_name):
            start = time.time()
            if model_name == "logreg":
                model, scores, extra = fit_logreg(train, valid, columns, cfg)
            else:
                model, scores, extra = fit_lightgbm(train, valid, columns, cfg, weighting)
            metrics = evaluate(valid["is_fraud"].values, scores)
            metrics["train_seconds"] = time.time() - start

            mlflow.log_params({"model": model_name, "feature_set": feature_set,
                               "class_weighting": weighting, "features": ",".join(columns),
                               "train_days": str(cfg["split"]["train_days"]),
                               "valid_days": str(cfg["split"]["valid_days"]), **extra})
            if model_name == "lightgbm":
                mlflow.log_params(cfg["model"]["lightgbm"])
            else:
                mlflow.log_params(cfg["model"]["logistic_regression"])
            mlflow.log_metrics(metrics)
            model_path = models_dir / f"{run_name}.joblib"
            joblib.dump({"model": model, "columns": columns}, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        results.append({"run": run_name, **metrics})
        print({k: round(v, 4) for k, v in metrics.items()})
        if feature_set == "realistic" or model_name == "lightgbm":
            label = {"logreg_realistic_balanced": "Logistic regression (realistic)",
                     "lightgbm_realistic_none": "LightGBM (realistic, unweighted)",
                     "lightgbm_realistic_balanced": "LightGBM (realistic, weighted)",
                     "lightgbm_full_none": "LightGBM (full, includes artifact)"}.get(run_name)
            if label:
                curves[label] = (valid["is_fraud"].values, scores)

    table = pd.DataFrame(results)
    out = PROJECT_ROOT / "reports" / "model_comparison_validation.csv"
    table.round(4).to_csv(out, index=False)
    plot_pr_curves(curves, (rule_p, rule_r), float(valid["is_fraud"].mean()),
                   paths["figures_dir"] / "pr_curves_validation.png")
    cols = ["run", "pr_auc", "roc_auc", "recall_at_precision_0.5", "recall_at_precision_0.8",
            "precision_at_0.5", "recall_at_0.5"]
    print("\n" + table[cols].round(4).to_string(index=False))
    print(f"\nSaved {out} and pr_curves_validation.png. View runs with: make mlflow")


if __name__ == "__main__":
    main()
