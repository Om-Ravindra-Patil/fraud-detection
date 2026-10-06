"""Evaluation metrics suited to rare-event problems. Accuracy is deliberately not used."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def precision_recall_at(y_true, y_score, threshold: float) -> tuple[float, float]:
    flagged = y_score >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    precision = tp / flagged.sum() if flagged.sum() else 0.0
    recall = tp / max(int(np.sum(y_true == 1)), 1)
    return float(precision), float(recall)


def recall_at_precision(y_true, y_score, min_precision: float) -> float:
    """Highest recall reachable while keeping precision at or above min_precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(y_true, y_score) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    precision_05, recall_05 = precision_recall_at(y_true, y_score, 0.5)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision_at_0.5": precision_05,
        "recall_at_0.5": recall_05,
        "recall_at_precision_0.5": recall_at_precision(y_true, y_score, 0.5),
        "recall_at_precision_0.8": recall_at_precision(y_true, y_score, 0.8),
        "fraud_rate": float(y_true.mean()),
    }
