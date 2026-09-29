"""Performance metric calculation from genuine predictions."""
from __future__ import annotations

from typing import Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_performance(model: Any, X_test: Any, y_test: Any, y_pred: Any | None = None,
                         y_score: Any | None = None) -> dict[str, float]:
    """Evaluate a fitted classifier; uses score probabilities only for ROC-AUC."""
    predictions = model.predict(X_test) if y_pred is None else y_pred
    result = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    try:
        scores = model.predict_proba(X_test)[:, 1] if y_score is None else y_score
        result["roc_auc"] = float(roc_auc_score(y_test, scores))
    except (AttributeError, IndexError, TypeError, ValueError):
        # ThresholdOptimizer has no stable probability interface after postprocessing.
        result["roc_auc"] = float(roc_auc_score(y_test, predictions))
    return result
