"""Fairlearn-based group metrics with explicitly safe undefined-value handling."""
from __future__ import annotations

from typing import Any
import math
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate


def evaluate_fairness(y_true: Any, y_pred: Any, sensitive_feature: Any,
                      low_sample_threshold: int = 30) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Measure group disparities for one sensitive attribute using Fairlearn MetricFrame.

    Results are descriptive statistical indicators. They are not an overall legal or
    moral judgement about a model, organisation, or population.
    """
    groups = pd.Series(sensitive_feature, name="group").astype(str).reset_index(drop=True)
    truth = pd.Series(y_true).reset_index(drop=True)
    predictions = pd.Series(y_pred).reset_index(drop=True)
    metric_frame = MetricFrame(
        metrics={"selection_rate": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
        y_true=truth, y_pred=predictions, sensitive_features=groups,
    )
    by_group = metric_frame.by_group
    selection = by_group["selection_rate"].astype(float)
    tpr = by_group["tpr"].astype(float)
    fpr = by_group["fpr"].astype(float)
    selection_max = selection.max()
    ratio = float("nan") if selection_max == 0 or pd.isna(selection_max) else float(selection.min() / selection_max)
    summary = {
        "demographic_parity_diff": float(selection.max() - selection.min()),
        "equal_opportunity_diff": float(tpr.max() - tpr.min()),
        "equalized_odds_diff": float(max(tpr.max() - tpr.min(), fpr.max() - fpr.min())),
        "disparate_impact_ratio": ratio,
        "min_group_size": int(groups.value_counts().min()),
        "low_confidence_flag": bool(groups.value_counts().min() < low_sample_threshold),
    }
    detail: list[dict[str, Any]] = []
    counts = groups.value_counts()
    for group in by_group.index:
        count = int(counts[group])
        detail.append({
            "group": str(group), "group_count": count,
            "selection_rate": float(selection.loc[group]), "tpr": float(tpr.loc[group]),
            "fpr": float(fpr.loc[group]), "low_sample_warning": count < low_sample_threshold,
        })
    return summary, detail


def is_defined(value: float) -> bool:
    """Whether a stored fairness statistic can safely be displayed as a number."""
    return value is not None and not (isinstance(value, float) and math.isnan(value))
