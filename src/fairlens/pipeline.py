"""Offline FairLens experimental pipeline; dashboard code is deliberately absent."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any
import numpy as np
import pandas as pd

from fairlens.config import DEFAULT_CONFIG, FairLensConfig
from fairlens.data.loader import load_raw_data
from fairlens.data.preprocess import PreprocessedData, preprocess_data
from fairlens.evaluation.fairness import evaluate_fairness
from fairlens.evaluation.comparison import calculate_fairness_changes
from fairlens.evaluation.performance import evaluate_performance
from fairlens.mitigation.reweighing import compute_reweighing_weights
from fairlens.mitigation.threshold_adjustment import fit_threshold_optimizer
from fairlens.models.train import build_model_instances, train_models
from fairlens.results_store import save_model, save_results

LOGGER = logging.getLogger(__name__)


def _evaluate_arm(model_name: str, mitigation_state: str, model: Any, data: PreprocessedData,
                  config: FairLensConfig, requires_sensitive_prediction: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build all genuine result rows for a fitted model arm using only held-out test data."""
    score_values = None
    if requires_sensitive_prediction:
        # Fairlearn's postprocessor can randomise between threshold operations.
        # Pinning its prediction seed makes the final reported test evaluation
        # reproducible without involving any test value in threshold selection.
        predictions = model.predict(
            data.X_test, sensitive_features=data.sensitive_test["gender"], random_state=config.random_seed,
        )
        # ThresholdOptimizer is a postprocessor. ROC-AUC remains a ranking measure
        # of its fitted, training-derived base estimator rather than of randomized
        # binary threshold decisions.
        score_values = model.estimator.predict_proba(data.X_test)[:, 1]
    else:
        predictions = model.predict(data.X_test)
    performance = {"model": model_name, "mitigation_state": mitigation_state,
                   **evaluate_performance(model, data.X_test, data.y_test, predictions, score_values)}
    fairness_rows: list[dict[str, Any]] = []
    for attribute in config.sensitive_attributes:
        summary, groups = evaluate_fairness(data.y_test, predictions, data.sensitive_test[attribute],
                                            config.low_sample_threshold)
        for group in groups:
            fairness_rows.append({
                "model": model_name, "mitigation_state": mitigation_state,
                "sensitive_attribute": attribute, **summary, **group,
            })
    return performance, fairness_rows


def run_full_pipeline(config: FairLensConfig = DEFAULT_CONFIG) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute one reproducible Adult-data benchmark and populate the Results Store.

    Mitigation choices and thresholds are learned only from training-derived partitions.
    The held-out test split is evaluated once per arm and is never supplied to a fit call.
    """
    raw = load_raw_data(config.data_path)
    data = preprocess_data(raw, config)
    performance_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    LOGGER.info("Training baseline models")
    baseline_models = train_models(data.X_train, data.y_train, config)
    for name, model in baseline_models.items():
        performance, fairness = _evaluate_arm(name, "baseline", model, data, config)
        performance_rows.append(performance)
        fairness_rows.extend(fairness)
        save_model(config.results_dir, model, name, "baseline")

    # Reweighing uses one documented sensitive attribute. Gender is selected as the
    # primary mitigation attribute, while both gender and age groups remain evaluated.
    LOGGER.info("Training reweighed models")
    weights = compute_reweighing_weights(data.y_train, data.sensitive_train["gender"])
    reweighed_models = train_models(data.X_train, data.y_train, config, sample_weight=weights)
    for name, model in reweighed_models.items():
        performance, fairness = _evaluate_arm(name, "reweighing", model, data, config)
        performance_rows.append(performance)
        fairness_rows.extend(fairness)
        save_model(config.results_dir, model, name, "reweighing")

    LOGGER.info("Fitting validation-only threshold optimizers")
    for name in config.models:
        optimizer = fit_threshold_optimizer(data.X_train, data.y_train, data.sensitive_train["gender"], config,
                                            build_model_instances(config)[name])
        performance, fairness = _evaluate_arm(name, "threshold_adjustment", optimizer, data, config, True)
        performance_rows.append(performance)
        fairness_rows.extend(fairness)
        save_model(config.results_dir, optimizer, name, "threshold_adjustment")

    fairness_changes = calculate_fairness_changes(pd.DataFrame(fairness_rows))
    run_config = {
        **config.serializable(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_report": data.report,
        "mitigation_notes": {
            "reweighing_sensitive_attribute": "gender",
            "threshold_adjustment_sensitive_attribute": "gender",
            "threshold_adjustment_fit_data": "training split further separated into train and validation; test untouched",
            "cross_group_effect_analysis": (
                "Mitigation is targeted at gender; fairness is independently measured for gender, "
                "age_group, and the evaluation-only gender_age_group intersection."
            ),
        },
    }
    save_results(config.results_dir, performance_rows, fairness_rows, fairness_changes.to_dict("records"), run_config)
    return performance_rows, fairness_rows
