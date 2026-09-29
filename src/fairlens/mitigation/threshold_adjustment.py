"""Validation-only Fairlearn ThresholdOptimizer wrapper."""
from __future__ import annotations

from typing import Any
import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split

from fairlens.config import FairLensConfig


def fit_threshold_optimizer(X_train: Any, y_train: Any, sensitive_train: Any,
                            config: FairLensConfig, estimator: Any) -> ThresholdOptimizer:
    """Fit estimator and thresholds using only a split of training data.

    The original held-out test set is deliberately absent from this API. The estimator
    trains on a training sub-partition; thresholds are selected on validation data.
    """
    indices = np.arange(len(y_train))
    fit_indices, validation_indices = train_test_split(
        indices, test_size=config.validation_size, random_state=config.random_seed,
        stratify=np.asarray(y_train),
    )
    estimator.fit(X_train[fit_indices], np.asarray(y_train)[fit_indices])
    optimizer = ThresholdOptimizer(
        estimator=estimator, constraints=config.threshold_constraint,
        objective="accuracy_score", predict_method="predict_proba", prefit=True,
    )
    optimizer.fit(X_train[validation_indices], np.asarray(y_train)[validation_indices],
                  sensitive_features=np.asarray(sensitive_train)[validation_indices])
    return optimizer
