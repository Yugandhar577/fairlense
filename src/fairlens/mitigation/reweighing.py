"""Training-only Kamiran and Calders style reweighing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reweighing_weights(y_train: pd.Series | np.ndarray,
                               sensitive_train: pd.Series | np.ndarray) -> np.ndarray:
    """Return weights that make sensitive group and label independent in expectation.

    No test values are accepted by this function; callers must pass a training split.
    Empty group-label cells retain weight one because there are no instances to weight.
    """
    labels = pd.Series(y_train).reset_index(drop=True)
    groups = pd.Series(sensitive_train).astype(str).reset_index(drop=True)
    if len(labels) != len(groups):
        raise ValueError("y_train and sensitive_train must have equal length")
    weights = np.ones(len(labels), dtype=float)
    for group in groups.unique():
        for label in labels.unique():
            mask = (groups == group) & (labels == label)
            observed = float(mask.mean())
            expected = float((groups == group).mean() * (labels == label).mean())
            if observed > 0:
                weights[mask.to_numpy()] = expected / observed
    return weights
