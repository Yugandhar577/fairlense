"""Consistent model construction for the comparative benchmark."""
from __future__ import annotations

from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from fairlens.config import FairLensConfig


def build_model_instances(config: FairLensConfig) -> dict[str, Any]:
    """Create new, intentionally lightly tuned estimators for every experiment arm."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.random_seed),
        "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=config.random_seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=config.random_seed, n_jobs=-1,
        ),
    }


def train_models(X_train: Any, y_train: Any, config: FairLensConfig,
                 sample_weight: Any | None = None) -> dict[str, Any]:
    """Fit fresh model instances using optional training-only sample weights."""
    trained = build_model_instances(config)
    for model in trained.values():
        if sample_weight is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)
    return trained
