"""Allowlisted, extensible estimator registry for tabular supervised learning.

User uploads are data, never Python code or serialized estimators.
"""
from __future__ import annotations

MODEL_REGISTRY = {
    "logistic_regression": ("Logistic Regression", "classification", True),
    "decision_tree": ("Decision Tree Classifier", "classification", True),
    "random_forest": ("Random Forest Classifier", "classification", True),
    "extra_trees": ("Extra Trees Classifier", "classification", True),
    "gradient_boosting": ("Gradient Boosting Classifier", "classification", True),
    "knn_classifier": ("K-Nearest Neighbors Classifier", "classification", False),
    "gaussian_nb": ("Gaussian Naive Bayes", "classification", True),
    "linear_regression": ("Linear Regression", "regression", True),
    "ridge": ("Ridge Regression", "regression", True),
    "lasso": ("Lasso Regression", "regression", True),
    "decision_tree_regressor": ("Decision Tree Regressor", "regression", True),
    "random_forest_regressor": ("Random Forest Regressor", "regression", True),
    "gradient_boosting_regressor": ("Gradient Boosting Regressor", "regression", True),
    "knn_regressor": ("K-Nearest Neighbors Regressor", "regression", False),
}


def model_catalog() -> list[dict]:
    return [dict(id=key, name=value[0], task=value[1], sample_weight=value[2])
            for key, value in MODEL_REGISTRY.items()]


def build_estimator(name: str, seed: int = 42):
    """Construct a fresh model with fixed, documented, bounded defaults."""
    from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                                  GradientBoostingRegressor, RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    factories = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=seed),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=10, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=1),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=1),
        "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed),
        "knn_classifier": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        "gaussian_nb": GaussianNB,
        "linear_regression": LinearRegression,
        "ridge": lambda: Ridge(alpha=1.0),
        "lasso": lambda: Lasso(alpha=0.1, max_iter=2000, random_state=seed),
        "decision_tree_regressor": lambda: DecisionTreeRegressor(max_depth=10, random_state=seed),
        "random_forest_regressor": lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=1),
        "gradient_boosting_regressor": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed),
        "knn_regressor": lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=1),
    }
    if name not in MODEL_REGISTRY:
        raise ValueError("Choose a supported model from the registry.")
    return factories[name]()
