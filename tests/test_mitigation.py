import inspect
import numpy as np
from sklearn.linear_model import LogisticRegression

from fairlens.config import FairLensConfig
from fairlens.mitigation.reweighing import compute_reweighing_weights
from fairlens.mitigation.threshold_adjustment import fit_threshold_optimizer


def test_reweighing_returns_valid_weights_and_reduces_joint_dependence():
    labels = np.array([1, 1, 1, 0, 0, 0])
    groups = np.array(["A", "A", "B", "A", "B", "B"])
    weights = compute_reweighing_weights(labels, groups)
    assert len(weights) == len(labels)
    assert np.all(np.isfinite(weights)) and np.all(weights > 0)
    weighted_a_positive = weights[(groups == "A") & (labels == 1)].sum()
    weighted_b_positive = weights[(groups == "B") & (labels == 1)].sum()
    assert np.isclose(weighted_a_positive, weighted_b_positive)


def test_threshold_optimizer_uses_train_validation_api_only():
    signature = inspect.signature(fit_threshold_optimizer)
    assert "X_test" not in signature.parameters
    assert "y_test" not in signature.parameters


def test_threshold_optimizer_produces_binary_predictions():
    # Repeated group/label combinations ensure both groups have both classes in
    # the validation fold, which ThresholdOptimizer requires to draw trade-off curves.
    X = np.arange(40).reshape(-1, 1)
    groups = np.array(["A", "A", "B", "B"] * 10)
    y = np.array([0, 1, 0, 1] * 10)
    optimizer = fit_threshold_optimizer(X, y, groups, FairLensConfig(validation_size=0.5), LogisticRegression())
    prediction = optimizer.predict(X, sensitive_features=groups)
    assert set(np.unique(prediction)).issubset({0, 1})
