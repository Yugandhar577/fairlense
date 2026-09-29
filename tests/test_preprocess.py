import numpy as np
import pandas as pd

from fairlens.config import FairLensConfig
from fairlens.data.preprocess import add_age_group, add_intersectional_group, preprocess_data


def _adult_like(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        "age": [20, 25, 44, 45, 64, 65, 30, 50] * (rows // 8),
        "workclass": ["Private", "?"] * (rows // 2),
        "fnlwgt": list(range(100, 100 + rows)), "education": ["HS-grad"] * rows,
        "education-num": [9] * rows, "marital-status": ["Never-married"] * rows,
        "occupation": ["Sales"] * rows, "relationship": ["Not-in-family"] * rows,
        "race": ["White"] * rows, "gender": ["Male", "Female"] * (rows // 2),
        "capital-gain": [0] * rows, "capital-loss": [0] * rows,
        "hours-per-week": [40] * rows, "native-country": ["United-States"] * rows,
        "income": [">50K", "<=50K"] * (rows // 2),
    })


def test_age_group_boundaries():
    data = add_age_group(_adult_like(), FairLensConfig())
    assert data.loc[:5, "age_group"].astype(str).tolist() == ["<25", "25-44", "25-44", "45-64", "45-64", "65+"]


def test_intersectional_groups_are_created_and_excluded_from_features_by_default():
    data = add_intersectional_group(add_age_group(_adult_like(), FairLensConfig()))
    assert data.loc[0, "gender_age_group"] == "Male | <25"
    assert data.loc[1, "gender_age_group"] == "Female | 25-44"
    prepared = preprocess_data(_adult_like(), FairLensConfig())
    assert "gender_age_group" in prepared.sensitive_train.columns
    assert "gender_age_group" not in prepared.transformer.feature_names_in_


def test_preprocessing_handles_missing_encodes_and_splits_reproducibly():
    config = FairLensConfig(test_size=0.25)
    first = preprocess_data(_adult_like(), config)
    second = preprocess_data(_adult_like(), config)
    assert first.report["training_samples"] + first.report["test_samples"] == 40
    assert first.X_train.shape[1] > 0
    assert np.array_equal(first.X_test, second.X_test)
    assert "gender" not in first.transformer.feature_names_in_  # default excludes sensitive attributes


def test_scaler_is_fit_on_train_only():
    data = _adult_like()
    prepared = preprocess_data(data, FairLensConfig())
    scaler = prepared.transformer.named_transformers_["numeric"].named_steps["scaler"]
    # Full-data age mean differs from the training-only scaler mean under a fixed split.
    assert not np.isclose(scaler.mean_[0], data["age"].mean())
