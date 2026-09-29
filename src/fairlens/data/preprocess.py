"""Leakage-safe preprocessing and split construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairlens.config import FairLensConfig


@dataclass
class PreprocessedData:
    X_train: Any
    X_test: Any
    y_train: pd.Series
    y_test: pd.Series
    sensitive_train: pd.DataFrame
    sensitive_test: pd.DataFrame
    transformer: ColumnTransformer
    report: dict[str, Any]


def add_age_group(frame: pd.DataFrame, config: FairLensConfig) -> pd.DataFrame:
    """Create documented, left-inclusive age bands (e.g. age 25 is 25-44)."""
    result = frame.copy()
    result["age_group"] = pd.cut(
        result["age"], bins=list(config.age_bins), labels=list(config.age_labels),
        right=False, include_lowest=True,
    )
    return result


def add_intersectional_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Add an evaluation-only gender × age-group sensitive attribute.

    This deliberately happens before the train/test split, but it is a
    deterministic row-wise label construction: it learns no dataset parameter
    and therefore cannot leak information from the test partition.
    """
    result = frame.copy()
    result["gender_age_group"] = (
        result["gender"].astype("string") + " | " + result["age_group"].astype("string")
    )
    return result


def _build_transformer(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric = feature_frame.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in feature_frame.columns if column not in numeric]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop")


def preprocess_data(frame: pd.DataFrame, config: FairLensConfig) -> PreprocessedData:
    """Clean, split, and encode data without fitting test-derived preprocessing state."""
    input_rows = len(frame)
    data = add_intersectional_group(add_age_group(frame, config))
    required = [config.target_column, "age", "gender", *config.sensitive_attributes]
    data = data.dropna(subset=required).copy()
    data[config.target_column] = (data[config.target_column] == ">50K").astype(int)
    sensitive = data.loc[:, list(config.sensitive_attributes)].copy()
    for attribute in config.sensitive_attributes:
        sensitive[attribute] = sensitive[attribute].astype(str)
    y = data[config.target_column].copy()
    feature_columns = [column for column in data.columns if column != config.target_column]
    if not config.include_sensitive_as_feature:
        feature_columns = [column for column in feature_columns if column not in config.sensitive_attributes]
    X = data[feature_columns].copy()
    X_train_raw, X_test_raw, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive, test_size=config.test_size, random_state=config.random_seed, stratify=y,
    )
    transformer = _build_transformer(X_train_raw)
    X_train = transformer.fit_transform(X_train_raw)
    X_test = transformer.transform(X_test_raw)
    report = {
        "input_rows": input_rows,
        "rows_removed": input_rows - len(data),
        "rows_after_cleaning": len(data),
        "training_samples": len(y_train),
        "test_samples": len(y_test),
        "feature_count": int(X_train.shape[1]),
        "sensitive_group_counts": {
            "train": {attribute: sensitive_train[attribute].value_counts().to_dict() for attribute in config.sensitive_attributes},
            "test": {attribute: sensitive_test[attribute].value_counts().to_dict() for attribute in config.sensitive_attributes},
        },
        "target_distribution": {str(key): float(value) for key, value in y.value_counts(normalize=True).items()},
    }
    return PreprocessedData(X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True),
                            sensitive_train.reset_index(drop=True), sensitive_test.reset_index(drop=True),
                            transformer, report)
