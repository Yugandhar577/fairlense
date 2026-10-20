"""Dataset-independent train/validation/test experiments; no UI or HTTP code."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairlens.evaluation.comparison import calculate_fairness_changes
from fairlens.evaluation.fairness import evaluate_fairness
from fairlens.mitigation.reweighing import compute_reweighing_weights
from fairlens.models.registry import MODEL_REGISTRY, build_estimator

MAX_ROWS = 50_000
MAX_COLUMNS = 100
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_MODELS = 7
MAX_ENCODED_CELLS = 5_000_000


def parse_csv(csv_text: str) -> pd.DataFrame:
    if not isinstance(csv_text, str) or len(csv_text.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload a UTF-8 CSV no larger than 20 MB.")
    if "\x00" in csv_text:
        raise ValueError("CSV contains binary/null bytes. Upload a plain UTF-8 text CSV.")
    import csv
    try:
        reader = csv.reader(io.StringIO(csv_text), strict=True)
        header = next(reader)
        header = [name.lstrip("\ufeff").strip() for name in header]
        if len(header) < 2 or len(header) > MAX_COLUMNS or any(not name for name in header):
            raise ValueError("CSV needs 2–100 named columns, including a target and a feature.")
        if len(set(header)) != len(header):
            raise ValueError("CSV column names must be unique after trimming spaces.")
        if any(len(name) > 128 for name in header):
            raise ValueError("Column names must be at most 128 characters.")
        for index, row in enumerate(reader):
            if row and len(row) != len(header):
                raise ValueError(f"CSV record {index + 2} has a different number of fields than its header.")
            if index > MAX_ROWS:
                raise ValueError("Datasets support at most 50,000 rows.")
        frame = pd.read_csv(io.StringIO(csv_text), skipinitialspace=True,
                            na_values=["?"], nrows=MAX_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration, csv.Error) as error:
        raise ValueError("Could not read the CSV. Use comma-separated data with a header row.") from error
    if not 20 <= len(frame) <= MAX_ROWS:
        raise ValueError("Datasets must contain between 20 and 50,000 rows.")
    frame.columns = header
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].astype("string").str.strip().replace("", pd.NA)
    numeric = frame.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Replace infinite numeric values with finite values or blanks.")
    if frame.shape[1] > MAX_COLUMNS:
        raise ValueError("Datasets support at most 100 columns.")
    return frame


def dataset_profile(frame: pd.DataFrame, dataset_id: str, name: str) -> dict:
    return {
        "id": dataset_id, "name": name, "rows": len(frame),
        "columns": [dict(name=column,
                         kind="numeric" if pd.api.types.is_numeric_dtype(frame[column]) else "categorical",
                         missing=int(frame[column].isna().sum()),
                         unique=int(frame[column].nunique()),
                         # Only publish complete candidate-label lists for small-cardinality columns.
                         values=sorted(frame[column].dropna().astype(str).unique().tolist())
                         if frame[column].nunique() <= 30 else [])
                    for column in frame.columns],
    }


def validate_spec(spec: dict, columns: list[str]) -> dict:
    """Reject incompatible task/model choices before allocating a worker."""
    if not isinstance(spec, dict):
        raise ValueError("Experiment settings must be an object.")
    task = spec.get("task")
    if task not in {"classification", "regression"}:
        raise ValueError("Choose classification or regression.")
    models = spec.get("models")
    if (not isinstance(models, list) or not 1 <= len(models) <= MAX_MODELS
            or any(not isinstance(m, str) or m not in MODEL_REGISTRY for m in models)
            or len(set(models)) != len(models)):
        raise ValueError("Choose 1–7 distinct supported models.")
    if any(MODEL_REGISTRY[m][1] != task for m in models):
        raise ValueError("All selected models must support the chosen task. Linear regression and logistic regression cannot share a benchmark task.")
    target = spec.get("target")
    features = spec.get("features")
    sensitive = spec.get("sensitive", [])
    if not isinstance(target, str) or target not in columns:
        raise ValueError("Select an existing target column.")
    for names, label in [(features, "feature"), (sensitive, "sensitive")]:
        if (not isinstance(names, list) or any(not isinstance(n, str) or n not in columns for n in names)
                or len(set(names)) != len(names) or target in names):
            raise ValueError(f"Choose distinct existing {label} columns, excluding the target.")
    if not features or set(features) & set(sensitive):
        raise ValueError("Choose at least one feature; sensitive columns are excluded from predictive features.")
    if len(sensitive) > 3:
        raise ValueError("Choose at most three sensitive columns.")
    if any(spec.get(key) is not None and not isinstance(spec[key], bool) for key in ["intersection"]):
        raise ValueError("Intersection must be a boolean.")
    mitigations = spec.get("mitigations", [])
    if (not isinstance(mitigations, list)
            or any(m not in ["reweighing", "threshold_adjustment"] for m in mitigations)
            or len(set(mitigations)) != len(mitigations)):
        raise ValueError("Choose supported mitigation techniques.")
    if mitigations:
        if task != "classification" or not sensitive or spec.get("mitigation_attribute") not in sensitive:
            raise ValueError("Mitigation needs a binary classification task and a selected sensitive attribute.")
        if "reweighing" in mitigations and any(not MODEL_REGISTRY[m][2] for m in models):
            raise ValueError("K-nearest neighbors does not support sample weights. Remove reweighing or choose other models.")
    seed = spec.get("random_seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Random seed must be an integer between 0 and 4294967295.")
    for key, default in [("test_size", .2), ("validation_size", .2)]:
        value = spec.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not .1 <= value <= .4:
            raise ValueError(f"{key} must be between 0.10 and 0.40.")
    return {**spec, "random_seed": seed, "test_size": spec.get("test_size", .2),
            "validation_size": spec.get("validation_size", .2), "sensitive": sensitive,
            "mitigations": mitigations, "intersection": bool(spec.get("intersection", False))}


def prepare_experiment(frame: pd.DataFrame, spec: dict) -> dict:
    spec = validate_spec(spec, frame.columns.tolist())
    target, features, sensitive = spec["target"], spec["features"], spec["sensitive"]
    required = [target, *sensitive]
    selected = list(dict.fromkeys([*features, *required]))
    # Exact duplicate observations must not appear on both sides of a random split.
    data = frame.loc[:, selected].dropna(subset=required).drop_duplicates().reset_index(drop=True)
    if len(data) < 20:
        raise ValueError("At least 20 complete, distinct observations are required after cleaning.")
    if spec["task"] == "regression":
        try:
            y = pd.to_numeric(data[target], errors="raise").astype(float).to_numpy()
        except (ValueError, TypeError) as error:
            raise ValueError("Regression requires a finite numeric target. Choose classification for categories.") from error
        if not np.isfinite(y).all() or np.unique(y).size < 2:
            raise ValueError("Regression target needs at least two distinct finite numeric values.")
        classes = []
        positive = None
        stratify = None
    else:
        labels = data[target].astype(str)
        classes = sorted(labels.unique().tolist())
        if not 2 <= len(classes) <= 30:
            raise ValueError("Classification supports 2–30 classes; choose regression for a continuous target.")
        positive = str(spec.get("positive_label", ""))
        if positive not in classes:
            raise ValueError("Choose the positive / focus class from the target values.")
        classes = [label for label in classes if label != positive] + [positive]
        y = labels.map({label: i for i, label in enumerate(classes)}).to_numpy()
        stratify = y
        if spec["mitigations"] and len(classes) != 2:
            raise ValueError("Reweighing and threshold adjustment are supported only for binary classification.")
        if "threshold_adjustment" in spec["mitigations"]:
            group = data[spec["mitigation_attribute"]].astype(str)
            stratify = np.array([json.dumps([g, int(label)]) for g, label in zip(group, y)])
    groups = {column: data[column].astype(str).to_numpy() for column in sensitive}
    if spec["intersection"] and len(sensitive) > 1:
        name = " × ".join(sensitive)
        # JSON tuple encoding avoids collisions when a source value contains a delimiter.
        groups[name] = np.array([json.dumps(list(values), ensure_ascii=False)
                                for values in zip(*(groups[column] for column in sensitive))])
    if any(np.unique(values).size > 40 or np.unique(values).size < 2 for values in groups.values()):
        raise ValueError("Each sensitive view needs 2–40 groups. Bin continuous attributes into meaningful categories first.")
    indices = np.arange(len(data))
    try:
        remaining, test = train_test_split(indices, test_size=spec["test_size"], random_state=spec["random_seed"], stratify=stratify)
        fit, validation = train_test_split(remaining, test_size=spec["validation_size"], random_state=spec["random_seed"],
                                          stratify=None if stratify is None else np.asarray(stratify)[remaining])
    except ValueError as error:
        raise ValueError("Not enough observations per class/group for the requested train/validation/test split. Add data, change the split, or disable threshold adjustment.") from error
    if spec["task"] == "classification" and any(set(y[part]) != set(range(len(classes))) for part in [fit, validation, test]):
        raise ValueError("Every partition must contain all target classes. Add observations for rare classes.")
    if "threshold_adjustment" in spec["mitigations"]:
        mitigation_groups = groups[spec["mitigation_attribute"]]
        for part in [fit, validation]:
            if any(set(y[part][mitigation_groups[part] == group]) != {0, 1}
                   for group in np.unique(mitigation_groups)):
                raise ValueError("Threshold adjustment needs both classes in every mitigation group in training and validation.")
    feature_frame = data[features].copy()
    numeric = feature_frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numeric]
    # pandas nullable strings must become ordinary objects with np.nan for sklearn.
    for column in categorical:
        feature_frame[column] = feature_frame[column].astype(object).where(feature_frame[column].notna(), np.nan)
    if feature_frame.iloc[fit].isna().all().any():
        raise ValueError("A selected feature is entirely missing in training. Remove it or add observations.")
    upper_width = len(numeric) + sum(min(32, feature_frame.iloc[fit][c].nunique()) for c in categorical)
    if len(data) * upper_width > MAX_ENCODED_CELLS:
        raise ValueError("Encoded data exceeds the local memory limit. Select fewer features or upload fewer rows.")
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=32,
                                                           sparse_output=False))]), categorical),
    ])
    X_fit = transformer.fit_transform(feature_frame.iloc[fit])
    X_validation = transformer.transform(feature_frame.iloc[validation])
    X_test = transformer.transform(feature_frame.iloc[test])
    return dict(spec=spec, y=y, groups=groups, fit=fit, validation=validation, test=test,
                X_fit=X_fit, X_validation=X_validation, X_test=X_test, transformer=transformer,
                classes=classes, positive=positive,
                report=dict(input_rows=len(frame), rows_removed=len(frame) - len(data),
                            training_samples=len(fit), validation_samples=len(validation), test_samples=len(test),
                            feature_count=X_fit.shape[1], source_feature_count=len(features)))


def _performance(model, X, truth, predictions, task, classes, scores=None):
    if task == "regression":
        return dict(mae=float(mean_absolute_error(truth, predictions)),
                    rmse=float(np.sqrt(mean_squared_error(truth, predictions))),
                    r2=float(r2_score(truth, predictions)))
    average = "binary" if len(classes) == 2 else "macro"
    if scores is None:
        scores = model.predict_proba(X)
    try:
        auc = roc_auc_score(truth, scores[:, 1]) if len(classes) == 2 else roc_auc_score(truth, scores, multi_class="ovr", average="macro")
    except ValueError:
        auc = None
    return dict(accuracy=float(accuracy_score(truth, predictions)),
                precision=float(precision_score(truth, predictions, average=average, zero_division=0)),
                recall=float(recall_score(truth, predictions, average=average, zero_division=0)),
                f1=float(f1_score(truth, predictions, average=average, zero_division=0)), roc_auc=auc)


def run_experiment(frame: pd.DataFrame, spec: dict, directory: Path,
                   progress: Callable[[str], None] = lambda _: None) -> dict:
    from fairlearn.postprocessing import ThresholdOptimizer
    import sklearn
    import fairlearn

    progress("Validating data and creating train/validation/test partitions")
    prepared = prepare_experiment(frame, spec)
    spec = prepared["spec"]
    fit, validation, test = (prepared[k] for k in ["fit", "validation", "test"])
    y = prepared["y"]
    X_fit, X_validation, X_test = (prepared[k] for k in ["X_fit", "X_validation", "X_test"])
    performance, fairness, group_errors = [], [], []
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(prepared["transformer"], directory / "preprocessor.joblib")
    states = ["baseline", *spec["mitigations"]]
    for name in spec["models"]:
        baseline = None
        for state in states:
            progress(f"Training {MODEL_REGISTRY[name][0]} · {state}")
            estimator = build_estimator(name, spec["random_seed"])
            if state == "threshold_adjustment":
                estimator = ThresholdOptimizer(estimator=baseline, prefit=True, constraints="demographic_parity",
                                               objective="accuracy_score", predict_method="predict_proba")
                estimator.fit(X_validation, y[validation],
                              sensitive_features=prepared["groups"][spec["mitigation_attribute"]][validation])
                predictions = estimator.predict(X_test, sensitive_features=prepared["groups"][spec["mitigation_attribute"]][test],
                                                random_state=spec["random_seed"])
                scores = baseline.predict_proba(X_test)
            else:
                weights = None if state == "baseline" else compute_reweighing_weights(y[fit], prepared["groups"][spec["mitigation_attribute"]][fit])
                if weights is None:
                    estimator.fit(X_fit, y[fit])
                else:
                    estimator.fit(X_fit, y[fit], sample_weight=weights)
                predictions = estimator.predict(X_test)
                scores = None
                if state == "baseline":
                    baseline = estimator
            performance.append(dict(model=name, mitigation_state=state,
                                    **_performance(estimator, X_test, y[test], predictions, spec["task"], prepared["classes"], scores)))
            for attribute, values in prepared["groups"].items():
                if spec["task"] == "classification":
                    focus = len(prepared["classes"]) - 1
                    summary, details = evaluate_fairness((y[test] == focus).astype(int), (predictions == focus).astype(int), values[test])
                    # Rates with no denominator are undefined, not zero.
                    for detail in details:
                        members = values[test] == detail["group"]
                        if not np.any(y[test][members] == focus):
                            detail["tpr"] = None
                        if not np.any(y[test][members] != focus):
                            detail["fpr"] = None
                    if any(detail["tpr"] is None for detail in details):
                        summary["equal_opportunity_diff"] = None
                        summary["equalized_odds_diff"] = None
                    if any(detail["fpr"] is None for detail in details):
                        summary["equalized_odds_diff"] = None
                    if len(details) < 2:
                        for metric in ["demographic_parity_diff", "equal_opportunity_diff", "equalized_odds_diff", "disparate_impact_ratio"]:
                            summary[metric] = None
                    fairness.extend(dict(model=name, mitigation_state=state, sensitive_attribute=attribute, **summary, **detail) for detail in details)
                else:
                    for group in np.unique(values[test]):
                        mask = values[test] == group
                        group_errors.append(dict(model=name, mitigation_state=state, sensitive_attribute=attribute, group=group,
                                                 group_count=int(mask.sum()), low_sample_warning=bool(mask.sum() < 30),
                                                 mae=float(mean_absolute_error(y[test][mask], predictions[mask])),
                                                 rmse=float(np.sqrt(mean_squared_error(y[test][mask], predictions[mask]))),
                                                 mean_prediction=float(np.mean(predictions[mask]))))
            joblib.dump(estimator, directory / f"{state}_{name}.joblib")
    changes = calculate_fairness_changes(pd.DataFrame(fairness)).to_dict("records") if fairness and spec["mitigations"] else []
    config = {**spec, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "classes": prepared["classes"], "positive_label": prepared["positive"],
              "preprocessing_report": prepared["report"], "low_sample_threshold": 30,
              "model_names": {name: MODEL_REGISTRY[name][0] for name in spec["models"]},
              "model_parameters": {name: build_estimator(name, spec["random_seed"]).get_params() for name in spec["models"]},
              "dataset_sha256": hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest(),
              "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "fairlearn": fairlearn.__version__},
              "split_protocol": "Random row split; preprocessing and estimators fit only on training; thresholds fit only on validation; test used only for evaluation.",
              "categorical_encoding": "One-hot encoding, at most 32 categories per feature; infrequent categories pooled; unknowns safely handled."}
    result = dict(config=config, performance=performance, fairness=fairness, changes=changes, group_errors=group_errors)
    for key, rows in [("performance_metrics", performance), ("fairness_metrics", fairness),
                      ("fairness_changes", changes), ("group_errors", group_errors)]:
        empty_columns = {
            "fairness_metrics": ["model", "mitigation_state", "sensitive_attribute", "group", "group_count", "selection_rate", "tpr", "fpr", "demographic_parity_diff", "equal_opportunity_diff", "equalized_odds_diff", "disparate_impact_ratio"],
            "fairness_changes": ["model", "mitigation_state", "sensitive_attribute", "fairness_metric", "baseline_value", "mitigated_value", "raw_change", "distance_to_ideal_change", "effect"],
            "group_errors": ["model", "mitigation_state", "sensitive_attribute", "group", "group_count", "mae", "rmse", "mean_prediction", "low_sample_warning"],
        }
        table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=empty_columns.get(key, []))
        # Spreadsheet-safe exports: do not allow untrusted group names to become formulas.
        table = table.map(lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
        table.to_csv(directory / f"{key}.csv", index=False)
    return result
