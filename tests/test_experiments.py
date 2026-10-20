"""Exercise dataset-independent tasks, real model training, and leakage boundaries."""
import json
import time

import numpy as np
import pandas as pd
import pytest

from fairlens.experiments import dataset_profile, parse_csv, prepare_experiment, run_experiment, validate_spec
from fairlens.models.registry import MODEL_REGISTRY
from fairlens.workspace import Workspace, read_json, write_json


@pytest.fixture
def frame():
    i = np.arange(180)
    return pd.DataFrame({"value": i / 10, "category": np.where(i % 3, "urban", "rural"),
                         "group": np.where(i % 2, "B", "A"),
                         "age_band": np.where(i % 5, "younger", "older"),
                         "outcome": np.where(i % 4 < 2, "no", "yes")})


def specification(**overrides):
    return dict(task="classification", target="outcome", positive_label="yes",
                features=["value", "category"], sensitive=["group", "age_band"],
                intersection=True, models=["logistic_regression", "decision_tree"],
                mitigations=[], random_seed=42, test_size=.2, validation_size=.2,
                **overrides)


def test_csv_profile_and_invalid_uploads(frame):
    parsed = parse_csv(frame.to_csv(index=False))
    profile = dataset_profile(parsed, "example", "example.csv")
    assert profile["rows"] == 180
    assert profile["columns"][-1]["values"] == ["no", "yes"]
    for content in ["", "a,a\n" + "1,2\n" * 30, "a,b\n1,2\n", "a,b\n" + "1,inf\n" * 30]:
        with pytest.raises(ValueError):
            parse_csv(content)
    for content in ["a,b\n" + "1,2,3\n" * 30, "a,b\n" + "1,\x00\n" * 30]:
        with pytest.raises(ValueError):
            parse_csv(content)


def test_preprocessor_fits_only_training_and_partitions_are_disjoint(frame):
    result = prepare_experiment(frame, specification())
    fit, validation, test = [set(result[key]) for key in ["fit", "validation", "test"]]
    assert not fit & test and not fit & validation and not validation & test
    assert fit | validation | test == set(range(len(frame)))
    scaler = result["transformer"].named_transformers_["numeric"].named_steps["scale"]
    assert np.isclose(scaler.mean_[0], frame.iloc[result["fit"]]["value"].mean())
    assert result["report"]["training_samples"] + result["report"]["validation_samples"] + result["report"]["test_samples"] == 180
    assert "group" not in result["transformer"].feature_names_in_


@pytest.mark.parametrize("model", list(MODEL_REGISTRY))
def test_every_registered_model_trains_and_uses_task_specific_metrics(frame, tmp_path, model):
    spec = specification()
    spec["models"] = [model]
    if MODEL_REGISTRY[model][1] == "regression":
        spec["task"] = "regression"
        frame["outcome"] = frame["value"] * 2 + np.arange(len(frame)) % 3
    result = run_experiment(frame, spec, tmp_path)
    assert len(result["performance"]) == 1
    assert result["performance"][0]["model"] == model
    if spec["task"] == "regression":
        assert {"mae", "rmse", "r2"}.issubset(result["performance"][0])
        assert "accuracy" not in result["performance"][0]
        assert result["group_errors"] and not result["fairness"]
    else:
        assert {"accuracy", "precision", "recall", "f1", "roc_auc"}.issubset(result["performance"][0])
        assert result["fairness"] and not result["group_errors"]
        assert {r["sensitive_attribute"] for r in result["fairness"]} == {"group", "age_band", "group × age_band"}
    assert (tmp_path / "preprocessor.joblib").exists()
    assert result["config"]["model_parameters"][model]


def test_multiclass_and_no_sensitive_columns_are_supported(frame, tmp_path):
    frame["outcome"] = np.array(["red", "blue", "green"] * 60)
    spec = specification()
    spec.update(positive_label="red", sensitive=[], intersection=False)
    result = run_experiment(frame, spec, tmp_path)
    assert len(result["config"]["classes"]) == 3
    assert not result["fairness"]
    assert len(result["performance"]) == 2
    assert result["performance"][0]["roc_auc"] is not None
    spec["mitigations"] = ["reweighing"]
    spec["sensitive"] = ["group"]
    spec["mitigation_attribute"] = "group"
    with pytest.raises(ValueError, match="binary"):
        prepare_experiment(frame, spec)


def test_mitigation_produces_real_changes_and_reuses_baseline_scores(frame, tmp_path):
    spec = specification()
    spec.update(models=["logistic_regression"], mitigations=["reweighing", "threshold_adjustment"], mitigation_attribute="group")
    result = run_experiment(frame, spec, tmp_path)
    assert len(result["performance"]) == 3
    assert len(result["changes"]) == 24  # 2 interventions × 3 views × 4 metrics
    assert result["performance"][0]["roc_auc"] == result["performance"][2]["roc_auc"]


def test_invalid_model_target_feature_combinations_are_rejected(frame):
    for patch in [dict(models=["linear_regression", "logistic_regression"]),
                  dict(models=["decision_tree", "decision_tree"]),
                  dict(features=["outcome"]), dict(features=["value", "group"]),
                  dict(random_seed=-1), dict(test_size=.9), dict(target="missing"),
                  dict(models=["knn_classifier"], mitigations=["reweighing"], mitigation_attribute="group")]:
        spec = specification(); spec.update(patch)
        with pytest.raises(ValueError):
            validate_spec(spec, frame.columns.tolist())
    spec = specification(); spec.update(task="regression", models=["linear_regression"])
    with pytest.raises(ValueError, match="numeric"):
        prepare_experiment(frame, spec)


def test_training_is_reproducible_and_missing_values_do_not_leak(frame, tmp_path):
    frame.loc[::11, "value"] = np.nan
    frame.loc[::13, "category"] = None
    first = run_experiment(frame, specification(), tmp_path / "a")
    second = run_experiment(frame, specification(), tmp_path / "b")
    assert first["performance"] == second["performance"]
    assert first["config"]["dataset_sha256"] == second["config"]["dataset_sha256"]


def test_workspace_isolates_runs_and_restores_completed_history(frame, tmp_path):
    workspace = Workspace(tmp_path)
    profile = workspace.upload(dict(name="demo.csv", csv=frame.to_csv(index=False)))
    spec = specification(); spec["dataset_id"] = profile["id"]
    run = workspace.submit(spec)
    deadline = time.monotonic() + 20
    while workspace.status(run["id"])["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(.05)
    assert workspace.status(run["id"])["status"] == "completed"
    restored = Workspace(tmp_path)
    assert restored.datasets()[0]["id"] == profile["id"]
    assert len(restored.result(run["id"])["performance"]) == 2
    assert b"accuracy" in restored.export(run["id"], "performance_metrics.csv")
    with pytest.raises(ValueError):
        restored.export(run["id"], "baseline_logistic_regression.joblib")
    with pytest.raises(ValueError):
        restored.profile("../../private")


def test_interrupted_runs_are_marked_failed(tmp_path):
    folder = tmp_path / "runs" / ("a" * 32)
    folder.mkdir(parents=True)
    (folder / "status.json").write_text(json.dumps(dict(status="running")))
    workspace = Workspace(tmp_path)
    assert workspace.status("a" * 32)["status"] == "failed"


def test_missing_denominators_remain_undefined_and_exports_are_safe(frame, tmp_path):
    frame["group"] = np.where(frame["outcome"] == "yes", "=positive", "negative")
    spec = specification(); spec.update(sensitive=["group"], intersection=False, models=["decision_tree"])
    result = run_experiment(frame, spec, tmp_path)
    assert all(row["equalized_odds_diff"] is None for row in result["fairness"])
    positive = next(row for row in result["fairness"] if row["group"] == "=positive")
    assert positive["fpr"] is None
    assert "'=positive" in (tmp_path / "fairness_metrics.csv").read_text()


def test_failed_jobs_and_single_worker_are_explicit(frame, tmp_path):
    workspace = Workspace(tmp_path)
    profile = workspace.upload(dict(name="demo.csv", csv=frame.to_csv(index=False)))
    spec = specification(); spec.update(dataset_id=profile["id"], task="regression", models=["linear_regression"])
    workspace.active = True
    with pytest.raises(RuntimeError, match="already training"):
        workspace.submit(spec)
    workspace.active = False
    run = workspace.submit(spec)
    deadline = time.monotonic() + 10
    while workspace.status(run["id"])["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(.05)
    assert workspace.status(run["id"])["status"] == "failed"
    assert "numeric target" in workspace.status(run["id"])["message"]
    with pytest.raises(ValueError, match="successfully"):
        workspace.result(run["id"])


def test_progress_snapshots_support_concurrent_readers_and_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / "status.json"
    write_json(path, {"step": 0})
    def writer():
        for index in range(100):
            write_json(path, {"step": index})
    def reader():
        for _ in range(300):
            assert isinstance(read_json(path)["step"], int)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(writer), *(pool.submit(reader) for _ in range(3))]
        for future in futures:
            future.result()


def test_documented_demo_configurations_work(tmp_path):
    from pathlib import Path
    examples = Path(__file__).resolve().parents[1] / "examples"
    for task, filename, target, features, models in [
        ("classification", "classification_demo.csv", "outcome", ["experience", "qualification"], ["logistic_regression", "decision_tree"]),
        ("regression", "regression_demo.csv", "price", ["area", "rooms"], ["linear_regression", "ridge"]),
    ]:
        data = parse_csv((examples / filename).read_text())
        spec = specification()
        spec.update(task=task, target=target, features=features, models=models,
                    sensitive=["region"], intersection=False)
        if task == "classification":
            spec.update(mitigations=["reweighing", "threshold_adjustment"], mitigation_attribute="region")
        result = run_experiment(data, spec, tmp_path / task)
        assert len(result["performance"]) == (6 if task == "classification" else 2)
