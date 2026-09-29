"""File-backed results store shared by the batch pipeline and dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib
import pandas as pd

PERFORMANCE_FILE = "performance_metrics.csv"
FAIRNESS_FILE = "fairness_metrics.csv"
FAIRNESS_CHANGES_FILE = "fairness_changes.csv"
CONFIG_FILE = "run_config.json"


def save_results(results_dir: Path, performance_rows: list[dict[str, Any]],
                 fairness_rows: list[dict[str, Any]], fairness_change_rows: list[dict[str, Any]],
                 run_config: dict[str, Any]) -> None:
    """Persist actual measurement rows and the exact configuration used to create them."""
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(performance_rows).to_csv(results_dir / PERFORMANCE_FILE, index=False)
    pd.DataFrame(fairness_rows).to_csv(results_dir / FAIRNESS_FILE, index=False)
    pd.DataFrame(fairness_change_rows).to_csv(results_dir / FAIRNESS_CHANGES_FILE, index=False)
    (results_dir / CONFIG_FILE).write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")


def save_model(results_dir: Path, model: Any, model_name: str, mitigation_state: str) -> Path:
    """Save a fitted estimator/postprocessor for reproducibility, not dashboard execution."""
    model_dir = results_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / f"{mitigation_state}_{model_name}.joblib"
    joblib.dump(model, target)
    return target


def _read(results_dir: Path, filename: str) -> pd.DataFrame:
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError("Results are not available yet. Run `python run_pipeline.py` first.")
    return pd.read_csv(path)


def load_performance_metrics(results_dir: Path) -> pd.DataFrame:
    """Load precomputed performance measurements; never runs a model."""
    return _read(results_dir, PERFORMANCE_FILE)


def load_fairness_metrics(results_dir: Path) -> pd.DataFrame:
    """Load precomputed fairness measurements; never runs a model."""
    return _read(results_dir, FAIRNESS_FILE)


def load_fairness_changes(results_dir: Path) -> pd.DataFrame:
    """Load precomputed baseline-to-mitigation fairness comparisons."""
    return _read(results_dir, FAIRNESS_CHANGES_FILE)


def load_run_config(results_dir: Path) -> dict[str, Any]:
    """Load the configuration snapshot for a completed experiment."""
    path = results_dir / CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError("Run configuration is not available. Run `python run_pipeline.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))
