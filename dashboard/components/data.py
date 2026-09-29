"""Cached dashboard access to precomputed results; this module never trains models."""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fairlens.results_store import load_fairness_changes, load_fairness_metrics, load_performance_metrics, load_run_config


def result_paths():
    return ROOT / "results"


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load CSV/JSON outputs from the offline experiment."""
    directory = result_paths()
    return load_performance_metrics(directory), load_fairness_metrics(directory), load_run_config(directory)


def load_cross_group_effects() -> pd.DataFrame:
    """Load precomputed cross-group effects; this never executes mitigation."""
    return load_fairness_changes(result_paths())
