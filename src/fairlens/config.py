"""Central, reproducible configuration for FairLens."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FairLensConfig:
    random_seed: int = 42
    test_size: float = 0.20
    validation_size: float = 0.20
    data_path: Path = ROOT_DIR / "data" / "raw" / "adult.csv"
    results_dir: Path = ROOT_DIR / "results"
    include_sensitive_as_feature: bool = False
    threshold_constraint: str = "demographic_parity"
    low_sample_threshold: int = 30
    age_bins: tuple[int, ...] = (0, 25, 45, 65, 120)
    age_labels: tuple[str, ...] = ("<25", "25-44", "45-64", "65+")
    # ``gender_age_group`` is an evaluation-only intersectional view. It is
    # excluded from predictive features by default alongside its components.
    sensitive_attributes: tuple[str, ...] = ("gender", "age_group", "gender_age_group")
    target_column: str = "income"
    models: tuple[str, ...] = ("logistic_regression", "decision_tree", "random_forest")
    mitigations: tuple[str, ...] = ("reweighing", "threshold_adjustment")

    def serializable(self) -> dict:
        """Return JSON-safe configuration values."""
        values = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


DEFAULT_CONFIG = FairLensConfig()

# Public constant aliases retained for simple scripts, notebooks, and traceability
# to the documented experimental configuration.
RANDOM_SEED = DEFAULT_CONFIG.random_seed
TEST_SIZE = DEFAULT_CONFIG.test_size
DATA_PATH = DEFAULT_CONFIG.data_path
AGE_BINS = DEFAULT_CONFIG.age_bins
AGE_LABELS = DEFAULT_CONFIG.age_labels
SENSITIVE_ATTRIBUTES = DEFAULT_CONFIG.sensitive_attributes
TARGET_COLUMN = DEFAULT_CONFIG.target_column
MODELS = DEFAULT_CONFIG.models
MITIGATIONS = DEFAULT_CONFIG.mitigations
LOW_SAMPLE_THRESHOLD = DEFAULT_CONFIG.low_sample_threshold
