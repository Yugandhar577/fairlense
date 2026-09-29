"""Loading utilities for the UCI Adult/Census Income data."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load an Adult CSV, supporting both headered CSVs and UCI's headerless format.

    Leading/trailing whitespace is removed and the dataset's ``?`` missing-value marker
    is converted to a proper missing value before any preprocessing occurs.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Adult dataset was not found at {data_path}. Place adult.csv there before running the pipeline."
        )
    raw = pd.read_csv(data_path, skipinitialspace=True, na_values="?", keep_default_na=True)
    if not set(ADULT_COLUMNS).issubset(raw.columns):
        raw = pd.read_csv(data_path, header=None, names=ADULT_COLUMNS, skipinitialspace=True,
                          na_values="?", keep_default_na=True)
    raw.columns = [str(column).strip() for column in raw.columns]
    for column in raw.select_dtypes(include="object").columns:
        raw[column] = raw[column].str.strip().replace({"?": pd.NA})
    # adult.test has a harmless leading metadata line and labels ending with a period.
    raw = raw[raw["age"].astype(str).str.fullmatch(r"\d+")].copy()
    raw["age"] = pd.to_numeric(raw["age"], errors="coerce")
    raw["income"] = raw["income"].astype("string").str.rstrip(".")
    return raw.reset_index(drop=True)
