"""Derived comparisons for detecting cross-group mitigation side effects."""
from __future__ import annotations

import math
import pandas as pd

FAIRNESS_SUMMARY_METRICS = (
    "demographic_parity_diff",
    "equal_opportunity_diff",
    "equalized_odds_diff",
    "disparate_impact_ratio",
)
LOWER_IS_BETTER = {
    "demographic_parity_diff",
    "equal_opportunity_diff",
    "equalized_odds_diff",
}


def _effect(metric: str, baseline: float, mitigated: float) -> tuple[float, str]:
    """Return signed movement toward the metric ideal and a descriptive label.

    For disparity differences, the ideal is zero. For disparate impact, the
    ideal is one; its raw ratio change is therefore not itself an improvement
    score. Undefined values remain undefined rather than being inferred.
    """
    if pd.isna(baseline) or pd.isna(mitigated):
        return float("nan"), "undefined"
    if metric in LOWER_IS_BETTER:
        distance_change = abs(float(mitigated)) - abs(float(baseline))
    else:
        distance_change = abs(1.0 - float(mitigated)) - abs(1.0 - float(baseline))
    if math.isclose(distance_change, 0.0, abs_tol=1e-12):
        return distance_change, "unchanged"
    return distance_change, "improved" if distance_change < 0 else "worsened"


def calculate_fairness_changes(fairness_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare every mitigated fairness summary against the matching baseline.

    The returned data makes cross-group effects explicit: a mitigation trained
    for gender can be inspected for its separate age-group and intersectional
    outcomes. It uses measured summary statistics only and performs no fitting.
    """
    required = {"model", "mitigation_state", "sensitive_attribute", *FAIRNESS_SUMMARY_METRICS}
    missing = required.difference(fairness_metrics.columns)
    if missing:
        raise ValueError(f"Fairness metrics are missing required columns: {sorted(missing)}")

    key_columns = ["model", "mitigation_state", "sensitive_attribute"]
    summaries = fairness_metrics.loc[:, [*key_columns, *FAIRNESS_SUMMARY_METRICS]].drop_duplicates(key_columns)
    baseline = summaries[summaries["mitigation_state"] == "baseline"].drop(columns="mitigation_state")
    baseline = baseline.rename(columns={metric: f"baseline_{metric}" for metric in FAIRNESS_SUMMARY_METRICS})
    mitigated = summaries[summaries["mitigation_state"] != "baseline"]
    joined = mitigated.merge(baseline, on=["model", "sensitive_attribute"], how="left", validate="many_to_one")

    rows: list[dict[str, object]] = []
    for _, entry in joined.iterrows():
        for metric in FAIRNESS_SUMMARY_METRICS:
            baseline_value = entry[f"baseline_{metric}"]
            mitigated_value = entry[metric]
            distance_change, effect = _effect(metric, baseline_value, mitigated_value)
            rows.append({
                "model": entry["model"],
                "mitigation_state": entry["mitigation_state"],
                "sensitive_attribute": entry["sensitive_attribute"],
                "fairness_metric": metric,
                "baseline_value": baseline_value,
                "mitigated_value": mitigated_value,
                "raw_change": (mitigated_value - baseline_value)
                if pd.notna(baseline_value) and pd.notna(mitigated_value) else float("nan"),
                "distance_to_ideal_change": distance_change,
                "effect": effect,
            })
    return pd.DataFrame(rows)
