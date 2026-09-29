import math
import numpy as np
import pandas as pd

from fairlens.evaluation.comparison import calculate_fairness_changes
from fairlens.evaluation.fairness import evaluate_fairness


def test_hand_calculated_group_fairness_metrics():
    # A: selection=1, TPR=1, FPR=1; B: selection=0, TPR=0, FPR=0.
    y_true = [1, 0, 1, 0]
    y_pred = [1, 1, 0, 0]
    summary, details = evaluate_fairness(y_true, y_pred, ["A", "A", "B", "B"], low_sample_threshold=3)
    assert summary["demographic_parity_diff"] == 1.0
    assert summary["equal_opportunity_diff"] == 1.0
    assert summary["equalized_odds_diff"] == 1.0
    assert summary["disparate_impact_ratio"] == 0.0
    assert summary["low_confidence_flag"] is True
    assert {row["group_count"] for row in details} == {2}


def test_disparate_impact_is_undefined_when_every_selection_rate_is_zero():
    summary, _ = evaluate_fairness([0, 1, 0, 1], [0, 0, 0, 0], ["A", "A", "B", "B"])
    assert math.isnan(summary["disparate_impact_ratio"])


def test_fairness_change_analysis_identifies_cross_group_side_effects():
    rows = []
    for attribute, baseline_dp, mitigated_dp, baseline_di, mitigated_di in [
        ("gender", 0.20, 0.10, 0.60, 0.80),
        ("age_group", 0.10, 0.25, 0.90, 0.70),
        ("gender_age_group", 0.30, 0.35, 0.50, 0.45),
    ]:
        for state, dp, di in [("baseline", baseline_dp, baseline_di), ("reweighing", mitigated_dp, mitigated_di)]:
            # Production fairness data repeats summaries once per group. The
            # comparison must deduplicate these rows safely.
            for group in ("first", "second"):
                rows.append({
                    "model": "model_a", "mitigation_state": state, "sensitive_attribute": attribute,
                    "group": group, "demographic_parity_diff": dp,
                    "equal_opportunity_diff": dp, "equalized_odds_diff": dp,
                    "disparate_impact_ratio": di,
                })
    changes = calculate_fairness_changes(pd.DataFrame(rows))
    gender_dp = changes[(changes.sensitive_attribute == "gender") & (changes.fairness_metric == "demographic_parity_diff")].iloc[0]
    age_dp = changes[(changes.sensitive_attribute == "age_group") & (changes.fairness_metric == "demographic_parity_diff")].iloc[0]
    age_di = changes[(changes.sensitive_attribute == "age_group") & (changes.fairness_metric == "disparate_impact_ratio")].iloc[0]
    assert gender_dp.effect == "improved"
    assert age_dp.effect == "worsened"
    assert age_di.effect == "worsened"  # 0.70 is farther from the DI ideal of 1 than 0.90
