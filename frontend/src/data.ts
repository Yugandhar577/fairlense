export type MetricValue = number | null;
export const modelNames: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  decision_tree: "Decision Tree",
  random_forest: "Random Forest",
};
export const stateNames: Record<string, string> = {
  baseline: "Baseline",
  reweighing: "Reweighing",
  threshold_adjustment: "Threshold adjustment",
};
export const viewNames: Record<string, string> = {
  gender: "Gender",
  age_group: "Age group",
  gender_age_group: "Gender × age group",
};
export const metricNames = {
  demographic_parity_diff: "Demographic parity",
  equal_opportunity_diff: "Equal opportunity",
  equalized_odds_diff: "Equalized odds",
  disparate_impact_ratio: "Disparate impact",
};
export type FairnessKey = keyof typeof metricNames;
export const performanceNames = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1 score",
  roc_auc: "ROC-AUC",
};
export type PerformanceKey = keyof typeof performanceNames;
export const metricDescriptions: Record<FairnessKey, string> = {
  demographic_parity_diff:
    "The gap between the highest and lowest group selection rates. A smaller gap means more selection-rate parity.",
  equal_opportunity_diff:
    "The gap between group true-positive rates: how often each group’s actual positive cases are correctly identified.",
  equalized_odds_diff:
    "The larger of the true-positive-rate gap and false-positive-rate gap across groups.",
  disparate_impact_ratio:
    "The lowest group selection rate divided by the highest. Undefined when no group receives a positive prediction.",
};
export type Performance = { model: string; mitigation_state: string } & Record<
  PerformanceKey,
  MetricValue
>;
export type Fairness = {
  model: string;
  mitigation_state: string;
  sensitive_attribute: string;
  group: string;
  group_count: number;
  selection_rate: MetricValue;
  tpr: MetricValue;
  fpr: MetricValue;
  low_sample_warning: boolean;
  min_group_size: number;
  low_confidence_flag: boolean;
} & Record<FairnessKey, MetricValue>;
export interface Change {
  model: string;
  mitigation_state: string;
  sensitive_attribute: string;
  fairness_metric: FairnessKey;
  baseline_value: MetricValue;
  mitigated_value: MetricValue;
  raw_change: MetricValue;
  distance_to_ideal_change: MetricValue;
  effect: "improved" | "worsened" | "unchanged" | "undefined";
}
export interface RunConfig {
  random_seed: number;
  test_size: number;
  validation_size: number;
  created_at_utc: string;
  low_sample_threshold: number;
  threshold_constraint: string;
  sensitive_attributes: string[];
  models: string[];
  preprocessing_report: {
    input_rows: number;
    training_samples: number;
    test_samples: number;
    feature_count: number;
    rows_removed: number;
  };
}
export interface Results {
  performance: Performance[];
  fairness: Fairness[];
  changes: Change[];
  config: RunConfig;
}
export const modelColors: Record<string, string> = {
  logistic_regression: "#286f65",
  decision_tree: "#8a81bd",
  random_forest: "#d2a04e",
};
export const stateColors: Record<string, string> = {
  baseline: "#286f65",
  reweighing: "#90bda7",
  threshold_adjustment: "#b4a5d8",
};
export const valid = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
export const format = (value: MetricValue | undefined, percent = false) =>
  valid(value)
    ? percent
      ? `${(value * 100).toFixed(1)}%`
      : value.toFixed(3)
    : "Undefined";
export const signed = (value: MetricValue) =>
  valid(value) ? `${value > 0 ? "+" : ""}${value.toFixed(3)}` : "Undefined";
export const ideal = (metric: FairnessKey) =>
  metric === "disparate_impact_ratio" ? 1 : 0;
export const unique = (values: string[]) => [...new Set(values)];

export function summarize(rows: Fairness[]) {
  const seen = new Set<string>();
  return rows.filter((row) => {
    const key = JSON.stringify([
      row.model,
      row.mitigation_state,
      row.sensitive_attribute,
    ]);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export async function getResults(signal?: AbortSignal): Promise<Results> {
  const response = await fetch("/api/results", { signal });
  const body = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      body?.message ||
        "The results service could not be reached. Start python run_dashboard.py and try again.",
    );
  if (
    !body?.performance?.length ||
    !body?.fairness?.length ||
    !Array.isArray(body?.changes) ||
    !body?.config?.preprocessing_report
  ) {
    throw new Error(
      "The results store is incomplete. Run python run_pipeline.py and refresh.",
    );
  }
  return body;
}
