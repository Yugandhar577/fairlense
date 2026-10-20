import type { Change, Fairness } from "./data";

export interface ModelOption {
  id: string;
  name: string;
  task: string;
  sample_weight: boolean;
}
export interface DatasetColumn {
  name: string;
  kind: string;
  missing: number;
  unique: number;
  values: string[];
}
export interface Dataset {
  id: string;
  name: string;
  rows: number;
  columns: DatasetColumn[];
}
export interface Run {
  id: string;
  dataset_id: string;
  dataset_name: string;
  task: string;
  models: string[];
  status: string;
  message: string;
  created_at_utc: string;
}
export interface ExperimentResult {
  config: {
    dataset_name: string;
    task: string;
    target: string;
    positive_label: string | null;
    classes: string[];
    features: string[];
    sensitive: string[];
    mitigation_attribute?: string;
    mitigations: string[];
    models: string[];
    random_seed: number;
    test_size: number;
    validation_size: number;
    created_at_utc: string;
    split_protocol: string;
    categorical_encoding: string;
    model_names: Record<string, string>;
    versions: Record<string, string>;
    dataset_sha256: string;
    preprocessing_report: {
      input_rows: number;
      training_samples: number;
      validation_samples: number;
      test_samples: number;
      rows_removed: number;
      feature_count: number;
    };
  };
  performance: ({ model: string; mitigation_state: string } & Record<
    string,
    string | number | null
  >)[];
  fairness: Fairness[];
  changes: Change[];
  group_errors: {
    model: string;
    mitigation_state: string;
    sensitive_attribute: string;
    group: string;
    group_count: number;
    mae: number;
    rmse: number;
    mean_prediction: number;
    low_sample_warning: boolean;
  }[];
}

export async function workspaceRequest<T>(
  path: string,
  payload?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`/api/workspace/${path}`, {
    signal,
    ...(payload === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
  });
  const body = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      body?.message ||
        "The local training service is unavailable. Start python run_dashboard.py.",
    );
  return body;
}
